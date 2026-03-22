from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch import nn


def _looks_like_router_output(tensor: torch.Tensor) -> bool:
    if tensor.ndim < 2:
        return False
    num_experts = tensor.shape[-1]
    return 2 <= num_experts <= 512


def _to_logits(output) -> Optional[torch.Tensor]:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        float_tensors = []
        any_tensors = []
        for item in output:
            if isinstance(item, torch.Tensor):
                any_tensors.append(item)
                if item.is_floating_point():
                    float_tensors.append(item)
        if float_tensors:
            return float_tensors[0]
        if any_tensors:
            return any_tensors[0]
    if isinstance(output, dict):
        for key in ("router_logits", "gate_logits", "logits", "scores"):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value
        float_tensors = []
        any_tensors = []
        for value in output.values():
            if isinstance(value, torch.Tensor):
                any_tensors.append(value)
                if value.is_floating_point():
                    float_tensors.append(value)
        if float_tensors:
            return float_tensors[0]
        if any_tensors:
            return any_tensors[0]
    for key in ("router_logits", "gate_logits", "logits", "scores"):
        value = getattr(output, key, None)
        if isinstance(value, torch.Tensor):
            return value
    return None


def _expert_activation_metrics(dist: torch.Tensor) -> Dict[str, object]:
    dist = dist.float()
    num_experts = int(dist.numel())
    if num_experts == 0:
        return {
            "num_experts": 0,
            "expert_usage_entropy": 0.0,
            "expert_usage_entropy_normalized": 0.0,
            "effective_expert_count": 0.0,
            "effective_expert_ratio": 0.0,
            "active_expert_count_ge_1pct": 0,
            "active_expert_ratio_ge_1pct": 0.0,
            "active_expert_count_ge_0p1pct": 0,
            "active_expert_ratio_ge_0p1pct": 0.0,
            "top2_expert_share": 0.0,
            "top4_expert_share": 0.0,
            "gini_coefficient": 0.0,
            "load_balance_cv": 0.0,
        }

    eps = 1e-12
    entropy = -(dist * torch.log(dist.clamp_min(eps))).sum()
    entropy_val = float(entropy.item())
    if num_experts > 1:
        entropy_norm = entropy_val / float(torch.log(torch.tensor(float(num_experts))).item())
    else:
        entropy_norm = 0.0

    topk2 = min(2, num_experts)
    topk4 = min(4, num_experts)
    top2_share = float(torch.topk(dist, k=topk2).values.sum().item())
    top4_share = float(torch.topk(dist, k=topk4).values.sum().item())

    active_1pct = int((dist >= 0.01).sum().item())
    active_0p1pct = int((dist >= 0.001).sum().item())

    sorted_dist, _ = torch.sort(dist)
    idx = torch.arange(
        1,
        num_experts + 1,
        dtype=sorted_dist.dtype,
        device=sorted_dist.device,
    )
    denom = sorted_dist.sum().clamp_min(eps)
    gini = (2.0 * (idx * sorted_dist).sum() / (num_experts * denom)) - (
        (num_experts + 1) / num_experts
    )
    gini = gini.clamp(min=0.0, max=1.0)

    mean_load = dist.mean().clamp_min(eps)
    load_balance_cv = float((dist.std(unbiased=False) / mean_load).item())

    effective_experts = float(torch.exp(entropy).item())
    return {
        "num_experts": num_experts,
        "expert_usage_entropy": entropy_val,
        "expert_usage_entropy_normalized": float(entropy_norm),
        "effective_expert_count": effective_experts,
        "effective_expert_ratio": float(effective_experts / float(num_experts)),
        "active_expert_count_ge_1pct": active_1pct,
        "active_expert_ratio_ge_1pct": float(active_1pct / float(num_experts)),
        "active_expert_count_ge_0p1pct": active_0p1pct,
        "active_expert_ratio_ge_0p1pct": float(active_0p1pct / float(num_experts)),
        "top2_expert_share": top2_share,
        "top4_expert_share": top4_share,
        "gini_coefficient": float(gini.item()),
        "load_balance_cv": load_balance_cv,
    }


def _expert_frequency_mapping(dist: torch.Tensor) -> Dict[str, float]:
    return {str(i): float(v.item()) for i, v in enumerate(dist)}


@dataclass
class ModuleStats:
    token_count: int = 0
    entropy_sum: float = 0.0
    expert_counts: Optional[torch.Tensor] = None


@dataclass
class RouterStatsTracker:
    model: nn.Module
    name_patterns: List[str]
    handles: List = field(default_factory=list)
    stats: Dict[str, ModuleStats] = field(default_factory=lambda: defaultdict(ModuleStats))

    def _match(self, module_name: str, module_obj: nn.Module) -> bool:
        lname = module_name.lower()
        for p in self.name_patterns:
            lp = p.lower()
            # Config patterns are often parameter-level names (e.g. "mlp.gate.weight"),
            # while hooks are registered on module names (e.g. "mlp.gate").
            if lp in lname:
                return True
            if lp.endswith(".weight") and lp[:-7] in lname:
                return True
            if lp.endswith(".bias") and lp[:-5] in lname:
                return True
            if lp in f"{lname}.weight" or lp in f"{lname}.bias":
                return True
        cls_name = module_obj.__class__.__name__.lower()
        return any(p.lower() in cls_name for p in self.name_patterns)

    def register(self) -> int:
        count = 0
        for name, module in self.model.named_modules():
            if not self._match(name, module):
                continue
            handle = module.register_forward_hook(self._build_hook(name))
            self.handles.append(handle)
            count += 1
        return count

    def clear(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def reset(self) -> None:
        self.stats = defaultdict(ModuleStats)

    def _build_hook(self, module_name: str):
        def hook(_, __, output):
            logits = _to_logits(output)
            if logits is None or not isinstance(logits, torch.Tensor):
                return
            if not _looks_like_router_output(logits):
                return

            with torch.no_grad():
                flat = logits.float().reshape(-1, logits.shape[-1])
                probs = torch.softmax(flat, dim=-1)
                top_idx = torch.argmax(probs, dim=-1)
                counts = torch.bincount(top_idx, minlength=probs.shape[-1]).cpu()
                entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1).mean().item()

                entry = self.stats[module_name]
                entry.token_count += flat.shape[0]
                entry.entropy_sum += entropy * flat.shape[0]
                if entry.expert_counts is None:
                    entry.expert_counts = counts
                else:
                    entry.expert_counts += counts

        return hook

    def summary(self) -> Dict[str, Dict[str, object]]:
        result: Dict[str, Dict[str, object]] = {}
        for module_name, st in self.stats.items():
            if st.expert_counts is None or st.token_count == 0:
                continue
            dist = st.expert_counts.float() / float(st.expert_counts.sum().item())
            top_share = float(dist.max().item())
            activation_metrics = _expert_activation_metrics(dist)
            result[module_name] = {
                "token_count": st.token_count,
                "mean_entropy": st.entropy_sum / float(st.token_count),
                "top_expert_share": top_share,
                "expert_distribution": dist.tolist(),
                "expert_activation_frequency": dist.tolist(),
                "expert_activation_frequency_by_expert": _expert_frequency_mapping(dist),
                **activation_metrics,
            }
        return result


def aggregate_router_summaries(summaries: List[Dict[str, Dict[str, object]]]) -> Dict[str, Dict[str, object]]:
    aggregate: Dict[str, Dict[str, object]] = {}
    for summary in summaries:
        for module_name, entry in summary.items():
            target = aggregate.setdefault(
                module_name,
                {
                    "token_count": 0,
                    "entropy_weighted_sum": 0.0,
                    "expert_distribution_sum": None,
                },
            )
            token_count = int(entry["token_count"])
            target["token_count"] += token_count
            target["entropy_weighted_sum"] += float(entry["mean_entropy"]) * token_count

            distribution = torch.tensor(entry["expert_distribution"], dtype=torch.float32)
            if target["expert_distribution_sum"] is None:
                target["expert_distribution_sum"] = distribution * token_count
            else:
                target["expert_distribution_sum"] += distribution * token_count

    final: Dict[str, Dict[str, object]] = {}
    for module_name, data in aggregate.items():
        token_count = data["token_count"]
        if token_count == 0:
            continue
        dist_sum = data["expert_distribution_sum"] / float(token_count)
        activation_metrics = _expert_activation_metrics(dist_sum)
        final[module_name] = {
            "token_count": token_count,
            "mean_entropy": data["entropy_weighted_sum"] / float(token_count),
            "top_expert_share": float(dist_sum.max().item()),
            "expert_distribution": dist_sum.tolist(),
            "expert_activation_frequency": dist_sum.tolist(),
            "expert_activation_frequency_by_expert": _expert_frequency_mapping(dist_sum),
            **activation_metrics,
        }
    return final
