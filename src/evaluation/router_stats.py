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
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
    if isinstance(output, dict):
        for key in ("router_logits", "gate_logits", "logits", "scores"):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value
    return None


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
        if any(p.lower() in lname for p in self.name_patterns):
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
            result[module_name] = {
                "token_count": st.token_count,
                "mean_entropy": st.entropy_sum / float(st.token_count),
                "top_expert_share": top_share,
                "expert_distribution": dist.tolist(),
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
        final[module_name] = {
            "token_count": token_count,
            "mean_entropy": data["entropy_weighted_sum"] / float(token_count),
            "top_expert_share": float(dist_sum.max().item()),
            "expert_distribution": dist_sum.tolist(),
        }
    return final
