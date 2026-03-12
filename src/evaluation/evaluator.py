from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Subset

from src.evaluation.encoder_metrics import compute_group_compactness, compute_pair_metrics
from src.evaluation.router_stats import RouterStatsTracker, aggregate_router_summaries
from src.utils.distributed import all_gather_object, get_rank, get_world_size, is_main_process


def build_sharded_eval_loader(
    dataset,
    collator,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    rank = get_rank()
    world_size = get_world_size()
    indices = list(range(rank, len(dataset), world_size))
    sharded = Subset(dataset, indices)
    return DataLoader(
        sharded,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collator,
    )


def _move_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


@torch.no_grad()
def evaluate_text_encoder(
    model,
    dataloader: DataLoader,
    device: torch.device,
    router_tracker: Optional[RouterStatsTracker] = None,
    use_bf16: bool = False,
    use_fp16: bool = False,
) -> Dict[str, object]:
    model.eval()

    local_anchor = []
    local_positive = []
    local_sample_idx = []
    local_group_id: List[Optional[str]] = []
    local_prompt_type: List[Optional[str]] = []

    amp_dtype = None
    if use_bf16:
        amp_dtype = torch.bfloat16
    elif use_fp16:
        amp_dtype = torch.float16

    for batch in dataloader:
        batch = _move_to_device(batch, device)
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if (amp_dtype is not None and device.type == "cuda")
            else nullcontext()
        )
        with autocast_ctx:
            out_a = model(
                input_ids=batch["anchor_input_ids"],
                attention_mask=batch["anchor_attention_mask"],
                output_hidden_states=True,
            )
            out_p = model(
                input_ids=batch["positive_input_ids"],
                attention_mask=batch["positive_attention_mask"],
                output_hidden_states=True,
            )

        local_anchor.append(out_a["sentence_embeddings"].detach().float().cpu())
        local_positive.append(out_p["sentence_embeddings"].detach().float().cpu())
        local_sample_idx.extend(batch["sample_idx"].detach().cpu().tolist())
        local_group_id.extend(batch["group_id"])
        local_prompt_type.extend(batch["prompt_type"])

    local_payload = {
        "anchor_embeddings": torch.cat(local_anchor, dim=0) if local_anchor else None,
        "positive_embeddings": torch.cat(local_positive, dim=0) if local_positive else None,
        "sample_idx": local_sample_idx,
        "group_id": local_group_id,
        "prompt_type": local_prompt_type,
        "router_summary": router_tracker.summary() if router_tracker is not None else {},
    }
    gathered = all_gather_object(local_payload)

    if not is_main_process():
        return {}

    anchors = []
    positives = []
    sample_idx = []
    group_id = []
    prompt_type = []
    router_summaries = []
    for item in gathered:
        router_summaries.append(item["router_summary"])
        anchor_item = item["anchor_embeddings"]
        positive_item = item["positive_embeddings"]
        if anchor_item is None or positive_item is None:
            continue
        if anchor_item.numel() == 0 or positive_item.numel() == 0:
            continue
        anchors.append(anchor_item)
        positives.append(positive_item)
        sample_idx.extend(item["sample_idx"])
        group_id.extend(item["group_id"])
        prompt_type.extend(item["prompt_type"])

    if not sample_idx:
        return {
            "metrics": {},
            "routing": aggregate_router_summaries(router_summaries),
            "sample_count": 0,
            "prompt_type_distribution": {},
        }

    anchors = torch.cat(anchors, dim=0)
    positives = torch.cat(positives, dim=0)

    order = torch.tensor(sample_idx).argsort().tolist()
    anchors = anchors[order]
    positives = positives[order]
    group_id = [group_id[i] for i in order]
    prompt_type = [prompt_type[i] for i in order]

    metrics = compute_pair_metrics(anchors, positives)
    metrics.update(compute_group_compactness(anchors, group_id))
    routing = aggregate_router_summaries(router_summaries)

    return {
        "metrics": metrics,
        "routing": routing,
        "sample_count": len(sample_idx),
        "prompt_type_distribution": _count_prompt_types(prompt_type),
    }


def _count_prompt_types(prompt_types: List[Optional[str]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for t in prompt_types:
        key = "unknown" if t is None else str(t)
        counts[key] = counts.get(key, 0) + 1
    return counts
