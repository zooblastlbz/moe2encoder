from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.autograd import Function

from src.utils.distributed import get_rank, is_dist_available


def get_dp_group():
    try:
        import deepspeed

        if deepspeed.comm.is_initialized():
            return deepspeed.comm.get_data_parallel_group()
    except Exception:
        pass
    return None


class _AllGatherWithGrad(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, group=None):
        if not dist.is_initialized():
            ctx.world_size, ctx.rank, ctx.local_n = 1, 0, x.size(0)
            return x
        ws = dist.get_world_size(group=group)
        rk = dist.get_rank(group=group)
        bufs = [torch.empty_like(x) for _ in range(ws)]
        dist.all_gather(bufs, x.contiguous(), group=group)
        ctx.world_size, ctx.rank, ctx.local_n = ws, rk, x.size(0)
        return torch.cat(bufs, dim=0)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        if ctx.world_size == 1:
            return grad_out, None
        n = ctx.local_n
        s = ctx.rank * n
        e = s + n
        return grad_out[s:e].contiguous(), None


def all_gather_with_grad(x: torch.Tensor, group=None) -> torch.Tensor:
    return _AllGatherWithGrad.apply(x, group)


@torch.no_grad()
def all_gather_nograd(x: torch.Tensor, group=None) -> torch.Tensor:
    if not dist.is_initialized():
        return x
    ws = dist.get_world_size(group=group)
    bufs = [torch.empty_like(x) for _ in range(ws)]
    dist.all_gather(bufs, x.contiguous(), group=group)
    return torch.cat(bufs, dim=0)


def gather_features(z1: torch.Tensor, z2: torch.Tensor, with_grad: bool = False):
    group = get_dp_group()
    if with_grad:
        return all_gather_with_grad(z1, group), all_gather_with_grad(z2, group)
    return all_gather_nograd(z1, group), all_gather_nograd(z2, group)


@dataclass
class FeatureQueue:
    queue_size: int
    device: torch.device
    buffer: Optional[torch.Tensor] = None
    ptr: int = 0
    filled: bool = False

    def _init_if_needed(self, dim: int) -> None:
        if self.buffer is None:
            self.buffer = torch.zeros(self.queue_size, dim, dtype=torch.float32, device=self.device)

    def has_items(self) -> bool:
        if self.queue_size <= 0 or self.buffer is None:
            return False
        return self.filled or self.ptr > 0

    def get(self) -> Optional[torch.Tensor]:
        if not self.has_items():
            return None
        if self.filled:
            return self.buffer
        return self.buffer[: self.ptr]

    @torch.no_grad()
    def enqueue(self, features: torch.Tensor) -> None:
        if self.queue_size <= 0:
            return
        features = features.detach().float()
        self._init_if_needed(features.shape[-1])

        n = features.shape[0]
        if n >= self.queue_size:
            self.buffer.copy_(features[-self.queue_size :])
            self.ptr = 0
            self.filled = True
            return

        end = self.ptr + n
        if end <= self.queue_size:
            self.buffer[self.ptr : end] = features
        else:
            first = self.queue_size - self.ptr
            self.buffer[self.ptr :] = features[:first]
            self.buffer[: n - first] = features[first:]
            self.filled = True
        self.ptr = (self.ptr + n) % self.queue_size
        if self.ptr == 0:
            self.filled = True


def symmetric_info_nce_loss(
    anchor_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor,
    logit_scale: torch.Tensor,
    cross_device_negatives: bool = True,
    cross_device_gather_with_grad: bool = False,
    feature_queue: Optional[FeatureQueue] = None,
    return_stats: bool = False,
):
    # Keep contrastive math in a unified compute dtype to avoid mixed-precision
    # matmul/cat runtime errors across bf16/fp16/fp32 branches.
    compute_dtype = torch.float32
    anchor_embeddings = F.normalize(anchor_embeddings.to(dtype=compute_dtype), dim=-1)
    positive_embeddings = F.normalize(positive_embeddings.to(dtype=compute_dtype), dim=-1)
    logit_scale = logit_scale.to(dtype=compute_dtype)

    if cross_device_negatives and is_dist_available():
        all_anchor, all_positive = gather_features(
            anchor_embeddings,
            positive_embeddings,
            with_grad=cross_device_gather_with_grad,
        )
    else:
        all_anchor = anchor_embeddings
        all_positive = positive_embeddings

    local_bs = anchor_embeddings.shape[0]
    rank = get_rank()
    offset = rank * local_bs
    global_bs = all_anchor.shape[0]

    rows = torch.cat([anchor_embeddings, positive_embeddings], dim=0)
    cols = torch.cat([all_anchor, all_positive], dim=0)

    raw_logits = rows @ cols.T
    logits = logit_scale * raw_logits

    idx = torch.arange(local_bs, device=logits.device)
    self_col_anchor = offset + idx
    self_col_positive = global_bs + offset + idx

    self_mask = torch.zeros_like(logits, dtype=torch.bool)
    self_mask[idx, self_col_anchor] = True
    self_mask[idx + local_bs, self_col_positive] = True
    logits = logits.masked_fill(self_mask, float("-inf"))

    pos_cols_anchor = global_bs + (offset + idx)
    pos_cols_positive = offset + idx
    labels = torch.cat([pos_cols_anchor, pos_cols_positive], dim=0).long()

    queue_raw_logits = None

    if feature_queue is not None:
        queued = feature_queue.get()
        if queued is not None and queued.numel() > 0:
            # Queue is stored in fp32; align device/dtype before matmul.
            queued = F.normalize(
                queued.to(device=anchor_embeddings.device, dtype=compute_dtype),
                dim=-1,
            )
            queue_raw_logits = rows @ queued.T
            queue_logits = logit_scale * queue_raw_logits
            logits = torch.cat([logits, queue_logits], dim=1)

    row_max = logits.max(dim=-1, keepdim=True).values
    row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
    logits = logits - row_max
    logits = logits.clamp(min=-15.0, max=15.0)

    if logits.shape[1] != self_mask.shape[1]:
        full_self_mask = torch.zeros_like(logits, dtype=torch.bool)
        full_self_mask[:, : self_mask.shape[1]] = self_mask
        self_mask = full_self_mask
    logits = logits.masked_fill(self_mask, float("-inf"))

    loss = F.cross_entropy(logits, labels)

    if feature_queue is not None:
        with torch.no_grad():
            queue_features = torch.cat([all_anchor.detach(), all_positive.detach()], dim=0)
            feature_queue.enqueue(queue_features)

    if return_stats:
        pos_sim = torch.gather(raw_logits, 1, labels.unsqueeze(1)).squeeze(1)
        neg_mask = torch.ones_like(raw_logits, dtype=torch.bool)
        neg_mask.scatter_(1, labels.unsqueeze(1), False)
        neg_mask = neg_mask & (~self_mask[:, : raw_logits.shape[1]])
        neg_values = raw_logits[neg_mask]
        if queue_raw_logits is not None:
            neg_values = torch.cat([neg_values, queue_raw_logits.reshape(-1)], dim=0)
        neg_sim_mean = (
            neg_values.mean()
            if neg_values.numel() > 0
            else torch.tensor(0.0, device=raw_logits.device, dtype=raw_logits.dtype)
        )
        stats: Dict[str, torch.Tensor] = {
            "positive_similarity_mean": pos_sim.mean().detach(),
            "negative_similarity_mean": neg_sim_mean.detach(),
        }
        return loss, stats

    return loss
