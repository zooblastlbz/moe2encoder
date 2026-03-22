from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

from src.utils.distributed import get_rank, get_world_size, is_dist_available


def gather_with_grad(local_features: torch.Tensor) -> torch.Tensor:
    world_size = get_world_size()
    if world_size == 1:
        return local_features

    gathered = [torch.zeros_like(local_features) for _ in range(world_size)]
    dist.all_gather(gathered, local_features)
    gathered[get_rank()] = local_features
    return torch.cat(gathered, dim=0)


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
    feature_queue: Optional[FeatureQueue] = None,
) -> torch.Tensor:
    # Keep contrastive math in a unified compute dtype to avoid mixed-precision
    # matmul/cat runtime errors across bf16/fp16/fp32 branches.
    compute_dtype = torch.float32
    anchor_embeddings = F.normalize(anchor_embeddings.to(dtype=compute_dtype), dim=-1)
    positive_embeddings = F.normalize(positive_embeddings.to(dtype=compute_dtype), dim=-1)
    logit_scale = logit_scale.to(dtype=compute_dtype)

    if cross_device_negatives and is_dist_available():
        all_anchor = gather_with_grad(anchor_embeddings)
        all_positive = gather_with_grad(positive_embeddings)
        local_bs = anchor_embeddings.shape[0]
        start = get_rank() * local_bs
        targets = torch.arange(start, start + local_bs, device=anchor_embeddings.device)
    else:
        all_anchor = anchor_embeddings
        all_positive = positive_embeddings
        targets = torch.arange(anchor_embeddings.shape[0], device=anchor_embeddings.device)

    logits_a = logit_scale * (anchor_embeddings @ all_positive.T)
    logits_p = logit_scale * (positive_embeddings @ all_anchor.T)

    if feature_queue is not None:
        queued = feature_queue.get()
        if queued is not None and queued.numel() > 0:
            # Queue is stored in fp32; align device/dtype before matmul.
            queued = F.normalize(
                queued.to(device=anchor_embeddings.device, dtype=compute_dtype),
                dim=-1,
            )
            logits_a = torch.cat([logits_a, logit_scale * (anchor_embeddings @ queued.T)], dim=1)
            logits_p = torch.cat([logits_p, logit_scale * (positive_embeddings @ queued.T)], dim=1)

    loss_a = F.cross_entropy(logits_a, targets)
    loss_p = F.cross_entropy(logits_p, targets)
    loss = 0.5 * (loss_a + loss_p)

    if feature_queue is not None:
        with torch.no_grad():
            queue_features = torch.cat([all_anchor.detach(), all_positive.detach()], dim=0)
            feature_queue.enqueue(queue_features)

    return loss
