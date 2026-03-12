from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


def _recall_and_rank(query: torch.Tensor, candidates: torch.Tensor, chunk_size: int = 512):
    n = query.shape[0]
    ranks = []
    recall1 = 0
    recall5 = 0

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        q = query[start:end]
        sims = q @ candidates.T

        target_indices = torch.arange(start, end, device=query.device)
        target_scores = sims[torch.arange(end - start, device=query.device), target_indices]
        rank = (sims > target_scores.unsqueeze(1)).sum(dim=1) + 1
        ranks.append(rank)

        topk = torch.topk(sims, k=min(5, n), dim=1).indices
        recall1 += (topk[:, 0] == target_indices).sum().item()
        recall5 += (topk == target_indices.unsqueeze(1)).any(dim=1).sum().item()

    ranks = torch.cat(ranks, dim=0).float()
    return {
        "recall@1": recall1 / float(n),
        "recall@5": recall5 / float(n),
        "mean_rank": float(ranks.mean().item()),
        "median_rank": float(ranks.median().item()),
    }


def compute_pair_metrics(anchor_embeddings: torch.Tensor, positive_embeddings: torch.Tensor) -> Dict[str, float]:
    anchor_embeddings = F.normalize(anchor_embeddings.float(), dim=-1)
    positive_embeddings = F.normalize(positive_embeddings.float(), dim=-1)

    n = anchor_embeddings.shape[0]
    diag_cos = (anchor_embeddings * positive_embeddings).sum(dim=-1)
    sim_matrix = anchor_embeddings @ positive_embeddings.T
    sim_matrix.fill_diagonal_(-1.0)
    hardest_negative = sim_matrix.max(dim=1).values

    a2p = _recall_and_rank(anchor_embeddings, positive_embeddings)
    p2a = _recall_and_rank(positive_embeddings, anchor_embeddings)

    metrics = {
        "num_samples": float(n),
        "pos_cosine_mean": float(diag_cos.mean().item()),
        "pos_cosine_std": float(diag_cos.std().item()),
        "hardest_neg_cosine_mean": float(hardest_negative.mean().item()),
        "margin_pos_minus_hardneg": float((diag_cos - hardest_negative).mean().item()),
    }
    for k, v in a2p.items():
        metrics[f"a2p_{k}"] = float(v)
    for k, v in p2a.items():
        metrics[f"p2a_{k}"] = float(v)
    return metrics


def compute_group_compactness(
    embeddings: torch.Tensor,
    group_ids: Optional[List[Optional[str]]],
) -> Dict[str, float]:
    if group_ids is None:
        return {}

    normalized = F.normalize(embeddings.float(), dim=-1)
    group_to_indices = defaultdict(list)
    for idx, gid in enumerate(group_ids):
        if gid is None:
            continue
        group_to_indices[str(gid)].append(idx)

    valid_groups = [idxs for idxs in group_to_indices.values() if len(idxs) >= 2]
    if len(valid_groups) < 1:
        return {}

    intra_vals = []
    for idxs in valid_groups:
        vecs = normalized[idxs]
        sims = vecs @ vecs.T
        mask = ~torch.eye(sims.shape[0], dtype=torch.bool, device=sims.device)
        intra_vals.append(sims[mask].mean().item())

    centroids = []
    for idxs in group_to_indices.values():
        vecs = normalized[idxs]
        centroids.append(F.normalize(vecs.mean(dim=0), dim=0))
    if len(centroids) < 2:
        inter_val = 0.0
    else:
        centroids = torch.stack(centroids, dim=0)
        sims = centroids @ centroids.T
        mask = ~torch.eye(sims.shape[0], dtype=torch.bool, device=sims.device)
        inter_val = sims[mask].mean().item()

    return {
        "group_intra_cosine_mean": float(sum(intra_vals) / len(intra_vals)),
        "group_inter_centroid_cosine_mean": float(inter_val),
    }
