from __future__ import annotations

import math
from typing import Dict, List


def _safe_div(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return a / b


def compare_metrics(pre: Dict[str, float], post: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    keys = sorted(set(pre.keys()) | set(post.keys()))
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        pre_v = float(pre.get(k, 0.0))
        post_v = float(post.get(k, 0.0))
        delta = post_v - pre_v
        out[k] = {
            "pre": pre_v,
            "post": post_v,
            "delta": delta,
            "relative_delta": _safe_div(delta, abs(pre_v) + 1e-12),
        }
    return out


def _js_divergence(p: List[float], q: List[float]) -> float:
    m = [(pi + qi) / 2.0 for pi, qi in zip(p, q)]

    def kl(a, b):
        s = 0.0
        for ai, bi in zip(a, b):
            if ai <= 0:
                continue
            s += ai * math.log((ai + 1e-12) / (bi + 1e-12))
        return s

    return 0.5 * (kl(p, m) + kl(q, m))


def compare_routing(
    pre: Dict[str, Dict[str, object]],
    post: Dict[str, Dict[str, object]],
) -> Dict[str, Dict[str, float]]:
    modules = sorted(set(pre.keys()) & set(post.keys()))
    out: Dict[str, Dict[str, float]] = {}
    for m in modules:
        pre_m = pre[m]
        post_m = post[m]
        pre_dist = [float(x) for x in pre_m.get("expert_distribution", [])]
        post_dist = [float(x) for x in post_m.get("expert_distribution", [])]
        if len(pre_dist) != len(post_dist) or len(pre_dist) == 0:
            continue
        out[m] = {
            "pre_entropy": float(pre_m.get("mean_entropy", 0.0)),
            "post_entropy": float(post_m.get("mean_entropy", 0.0)),
            "entropy_delta": float(post_m.get("mean_entropy", 0.0))
            - float(pre_m.get("mean_entropy", 0.0)),
            "pre_top_expert_share": float(pre_m.get("top_expert_share", 0.0)),
            "post_top_expert_share": float(post_m.get("top_expert_share", 0.0)),
            "top_expert_share_delta": float(post_m.get("top_expert_share", 0.0))
            - float(pre_m.get("top_expert_share", 0.0)),
            "expert_distribution_jsd": _js_divergence(pre_dist, post_dist),
        }
    return out
