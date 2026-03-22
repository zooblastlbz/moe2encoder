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


def _activation_metrics_from_distribution(dist: List[float]) -> Dict[str, float]:
    n = len(dist)
    if n == 0:
        return {
            "num_experts": 0.0,
            "expert_usage_entropy": 0.0,
            "expert_usage_entropy_normalized": 0.0,
            "effective_expert_count": 0.0,
            "effective_expert_ratio": 0.0,
            "active_expert_count_ge_1pct": 0.0,
            "active_expert_ratio_ge_1pct": 0.0,
            "active_expert_count_ge_0p1pct": 0.0,
            "active_expert_ratio_ge_0p1pct": 0.0,
            "top2_expert_share": 0.0,
            "top4_expert_share": 0.0,
            "gini_coefficient": 0.0,
            "load_balance_cv": 0.0,
        }

    eps = 1e-12
    entropy = 0.0
    active_1pct = 0
    active_0p1pct = 0
    for p in dist:
        p = float(p)
        if p >= 0.01:
            active_1pct += 1
        if p >= 0.001:
            active_0p1pct += 1
        if p > 0:
            entropy += -p * math.log(max(p, eps))

    top2 = sum(sorted(dist, reverse=True)[: min(2, n)])
    top4 = sum(sorted(dist, reverse=True)[: min(4, n)])
    effective = math.exp(entropy)
    mean_load = 1.0 / float(n)
    var = sum((float(p) - mean_load) ** 2 for p in dist) / float(n)
    std = math.sqrt(max(0.0, var))

    sorted_dist = sorted(float(x) for x in dist)
    weighted = sum((i + 1) * p for i, p in enumerate(sorted_dist))
    gini = (2.0 * weighted) / float(n) - (float(n + 1) / float(n))
    gini = max(0.0, min(1.0, gini))

    return {
        "num_experts": float(n),
        "expert_usage_entropy": float(entropy),
        "expert_usage_entropy_normalized": float(_safe_div(entropy, math.log(n) if n > 1 else 0.0)),
        "effective_expert_count": float(effective),
        "effective_expert_ratio": float(_safe_div(effective, float(n))),
        "active_expert_count_ge_1pct": float(active_1pct),
        "active_expert_ratio_ge_1pct": float(_safe_div(active_1pct, float(n))),
        "active_expert_count_ge_0p1pct": float(active_0p1pct),
        "active_expert_ratio_ge_0p1pct": float(_safe_div(active_0p1pct, float(n))),
        "top2_expert_share": float(top2),
        "top4_expert_share": float(top4),
        "gini_coefficient": float(gini),
        "load_balance_cv": float(_safe_div(std, mean_load)),
    }


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
        pre_activation = _activation_metrics_from_distribution(pre_dist)
        post_activation = _activation_metrics_from_distribution(post_dist)
        pre_top1 = float(pre_m.get("top_expert_share", max(pre_dist)))
        post_top1 = float(post_m.get("top_expert_share", max(post_dist)))
        pre_top2 = float(pre_m.get("top2_expert_share", pre_activation["top2_expert_share"]))
        post_top2 = float(post_m.get("top2_expert_share", post_activation["top2_expert_share"]))
        pre_top4 = float(pre_m.get("top4_expert_share", pre_activation["top4_expert_share"]))
        post_top4 = float(post_m.get("top4_expert_share", post_activation["top4_expert_share"]))
        pre_effective = float(pre_m.get("effective_expert_count", pre_activation["effective_expert_count"]))
        post_effective = float(post_m.get("effective_expert_count", post_activation["effective_expert_count"]))
        pre_active_1pct = float(
            pre_m.get("active_expert_count_ge_1pct", pre_activation["active_expert_count_ge_1pct"])
        )
        post_active_1pct = float(
            post_m.get("active_expert_count_ge_1pct", post_activation["active_expert_count_ge_1pct"])
        )
        pre_active_ratio_1pct = float(
            pre_m.get("active_expert_ratio_ge_1pct", pre_activation["active_expert_ratio_ge_1pct"])
        )
        post_active_ratio_1pct = float(
            post_m.get("active_expert_ratio_ge_1pct", post_activation["active_expert_ratio_ge_1pct"])
        )
        pre_gini = float(pre_m.get("gini_coefficient", pre_activation["gini_coefficient"]))
        post_gini = float(post_m.get("gini_coefficient", post_activation["gini_coefficient"]))
        pre_usage_entropy_norm = float(
            pre_m.get("expert_usage_entropy_normalized", pre_activation["expert_usage_entropy_normalized"])
        )
        post_usage_entropy_norm = float(
            post_m.get("expert_usage_entropy_normalized", post_activation["expert_usage_entropy_normalized"])
        )
        dist_l1 = float(sum(abs(a - b) for a, b in zip(pre_dist, post_dist)))
        dist_jsd = _js_divergence(pre_dist, post_dist)
        out[m] = {
            "pre_entropy": float(pre_m.get("mean_entropy", 0.0)),
            "post_entropy": float(post_m.get("mean_entropy", 0.0)),
            "entropy_delta": float(post_m.get("mean_entropy", 0.0))
            - float(pre_m.get("mean_entropy", 0.0)),
            "pre_top_expert_share": pre_top1,
            "post_top_expert_share": post_top1,
            "top_expert_share_delta": post_top1 - pre_top1,
            "pre_top2_expert_share": pre_top2,
            "post_top2_expert_share": post_top2,
            "top2_expert_share_delta": post_top2 - pre_top2,
            "pre_top4_expert_share": pre_top4,
            "post_top4_expert_share": post_top4,
            "top4_expert_share_delta": post_top4 - pre_top4,
            "pre_effective_expert_count": pre_effective,
            "post_effective_expert_count": post_effective,
            "effective_expert_count_delta": post_effective - pre_effective,
            "pre_active_expert_count_ge_1pct": pre_active_1pct,
            "post_active_expert_count_ge_1pct": post_active_1pct,
            "active_expert_count_ge_1pct_delta": post_active_1pct - pre_active_1pct,
            "pre_active_expert_ratio_ge_1pct": pre_active_ratio_1pct,
            "post_active_expert_ratio_ge_1pct": post_active_ratio_1pct,
            "active_expert_ratio_ge_1pct_delta": post_active_ratio_1pct - pre_active_ratio_1pct,
            "pre_gini_coefficient": pre_gini,
            "post_gini_coefficient": post_gini,
            "gini_coefficient_delta": post_gini - pre_gini,
            "pre_expert_usage_entropy_normalized": pre_usage_entropy_norm,
            "post_expert_usage_entropy_normalized": post_usage_entropy_norm,
            "expert_usage_entropy_normalized_delta": post_usage_entropy_norm - pre_usage_entropy_norm,
            "expert_distribution_jsd": dist_jsd,
            "expert_distribution_l1": dist_l1,
            "expert_distribution_tv": 0.5 * dist_l1,
        }
    return out
