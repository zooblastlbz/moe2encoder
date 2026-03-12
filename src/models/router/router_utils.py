from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch
from torch import nn


def _matches(name: str, patterns: Iterable[str]) -> bool:
    n = name.lower()
    return any(p.lower() in n for p in patterns)


def freeze_all_but_router(
    model: nn.Module, router_name_patterns: List[str]
) -> Dict[str, object]:
    trainable_names: List[str] = []
    frozen_names: List[str] = []
    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        is_router = _matches(name, router_name_patterns)
        param.requires_grad = is_router
        total_params += param.numel()
        if is_router:
            trainable_names.append(name)
            trainable_params += param.numel()
        else:
            frozen_names.append(name)

    ratio = 0.0 if total_params == 0 else trainable_params / float(total_params)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio": ratio,
        "trainable_names": trainable_names,
        "frozen_names_count": len(frozen_names),
    }


def router_trainable_parameters(model: nn.Module):
    for _, param in model.named_parameters():
        if param.requires_grad:
            yield param


def extract_router_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    state = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            state[name] = param.detach().cpu()
    return state


def load_router_state_dict(
    model: nn.Module, router_state_dict: Dict[str, torch.Tensor]
) -> Tuple[int, int]:
    loaded = 0
    missing = 0
    name_to_param = dict(model.named_parameters())

    for name, tensor in router_state_dict.items():
        if name not in name_to_param:
            missing += 1
            continue
        with torch.no_grad():
            name_to_param[name].copy_(tensor.to(name_to_param[name].device))
        loaded += 1
    return loaded, missing
