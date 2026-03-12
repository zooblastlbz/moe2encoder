from __future__ import annotations

import os
from datetime import timedelta
from typing import Any, List

import torch
import torch.distributed as dist


def is_dist_available() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if not is_dist_available():
        return 0
    return dist.get_rank()


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_world_size() -> int:
    if not is_dist_available():
        return 1
    return dist.get_world_size()


def is_main_process() -> bool:
    return get_rank() == 0


def is_local_main_process() -> bool:
    return get_local_rank() == 0


def barrier() -> None:
    if is_dist_available():
        dist.barrier()


def init_distributed(backend: str = "auto", timeout_minutes: int = 60) -> torch.device:
    if "RANK" not in os.environ:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(0)
        return device

    if backend == "auto":
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    if not dist.is_initialized():
        local_rank = get_local_rank()
        if torch.cuda.is_available():
            num_devices = torch.cuda.device_count()
            if local_rank >= num_devices:
                raise RuntimeError(
                    f"LOCAL_RANK={local_rank} but only {num_devices} CUDA devices are visible."
                )
            torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(minutes=timeout_minutes),
        )
    if torch.cuda.is_available():
        return torch.device("cuda", get_local_rank())
    return torch.device("cpu")


def all_gather_object(obj: Any) -> List[Any]:
    world_size = get_world_size()
    gathered: List[Any] = [None for _ in range(world_size)]
    if world_size == 1:
        gathered[0] = obj
        return gathered
    dist.all_gather_object(gathered, obj)
    return gathered


def cleanup_distributed() -> None:
    if is_dist_available():
        dist.destroy_process_group()
