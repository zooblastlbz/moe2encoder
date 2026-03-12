from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ModelConfig:
    model_name_or_path: str = "Qwen/Qwen3.5-35B-A3B"
    tokenizer_name_or_path: Optional[str] = None
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"
    attn_implementation: Optional[str] = None
    gradient_checkpointing: bool = True
    use_cache: bool = False
    max_length: int = 256
    router_name_patterns: List[str] = field(
        default_factory=lambda: ["shared_expert_gate", "mlp.gate.weight"]
    )


@dataclass
class DataConfig:
    train_jsonl: Optional[str] = None
    eval_jsonl: Optional[str] = None
    num_workers: int = 4


@dataclass
class TrainingConfig:
    output_dir: str = "outputs/router_only"
    seed: int = 42
    epochs: int = 1
    train_batch_size: int = 4
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    log_every: int = 10
    save_every_steps: int = 500
    eval_every_steps: int = 500
    max_steps: int = -1


@dataclass
class ContrastiveConfig:
    temperature_init: float = 0.07
    logit_scale_max: float = 100.0
    cross_device_negatives: bool = True
    feature_queue_size: int = 0


@dataclass
class RuntimeConfig:
    bf16: bool = True
    fp16: bool = False
    tf32: bool = True
    distributed_backend: str = "auto"
    distributed_timeout_minutes: int = 60


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    contrastive: ContrastiveConfig = field(default_factory=ContrastiveConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def _build_dataclass(dc_cls, values: Optional[Dict[str, Any]]):
    values = values or {}
    return dc_cls(**values)


def load_experiment_config(path: str) -> ExperimentConfig:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    config = ExperimentConfig(
        model=_build_dataclass(ModelConfig, raw.get("model")),
        data=_build_dataclass(DataConfig, raw.get("data")),
        training=_build_dataclass(TrainingConfig, raw.get("training")),
        contrastive=_build_dataclass(ContrastiveConfig, raw.get("contrastive")),
        runtime=_build_dataclass(RuntimeConfig, raw.get("runtime")),
    )
    return config


def as_dict(config: ExperimentConfig) -> Dict[str, Any]:
    return {
        "model": config.model.__dict__,
        "data": config.data.__dict__,
        "training": config.training.__dict__,
        "contrastive": config.contrastive.__dict__,
        "runtime": config.runtime.__dict__,
    }
