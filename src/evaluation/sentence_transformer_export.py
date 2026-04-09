from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import torch

from src.core.config import load_experiment_config
from src.models.router.router_utils import load_router_state_dict
from src.models.text_encoder.moe_text_encoder import MoETextEncoder


def load_router_checkpoint_into_encoder(
    encoder: MoETextEncoder,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    ckpt_path = Path(checkpoint_path)
    payload = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(payload, dict) or "router_state_dict" not in payload:
        raise ValueError(
            f"Invalid router checkpoint format: {ckpt_path}. Expected key 'router_state_dict'."
        )

    loaded, missing = load_router_state_dict(
        encoder.model,
        payload["router_state_dict"],
    )
    return {
        "checkpoint": str(ckpt_path),
        "loaded_router_tensors": loaded,
        "missing_router_tensors": missing,
        "logit_scale": payload.get("logit_scale"),
        "meta": payload.get("meta", {}),
    }


def build_sentence_transformer(
    backbone_dir: str | Path,
    max_seq_length: int | None,
    pooling: str,
    device: str | None = None,
):
    try:
        from sentence_transformers import SentenceTransformer, models
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required. Install it with: "
            "pip install sentence-transformers"
        ) from exc

    backbone_dir = Path(backbone_dir)
    transformer = models.Transformer(
        str(backbone_dir),
        max_seq_length=max_seq_length,
        model_args={"trust_remote_code": True},
        tokenizer_args={"trust_remote_code": True},
    )
    pooling_layer = models.Pooling(
        word_embedding_dimension=transformer.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=(pooling == "mean"),
        pooling_mode_lasttoken=(pooling == "lasttoken"),
    )
    kwargs = {"modules": [transformer, pooling_layer]}
    if device is not None:
        kwargs["device"] = device
    return SentenceTransformer(**kwargs)


def export_sentence_transformer_from_config(
    *,
    config_path: str,
    output_dir: str | Path,
    router_ckpt: str | None = None,
    pooling: str = "mean",
    max_seq_length: int | None = None,
    safe_serialization: bool = False,
    device: str | None = "cpu",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_experiment_config(config_path)
    encoder = MoETextEncoder(config.model)
    encoder.model.eval()

    router_info = None
    if router_ckpt is not None:
        router_info = load_router_checkpoint_into_encoder(encoder, router_ckpt)

    resolved_max_seq_length = max_seq_length or config.model.max_length

    with tempfile.TemporaryDirectory(prefix="moe2encoder_st_export_") as tmp_dir:
        backbone_dir = Path(tmp_dir) / "backbone"
        encoder.model.save_pretrained(
            str(backbone_dir),
            safe_serialization=safe_serialization,
        )
        encoder.tokenizer.save_pretrained(str(backbone_dir))

        st_model = build_sentence_transformer(
            backbone_dir=backbone_dir,
            max_seq_length=resolved_max_seq_length,
            pooling=pooling,
            device=device,
        )
        st_model.save(str(output_dir))

    return {
        "source_config": str(Path(config_path).resolve()),
        "source_model_name_or_path": config.model.model_name_or_path,
        "router_checkpoint": router_info,
        "pooling": pooling,
        "max_seq_length": resolved_max_seq_length,
        "trust_remote_code": bool(config.model.trust_remote_code),
        "torch_dtype": config.model.torch_dtype,
    }


def load_sentence_transformer_from_config(
    *,
    config_path: str,
    router_ckpt: str | None = None,
    pooling: str = "mean",
    max_seq_length: int | None = None,
    safe_serialization: bool = False,
    device: str | None = None,
):
    config = load_experiment_config(config_path)
    encoder = MoETextEncoder(config.model)
    encoder.model.eval()

    router_info = None
    if router_ckpt is not None:
        router_info = load_router_checkpoint_into_encoder(encoder, router_ckpt)

    resolved_max_seq_length = max_seq_length or config.model.max_length
    tmp_dir = tempfile.TemporaryDirectory(prefix="moe2encoder_st_eval_")
    backbone_dir = Path(tmp_dir.name) / "backbone"
    encoder.model.save_pretrained(
        str(backbone_dir),
        safe_serialization=safe_serialization,
    )
    encoder.tokenizer.save_pretrained(str(backbone_dir))

    st_model = build_sentence_transformer(
        backbone_dir=backbone_dir,
        max_seq_length=resolved_max_seq_length,
        pooling=pooling,
        device=device,
    )

    export_meta = {
        "source_config": str(Path(config_path).resolve()),
        "source_model_name_or_path": config.model.model_name_or_path,
        "router_checkpoint": router_info,
        "pooling": pooling,
        "max_seq_length": resolved_max_seq_length,
        "trust_remote_code": bool(config.model.trust_remote_code),
        "torch_dtype": config.model.torch_dtype,
    }
    return st_model, export_meta, tmp_dir
