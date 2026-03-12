from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from src.core.config import ModelConfig


def _parse_dtype(dtype: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = dtype.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported torch_dtype: {dtype}")
    return mapping[key]


def masked_mean_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


class MoETextEncoder(nn.Module):
    def __init__(self, model_cfg: ModelConfig):
        super().__init__()
        self.model_cfg = model_cfg
        model_name = model_cfg.model_name_or_path
        tokenizer_name = model_cfg.tokenizer_name_or_path or model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=model_cfg.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = _parse_dtype(model_cfg.torch_dtype)
        kwargs: Dict[str, object] = {
            "trust_remote_code": model_cfg.trust_remote_code,
            "torch_dtype": dtype,
        }
        if model_cfg.attn_implementation:
            kwargs["attn_implementation"] = model_cfg.attn_implementation

        try:
            self.backbone = AutoModel.from_pretrained(model_name, **kwargs)
        except Exception:
            self.backbone = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

        if hasattr(self.backbone.config, "use_cache"):
            self.backbone.config.use_cache = model_cfg.use_cache
        if model_cfg.gradient_checkpointing and hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = True,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            use_cache=False,
        )

        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            token_hidden = outputs.hidden_states[-1]
        elif hasattr(outputs, "last_hidden_state"):
            token_hidden = outputs.last_hidden_state
        else:
            raise RuntimeError("Model output does not contain hidden states.")

        sentence_embed = masked_mean_pooling(token_hidden, attention_mask)
        sentence_embed = torch.nn.functional.normalize(sentence_embed, dim=-1)

        return {
            "sentence_embeddings": sentence_embed,
            "token_hidden_states": token_hidden,
        }

    @property
    def model(self) -> nn.Module:
        return self.backbone
