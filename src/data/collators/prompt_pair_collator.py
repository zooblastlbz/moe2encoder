from __future__ import annotations

from typing import Dict, List, Optional

import torch


class PromptPairCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Optional[str]]]) -> Dict[str, object]:
        anchors = [row["anchor_text"] for row in batch]
        positives = [row["positive_text"] for row in batch]

        anchor_encoded = self.tokenizer(
            anchors,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        positive_encoded = self.tokenizer(
            positives,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "anchor_input_ids": anchor_encoded["input_ids"],
            "anchor_attention_mask": anchor_encoded["attention_mask"],
            "positive_input_ids": positive_encoded["input_ids"],
            "positive_attention_mask": positive_encoded["attention_mask"],
            "sample_idx": torch.tensor([row["sample_idx"] for row in batch], dtype=torch.long),
            "group_id": [row.get("group_id") for row in batch],
            "prompt_type": [row.get("prompt_type") for row in batch],
            "anchor_text": anchors,
            "positive_text": positives,
        }
