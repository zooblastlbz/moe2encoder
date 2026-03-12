from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from torch.utils.data import Dataset


class PromptPairDataset(Dataset):
    """Dataset for contrastive text pairs in jsonl format."""

    def __init__(self, jsonl_path: str):
        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {jsonl_path}")

        self.samples: List[Dict[str, Optional[str]]] = []
        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "anchor_text" not in row or "positive_text" not in row:
                    raise ValueError(
                        f"Line {idx + 1} missing required fields anchor_text/positive_text"
                    )
                self.samples.append(
                    {
                        "sample_idx": idx,
                        "anchor_text": row["anchor_text"],
                        "positive_text": row["positive_text"],
                        "group_id": row.get("group_id"),
                        "prompt_type": row.get("prompt_type"),
                    }
                )

        if not self.samples:
            raise ValueError(f"No valid samples found in {jsonl_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Optional[str]]:
        return self.samples[idx]
