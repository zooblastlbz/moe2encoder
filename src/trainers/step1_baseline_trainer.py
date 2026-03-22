from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from src.core.config import as_dict, load_experiment_config
from src.data.collators.prompt_pair_collator import PromptPairCollator
from src.data.datasets.prompt_pair_dataset import PromptPairDataset
from src.evaluation.evaluator import build_sharded_eval_loader, evaluate_text_encoder
from src.evaluation.router_stats import RouterStatsTracker
from src.models.text_encoder.moe_text_encoder import MoETextEncoder
from src.utils.distributed import (
    cleanup_distributed,
    get_rank,
    init_distributed,
    is_main_process,
)
from src.utils.io import ensure_dir, save_json
from src.utils.seed import seed_everything


@dataclass
class Step1Options:
    config_path: str
    output_dir: Optional[str] = None
    eval_jsonl: Optional[str] = None


class Step1BaselineTrainer:
    def __init__(self, options: Step1Options):
        self.options = options

    def run(self) -> None:
        config = load_experiment_config(self.options.config_path)
        device = init_distributed(
            backend=config.runtime.distributed_backend,
            timeout_minutes=config.runtime.distributed_timeout_minutes,
        )
        seed_everything(config.training.seed + get_rank())

        if config.runtime.tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        output_dir = Path(
            self.options.output_dir or Path(config.training.output_dir) / "step1_baseline"
        )
        ensure_dir(output_dir)

        eval_path = self.options.eval_jsonl or config.data.eval_jsonl
        if eval_path is None:
            raise ValueError("Step1 requires data.eval_jsonl or --eval_jsonl.")

        encoder = MoETextEncoder(config.model).to(device)
        for param in encoder.model.parameters():
            param.requires_grad = False

        dataset = PromptPairDataset(eval_path)
        collator = PromptPairCollator(encoder.tokenizer, max_length=config.model.max_length)
        loader = build_sharded_eval_loader(
            dataset=dataset,
            collator=collator,
            batch_size=config.training.eval_batch_size,
            num_workers=config.data.num_workers,
        )

        tracker = RouterStatsTracker(encoder.model, config.model.router_name_patterns)
        hooked_modules = tracker.register()
        if is_main_process():
            print(f"[Step1] Router hooks registered: {hooked_modules}")
        result = evaluate_text_encoder(
            model=encoder,
            dataloader=loader,
            device=device,
            router_tracker=tracker,
            use_bf16=config.runtime.bf16,
            use_fp16=config.runtime.fp16,
        )
        tracker.clear()

        if is_main_process():
            save_json(output_dir / "metrics.json", result["metrics"])
            save_json(output_dir / "routing_stats.json", result["routing"])
            save_json(
                output_dir / "summary.json",
                {
                    "sample_count": result["sample_count"],
                    "prompt_type_distribution": result["prompt_type_distribution"],
                    "config": as_dict(config),
                },
            )
            print(f"[Step1] Finished. Results saved to: {output_dir}")

        cleanup_distributed()


def run_step1(options: Step1Options) -> None:
    Step1BaselineTrainer(options).run()
