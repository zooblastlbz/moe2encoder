from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch import nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from src.core.config import as_dict, load_experiment_config
from src.data.collators.prompt_pair_collator import PromptPairCollator
from src.data.datasets.prompt_pair_dataset import PromptPairDataset
from src.evaluation.evaluator import build_sharded_eval_loader, evaluate_text_encoder
from src.evaluation.router_stats import RouterStatsTracker
from src.losses.contrastive import FeatureQueue, symmetric_info_nce_loss
from src.models.router.router_utils import extract_router_state_dict, freeze_all_but_router
from src.models.text_encoder.moe_text_encoder import MoETextEncoder
from src.utils.io import ensure_dir, load_json, save_json


@dataclass
class Step2Options:
    config_path: str
    output_dir: Optional[str] = None
    train_jsonl: Optional[str] = None
    eval_jsonl: Optional[str] = None
    run_post_eval: bool = False
    resume_from: Optional[str] = None


class RouterContrastiveTrainModule(nn.Module):
    def __init__(self, encoder: MoETextEncoder, temperature_init: float):
        super().__init__()
        self.encoder = encoder
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / temperature_init), dtype=torch.float32)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = True,
    ):
        return self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )

    def current_logit_scale(self, max_scale: float) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=max_scale)


class RouterContrastiveTrainer:
    TRAINER_STATE_FILE = "trainer_state.json"

    def __init__(self, options: Step2Options):
        self.options = options

    @staticmethod
    def _resolve_mixed_precision(runtime_cfg) -> str:
        if runtime_cfg.bf16:
            return "bf16"
        if runtime_cfg.fp16:
            return "fp16"
        return "no"

    @staticmethod
    def _move_to_device(batch, device):
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v.to(device, non_blocking=True)
            else:
                out[k] = v
        return out

    @staticmethod
    def _save_router_checkpoint(path: Path, train_module, meta: dict):
        payload = {
            "router_state_dict": extract_router_state_dict(train_module.encoder.model),
            "logit_scale": float(train_module.logit_scale.detach().exp().item()),
            "meta": meta,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    @classmethod
    def _save_training_checkpoint(
        cls,
        accelerator: Accelerator,
        checkpoint_dir: Path,
        trainer_state: dict,
    ) -> None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        accelerator.save_state(str(checkpoint_dir))
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            save_json(checkpoint_dir / cls.TRAINER_STATE_FILE, trainer_state)

    @classmethod
    def _load_trainer_state(cls, checkpoint_dir: Path) -> dict:
        state_path = checkpoint_dir / cls.TRAINER_STATE_FILE
        if not state_path.exists():
            return {}
        return load_json(state_path)

    def run(self) -> None:
        config = load_experiment_config(self.options.config_path)
        mixed_precision = self._resolve_mixed_precision(config.runtime)
        accelerator = Accelerator(
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
        )
        set_seed(config.training.seed, device_specific=True)

        if config.runtime.tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        output_dir = Path(self.options.output_dir or Path(config.training.output_dir) / "step2_train")
        ensure_dir(output_dir)
        log_path = output_dir / "train_log.jsonl"

        train_path = self.options.train_jsonl or config.data.train_jsonl
        eval_path = self.options.eval_jsonl or config.data.eval_jsonl
        if train_path is None:
            raise ValueError("Step2 requires data.train_jsonl or --train_jsonl.")

        encoder = MoETextEncoder(config.model)
        train_module = RouterContrastiveTrainModule(
            encoder=encoder,
            temperature_init=config.contrastive.temperature_init,
        )
        freeze_report = freeze_all_but_router(
            train_module.encoder.model, config.model.router_name_patterns
        )
        freeze_report["logit_scale_trainable"] = True
        if freeze_report["trainable_params"] == 0:
            raise RuntimeError(
                "No router parameters matched. Update model.router_name_patterns in your config."
            )
        trainable_param_count = sum(
            p.numel() for p in train_module.parameters() if p.requires_grad
        )

        if accelerator.is_main_process:
            save_json(output_dir / "freeze_report.json", freeze_report)
            save_json(output_dir / "resolved_config.json", as_dict(config))
            accelerator.print(
                "[Step2-ACC] Trainable parameters: "
                f"{trainable_param_count:,} "
                f"(router={freeze_report['trainable_params']:,}, "
                f"logit_scale={trainable_param_count - int(freeze_report['trainable_params'])})"
            )

        train_dataset = PromptPairDataset(train_path)
        collator = PromptPairCollator(
            train_module.encoder.tokenizer, max_length=config.model.max_length
        )

        drop_last = bool(
            config.contrastive.cross_device_negatives and accelerator.num_processes > 1
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.train_batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            pin_memory=True,
            drop_last=drop_last,
            collate_fn=collator,
        )
        if len(train_loader) == 0:
            raise ValueError(
                "Train loader is empty. Check dataset size and batch settings. "
                "With cross-device negatives enabled, drop_last=True is required."
            )

        optimizer = torch.optim.AdamW(
            [p for p in train_module.parameters() if p.requires_grad],
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        train_module, optimizer, train_loader = accelerator.prepare(
            train_module, optimizer, train_loader
        )

        steps_per_epoch = math.ceil(
            len(train_loader) / config.training.gradient_accumulation_steps
        )
        max_steps = (
            config.training.max_steps
            if config.training.max_steps > 0
            else config.training.epochs * steps_per_epoch
        )
        warmup_steps = int(max_steps * config.training.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
        accelerator.register_for_checkpointing(scheduler)

        queue = None
        if config.contrastive.feature_queue_size > 0:
            queue = FeatureQueue(
                queue_size=config.contrastive.feature_queue_size,
                device=accelerator.device,
            )

        global_step = 0
        start_epoch = 0
        start_batch_in_epoch = 0
        stop = False
        max_logit = math.log(config.contrastive.logit_scale_max)

        if self.options.resume_from:
            resume_dir = Path(self.options.resume_from)
            if not resume_dir.exists():
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_dir}")
            accelerator.print(f"[Step2-ACC] Loading checkpoint from: {resume_dir}")
            accelerator.load_state(str(resume_dir))
            trainer_state = self._load_trainer_state(resume_dir)
            global_step = int(trainer_state.get("global_step", 0))
            start_epoch = int(trainer_state.get("epoch", 0))
            start_batch_in_epoch = int(trainer_state.get("batch_in_epoch", 0))
            if accelerator.is_main_process:
                accelerator.print(
                    f"[Step2-ACC] Resume state: global_step={global_step}, "
                    f"epoch={start_epoch}, batch_in_epoch={start_batch_in_epoch}"
                )

        if global_step >= max_steps:
            stop = True

        train_module.train()
        optimizer.zero_grad(set_to_none=True)
        last_epoch = start_epoch - 1
        for epoch in range(start_epoch, config.training.epochs):
            if stop:
                break
            last_epoch = epoch
            resume_skip = start_batch_in_epoch if epoch == start_epoch else 0
            for batch_idx, batch in enumerate(train_loader):
                if resume_skip and batch_idx < resume_skip:
                    continue
                batch = self._move_to_device(batch, accelerator.device)
                with accelerator.accumulate(train_module):
                    out_a = train_module(
                        input_ids=batch["anchor_input_ids"],
                        attention_mask=batch["anchor_attention_mask"],
                        output_hidden_states=True,
                    )
                    out_p = train_module(
                        input_ids=batch["positive_input_ids"],
                        attention_mask=batch["positive_attention_mask"],
                        output_hidden_states=True,
                    )

                    scale = accelerator.unwrap_model(train_module).current_logit_scale(
                        config.contrastive.logit_scale_max
                    )
                    loss, sim_stats = symmetric_info_nce_loss(
                        anchor_embeddings=out_a["sentence_embeddings"],
                        positive_embeddings=out_p["sentence_embeddings"],
                        logit_scale=scale,
                        cross_device_negatives=config.contrastive.cross_device_negatives,
                        feature_queue=queue,
                        return_stats=True,
                    )
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        trainable_params = [
                            p for p in train_module.parameters() if p.requires_grad
                        ]
                        accelerator.clip_grad_norm_(
                            trainable_params, config.training.max_grad_norm
                        )
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        with torch.no_grad():
                            accelerator.unwrap_model(train_module).logit_scale.clamp_(
                                max=max_logit
                            )

                if accelerator.sync_gradients:
                    global_step += 1
                    reduced_loss = accelerator.gather(loss.detach()).mean().item()
                    reduced_pos_sim = (
                        accelerator.gather(sim_stats["positive_similarity_mean"]).mean().item()
                    )
                    reduced_neg_sim = (
                        accelerator.gather(sim_stats["negative_similarity_mean"]).mean().item()
                    )

                    if accelerator.is_main_process:
                        log_row = {
                            "step": global_step,
                            "epoch": epoch,
                            "loss": float(reduced_loss),
                            "positive_similarity_mean": float(reduced_pos_sim),
                            "negative_similarity_mean": float(reduced_neg_sim),
                            "lr": scheduler.get_last_lr()[0],
                            "logit_scale": float(
                                accelerator.unwrap_model(train_module)
                                .logit_scale.detach()
                                .exp()
                                .item()
                            ),
                        }
                        with log_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(log_row, ensure_ascii=False) + "\n")
                        accelerator.print(
                            f"[Step2-ACC] step={global_step} loss={log_row['loss']:.4f} "
                            f"pos_sim={log_row['positive_similarity_mean']:.4f} "
                            f"neg_sim={log_row['negative_similarity_mean']:.4f} "
                            f"lr={log_row['lr']:.3e} logit_scale={log_row['logit_scale']:.3f}"
                        )

                    if (
                        config.training.save_every_steps > 0
                        and global_step % config.training.save_every_steps == 0
                    ):
                        trainer_state = {
                            "global_step": global_step,
                            "epoch": epoch,
                            "batch_in_epoch": batch_idx + 1,
                            "max_steps": max_steps,
                        }
                        state_dir = (
                            output_dir / "checkpoints" / f"state_step_{global_step}"
                        )
                        self._save_training_checkpoint(
                            accelerator=accelerator,
                            checkpoint_dir=state_dir,
                            trainer_state=trainer_state,
                        )
                    if (
                        accelerator.is_main_process
                        and config.training.save_every_steps > 0
                        and global_step % config.training.save_every_steps == 0
                    ):
                        ckpt = output_dir / "checkpoints" / f"router_step_{global_step}.pt"
                        self._save_router_checkpoint(
                            ckpt,
                            accelerator.unwrap_model(train_module),
                            meta=trainer_state,
                        )

                    if global_step >= max_steps:
                        stop = True
                        break
            start_batch_in_epoch = 0
            if stop:
                break

        accelerator.wait_for_everyone()
        final_state = {
            "global_step": global_step,
            "epoch": last_epoch if last_epoch >= 0 else start_epoch,
            "batch_in_epoch": 0,
            "max_steps": max_steps,
            "completed": bool(global_step >= max_steps),
        }
        self._save_training_checkpoint(
            accelerator=accelerator,
            checkpoint_dir=output_dir / "checkpoints" / "state_final",
            trainer_state=final_state,
        )
        if accelerator.is_main_process:
            self._save_router_checkpoint(
                output_dir / "checkpoints" / "router_final.pt",
                accelerator.unwrap_model(train_module),
                meta=final_state,
            )

        if self.options.run_post_eval and eval_path is not None:
            eval_dataset = PromptPairDataset(eval_path)
            eval_loader = build_sharded_eval_loader(
                dataset=eval_dataset,
                collator=collator,
                batch_size=config.training.eval_batch_size,
                num_workers=config.data.num_workers,
            )
            tracker = RouterStatsTracker(
                accelerator.unwrap_model(train_module).encoder.model,
                config.model.router_name_patterns,
            )
            tracker.register()
            result = evaluate_text_encoder(
                model=train_module,
                dataloader=eval_loader,
                device=accelerator.device,
                router_tracker=tracker,
                use_bf16=config.runtime.bf16,
                use_fp16=config.runtime.fp16,
            )
            tracker.clear()

            if accelerator.is_main_process:
                post_dir = output_dir / "step2_post_eval"
                ensure_dir(post_dir)
                save_json(post_dir / "metrics.json", result["metrics"])
                save_json(post_dir / "routing_stats.json", result["routing"])
                save_json(
                    post_dir / "summary.json",
                    {
                        "sample_count": result["sample_count"],
                        "prompt_type_distribution": result["prompt_type_distribution"],
                        "from_step2": str(output_dir),
                    },
                )

        if accelerator.is_main_process:
            accelerator.print(f"[Step2-ACC] Finished. Outputs: {output_dir}")
        accelerator.end_training()


def run_step2(options: Step2Options) -> None:
    RouterContrastiveTrainer(options).run()
