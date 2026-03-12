# moe2encoder

Router-only contrastive fine-tuning toolkit for MoE text encoders.

Current implementation scope:

- Step1: frozen baseline evaluation.
- Step2: router-only contrastive training.
- Step3: post-training metric and routing comparison.

## Project structure

- `scripts/` only contains CLI entry points.
- Startup commands are provided as `.sh` wrappers in `scripts/`.
- Core logic lives in `src/trainers/`:
  - `step1_baseline_trainer.py`
  - `router_contrastive_trainer.py` (Accelerate + DeepSpeed ZeRO-2)
  - `step3_analysis_runner.py`

## Implemented features

- Support loading `Qwen/Qwen3.5-35B-A3B` via HuggingFace `transformers`.
- Freeze all non-router parameters using configurable router name patterns.
- Symmetric InfoNCE contrastive objective (`anchor->positive` and `positive->anchor`).
- Step2 trainer backend: `Accelerate + DeepSpeed ZeRO-2`.
- CLIP-style large-batch training tricks:
  - cross-device negatives via distributed all-gather;
  - gradient accumulation;
  - mixed precision (`bf16` / `fp16`);
  - gradient checkpointing;
  - learnable `logit_scale`;
  - optional feature queue for extra negatives.
- Router statistics collection with entropy and expert usage distribution.

## Install

```bash
pip install -r requirements.txt
```

## Data format

See [dataset_schema.md](/Users/libozhou/Desktop/moe2encoder/configs/data/dataset_schema.md).

## Run Step1 (frozen baseline)

```bash
NPROC_PER_NODE=8 ./scripts/step1_baseline.sh \
  --config configs/train/router_only_qwen35a3b.yaml
```

Outputs:

- `outputs/.../step1_baseline/metrics.json`
- `outputs/.../step1_baseline/routing_stats.json`

## Run Step2 (router-only training)

```bash
ACCELERATE_CONFIG=configs/accelerate/deepspeed_zero2.yaml \
  ./scripts/step2_train_router.sh \
  --config configs/train/router_only_qwen35a3b.yaml \
  --run_post_eval
```

Outputs:

- `outputs/.../step2_train/checkpoints/router_final.pt`
- `outputs/.../step2_train/train_log.jsonl`
- `outputs/.../step2_train/step2_post_eval/*.json` (when `--run_post_eval`)

## Run Step3 (post analysis)

```bash
./scripts/step3_analysis.sh \
  --baseline_dir outputs/qwen35a3b_router_only/step1_baseline \
  --post_dir outputs/qwen35a3b_router_only/step2_train/step2_post_eval \
  --output_dir outputs/qwen35a3b_router_only/step3_analysis
```

Outputs:

- `metrics_diff.json`
- `routing_diff.json`
- `analysis_report.md`

## Notes for Qwen3.5-35B-A3B

- Use `accelerate launch` for Step2 training.
- Keep `cross_device_negatives=true` to maximize effective negatives.
- Increase effective batch via `train_batch_size * gradient_accumulation_steps * world_size`.
- With distributed negatives enabled, train loader uses `drop_last=true` to keep shape consistency.

## Multi-node launch example (Accelerate + DeepSpeed)

Assume:

- `2` nodes
- `8` GPUs per node
- node0 IP is `10.0.0.1`
- same code path and data path on all nodes

Run on node0:

```bash
ACCELERATE_CONFIG=configs/accelerate/deepspeed_zero2.yaml \
  ACCELERATE_EXTRA_ARGS="--num_machines 2 --num_processes 16 --machine_rank 0 --main_process_ip 10.0.0.1 --main_process_port 29500" \
  ./scripts/step2_train_router.sh \
  --config configs/train/router_only_qwen35a3b.yaml \
  --run_post_eval
```

Run on node1:

```bash
ACCELERATE_CONFIG=configs/accelerate/deepspeed_zero2.yaml \
  ACCELERATE_EXTRA_ARGS="--num_machines 2 --num_processes 16 --machine_rank 1 --main_process_ip 10.0.0.1 --main_process_port 29500" \
  ./scripts/step2_train_router.sh \
  --config configs/train/router_only_qwen35a3b.yaml \
  --run_post_eval
```
