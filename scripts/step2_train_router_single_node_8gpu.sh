#!/usr/bin/env bash
set -euo pipefail

# Single-node 8-GPU launcher for Step2 router-only contrastive training.
# Edit fixed parameters here:
ACCELERATE_CONFIG="configs/accelerate/deepspeed_zero2.yaml"
CONFIG_PATH="configs/train/router_only_qwen35a3b.yaml"
NUM_MACHINES=1
MACHINE_RANK=0
NUM_PROCESSES=8
MAIN_PROCESS_PORT=29500
RUN_POST_EVAL=1
# Optional: set to checkpoint directory path; keep empty to start fresh.
RESUME_FROM=""
# Optional: override dataset paths; keep empty to use config file values.
TRAIN_JSONL=""
EVAL_JSONL=""

POST_EVAL_ARGS=()
if [ "${RUN_POST_EVAL}" -eq 1 ]; then
  POST_EVAL_ARGS=(--run_post_eval)
fi

RESUME_ARGS=()
if [ -n "${RESUME_FROM}" ]; then
  RESUME_ARGS=(--resume_from "${RESUME_FROM}")
fi

TRAIN_DATA_ARGS=()
if [ -n "${TRAIN_JSONL}" ]; then
  TRAIN_DATA_ARGS+=(--train_jsonl "${TRAIN_JSONL}")
fi
if [ -n "${EVAL_JSONL}" ]; then
  TRAIN_DATA_ARGS+=(--eval_jsonl "${EVAL_JSONL}")
fi

accelerate launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_machines "${NUM_MACHINES}" \
  --machine_rank "${MACHINE_RANK}" \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  -m src.cli.step2_train_router \
  --config "${CONFIG_PATH}" \
  "${POST_EVAL_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  "${TRAIN_DATA_ARGS[@]}"
