#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# =========================
# Fixed parameters (edit here)
# =========================
CONFIG_PATH="configs/train/router_only_qwen35a3b.yaml"
OUTPUT_ROOT="outputs/qwen35a3b_router_only"

RUN_STEP1=1
RUN_STEP2=1
RUN_STEP3=1

STEP1_NPROC_PER_NODE=8
STEP1_OUTPUT_DIR="${OUTPUT_ROOT}/step1_baseline"
# Optional: keep empty to use config value.
STEP1_EVAL_JSONL=""

ACCELERATE_CONFIG="configs/accelerate/deepspeed_zero2.yaml"
STEP2_NUM_MACHINES=1
STEP2_MACHINE_RANK=0
STEP2_NUM_PROCESSES=8
STEP2_MAIN_PROCESS_PORT=29500
STEP2_OUTPUT_DIR="${OUTPUT_ROOT}/step2_train"
STEP2_RUN_POST_EVAL=1
# Optional: set checkpoint state dir to resume, e.g. outputs/.../state_step_1000
STEP2_RESUME_FROM=""
# Optional: keep empty to use config values.
STEP2_TRAIN_JSONL=""
STEP2_EVAL_JSONL=""

STEP2_POST_EVAL_DIR="${STEP2_OUTPUT_DIR}/step2_post_eval"
STEP3_OUTPUT_DIR="${OUTPUT_ROOT}/step3_analysis"

echo "[Pipeline] repo_root=${REPO_ROOT}"
echo "[Pipeline] config=${CONFIG_PATH}"

if [ "${RUN_STEP1}" -eq 1 ]; then
  echo "[Pipeline] Step1 baseline start..."
  STEP1_ARGS=(
    --config "${CONFIG_PATH}"
    --output_dir "${STEP1_OUTPUT_DIR}"
  )
  if [ -n "${STEP1_EVAL_JSONL}" ]; then
    STEP1_ARGS+=(--eval_jsonl "${STEP1_EVAL_JSONL}")
  fi

  if [ "${STEP1_NPROC_PER_NODE}" -gt 1 ]; then
    torchrun --nproc_per_node "${STEP1_NPROC_PER_NODE}" -m src.cli.step1_baseline "${STEP1_ARGS[@]}"
  else
    python3 -m src.cli.step1_baseline "${STEP1_ARGS[@]}"
  fi
  echo "[Pipeline] Step1 baseline done."
fi

if [ "${RUN_STEP2}" -eq 1 ]; then
  echo "[Pipeline] Step2 train start..."
  STEP2_ARGS=(
    --config "${CONFIG_PATH}"
    --output_dir "${STEP2_OUTPUT_DIR}"
  )
  if [ "${STEP2_RUN_POST_EVAL}" -eq 1 ]; then
    STEP2_ARGS+=(--run_post_eval)
  fi
  if [ -n "${STEP2_RESUME_FROM}" ]; then
    STEP2_ARGS+=(--resume_from "${STEP2_RESUME_FROM}")
  fi
  if [ -n "${STEP2_TRAIN_JSONL}" ]; then
    STEP2_ARGS+=(--train_jsonl "${STEP2_TRAIN_JSONL}")
  fi
  if [ -n "${STEP2_EVAL_JSONL}" ]; then
    STEP2_ARGS+=(--eval_jsonl "${STEP2_EVAL_JSONL}")
  fi

  accelerate launch \
    --config_file "${ACCELERATE_CONFIG}" \
    --num_machines "${STEP2_NUM_MACHINES}" \
    --machine_rank "${STEP2_MACHINE_RANK}" \
    --num_processes "${STEP2_NUM_PROCESSES}" \
    --main_process_port "${STEP2_MAIN_PROCESS_PORT}" \
    -m src.cli.step2_train_router \
    "${STEP2_ARGS[@]}"
  echo "[Pipeline] Step2 train done."
fi

if [ "${RUN_STEP3}" -eq 1 ]; then
  echo "[Pipeline] Step3 analysis start..."
  python3 -m src.cli.step3_analysis \
    --baseline_dir "${STEP1_OUTPUT_DIR}" \
    --post_dir "${STEP2_POST_EVAL_DIR}" \
    --output_dir "${STEP3_OUTPUT_DIR}"
  echo "[Pipeline] Step3 analysis done."
fi

echo "[Pipeline] All enabled steps completed."
