#!/usr/bin/env bash
set -euo pipefail

ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate/deepspeed_zero2.yaml}"
ACCELERATE_EXTRA_ARGS="${ACCELERATE_EXTRA_ARGS:-}"

if [ $# -eq 0 ]; then
  set -- --config configs/train/router_only_qwen35a3b.yaml --run_post_eval
fi

EXTRA_ARGS=()
if [ -n "${ACCELERATE_EXTRA_ARGS}" ]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=(${ACCELERATE_EXTRA_ARGS})
fi

accelerate launch --config_file "${ACCELERATE_CONFIG}" "${EXTRA_ARGS[@]}" -m src.cli.step2_train_router "$@"
