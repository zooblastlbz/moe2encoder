#!/usr/bin/env bash
set -euo pipefail

if [ $# -eq 0 ]; then
  set -- --config configs/train/router_only_qwen35a3b.yaml
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

if [ "${NPROC_PER_NODE}" -gt 1 ]; then
  torchrun --nproc_per_node "${NPROC_PER_NODE}" -m src.cli.step1_baseline "$@"
else
  python3 -m src.cli.step1_baseline "$@"
fi
