#!/usr/bin/env bash
set -euo pipefail

if [ $# -eq 0 ]; then
  set -- \
    --baseline_dir outputs/qwen35a3b_router_only/step1_baseline \
    --post_dir outputs/qwen35a3b_router_only/step2_train/step2_post_eval \
    --output_dir outputs/qwen35a3b_router_only/step3_analysis
fi

python3 -m src.cli.step3_analysis "$@"
