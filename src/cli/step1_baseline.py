from __future__ import annotations

import argparse

from src.trainers.step1_baseline_trainer import Step1Options, run_step1


def parse_args():
    parser = argparse.ArgumentParser("Step1 Frozen Baseline")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--eval_jsonl", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    run_step1(
        Step1Options(
            config_path=args.config,
            output_dir=args.output_dir,
            eval_jsonl=args.eval_jsonl,
        )
    )


if __name__ == "__main__":
    main()
