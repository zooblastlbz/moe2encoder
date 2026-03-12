from __future__ import annotations

import argparse

from src.trainers.step3_analysis_runner import Step3Options, run_step3


def parse_args():
    parser = argparse.ArgumentParser("Step3 Post-Training Analysis")
    parser.add_argument("--baseline_dir", type=str, required=True)
    parser.add_argument("--post_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    run_step3(
        Step3Options(
            baseline_dir=args.baseline_dir,
            post_dir=args.post_dir,
            output_dir=args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
