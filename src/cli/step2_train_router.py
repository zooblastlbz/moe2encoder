from __future__ import annotations

import argparse

from src.trainers.router_contrastive_trainer import Step2Options, run_step2


def parse_args():
    parser = argparse.ArgumentParser("Step2 Router-Only Contrastive Training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--train_jsonl", type=str, default=None)
    parser.add_argument("--eval_jsonl", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--run_post_eval", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    run_step2(
        Step2Options(
            config_path=args.config,
            output_dir=args.output_dir,
            train_jsonl=args.train_jsonl,
            eval_jsonl=args.eval_jsonl,
            run_post_eval=args.run_post_eval,
            resume_from=args.resume_from,
        )
    )


if __name__ == "__main__":
    main()
