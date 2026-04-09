from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the current MoE text encoder as a SentenceTransformer package."
    )
    parser.add_argument("--config", type=str, required=True, help="Experiment config path.")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for the exported SentenceTransformer package.",
    )
    parser.add_argument(
        "--router_ckpt",
        type=str,
        default=None,
        help="Optional router checkpoint exported by Step2, e.g. router_final.pt.",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "lasttoken"],
        help="SentenceTransformer pooling mode.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="Optional max sequence length override for the exported model.",
    )
    parser.add_argument(
        "--safe_serialization",
        action="store_true",
        help="Save the backbone with safetensors when supported.",
    )
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    from src.evaluation.sentence_transformer_export import export_sentence_transformer_from_config

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use a new directory for export."
        )
    export_meta = export_sentence_transformer_from_config(
        config_path=args.config,
        output_dir=output_dir,
        router_ckpt=args.router_ckpt,
        pooling=args.pooling,
        max_seq_length=args.max_seq_length,
        safe_serialization=args.safe_serialization,
        device="cpu",
    )
    with (output_dir / "moe2encoder_export_meta.json").open("w", encoding="utf-8") as f:
        json.dump(export_meta, f, ensure_ascii=False, indent=2)

    print(f"Exported SentenceTransformer model to: {output_dir}")
    print(f"Pooling: {args.pooling}")
    router_info = export_meta["router_checkpoint"]
    if router_info is not None:
        print(
            "Router checkpoint loaded: "
            f"loaded={router_info['loaded_router_tensors']}, "
            f"missing={router_info['missing_router_tensors']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
