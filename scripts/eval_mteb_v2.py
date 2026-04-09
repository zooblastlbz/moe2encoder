#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Iterable
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the local MoE text encoder on MTEB v2 benchmarks or tasks."
    )
    parser.add_argument("--config", type=str, required=True, help="Experiment config path.")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory used for cached MTEB outputs and exported metadata.",
    )
    parser.add_argument(
        "--router_ckpt",
        type=str,
        default=None,
        help="Optional router checkpoint exported by Step2, e.g. router_final.pt.",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="MTEB(eng, v2)",
        help="Benchmark name passed to mteb.get_benchmark(...). Ignored when --tasks is set.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Optional explicit task names. When set, only these tasks are evaluated.",
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=None,
        help="Optional language filter used with --tasks, e.g. --languages eng zho.",
    )
    parser.add_argument(
        "--task_types",
        type=str,
        nargs="+",
        default=None,
        help="Optional task type filter used with --tasks, e.g. Retrieval STS.",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "lasttoken"],
        help="SentenceTransformer pooling mode.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size forwarded to SentenceTransformer.encode(...).",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="Optional max sequence length override.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional SentenceTransformer device, e.g. cuda, cuda:0, cpu.",
    )
    parser.add_argument(
        "--normalize_embeddings",
        action="store_true",
        help="Pass normalize_embeddings=True to SentenceTransformer.encode(...).",
    )
    parser.add_argument(
        "--query_prefix",
        type=str,
        default=None,
        help="Optional prompt prefix for query encodes in retrieval-style tasks.",
    )
    parser.add_argument(
        "--passage_prefix",
        type=str,
        default=None,
        help="Optional prompt prefix for passage/document encodes.",
    )
    parser.add_argument(
        "--default_prompt_name",
        type=str,
        default=None,
        help="Optional SentenceTransformer default_prompt_name.",
    )
    parser.add_argument(
        "--prediction_folder",
        type=str,
        default=None,
        help="Optional folder for saved MTEB predictions.",
    )
    parser.add_argument(
        "--overwrite_strategy",
        type=str,
        default="only-missing",
        help="MTEB overwrite strategy, e.g. always or only-missing.",
    )
    parser.add_argument(
        "--safe_serialization",
        action="store_true",
        help="Save temporary backbone weights with safetensors when supported.",
    )
    return parser.parse_args()


def _ensure_mteb():
    try:
        import mteb
    except ImportError as exc:
        raise ImportError(
            "mteb is required for evaluation. Install it with: "
            "pip install 'mteb>=2.2.0' sentence-transformers"
        ) from exc
    return mteb


def _resolve_tasks(mteb_module, args: argparse.Namespace):
    def _as_task_list(value: Any) -> list[Any]:
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return [value]

    if args.tasks:
        kwargs: dict[str, Any] = {}
        if args.languages:
            kwargs["languages"] = args.languages
        if args.task_types:
            kwargs["task_types"] = args.task_types

        if hasattr(mteb_module, "get_tasks"):
            return _as_task_list(mteb_module.get_tasks(tasks=args.tasks, **kwargs))
        if hasattr(mteb_module, "get_task"):
            return _as_task_list(mteb_module.get_task(args.tasks, **kwargs))
        raise AttributeError("The installed mteb package does not expose get_tasks/get_task.")

    if args.languages or args.task_types:
        print(
            "Warning: --languages and --task_types are ignored when --benchmark is used. "
            "Pass explicit task names via --tasks to filter tasks.",
            file=sys.stderr,
        )

    benchmark = mteb_module.get_benchmark(args.benchmark)
    tasks = getattr(benchmark, "tasks", None)
    if tasks is None:
        if isinstance(benchmark, Iterable):
            return list(benchmark)
        raise RuntimeError(
            f"Unable to extract tasks from benchmark {args.benchmark!r}. "
            "The installed mteb version may expose a different benchmark API."
        )
    return list(tasks)


def _set_sentence_transformer_prompts(
    st_model,
    *,
    query_prefix: str | None,
    passage_prefix: str | None,
    default_prompt_name: str | None,
) -> dict[str, str]:
    prompts: dict[str, str] = {}
    if query_prefix is not None:
        prompts["query"] = query_prefix
    if passage_prefix is not None:
        prompts["passage"] = passage_prefix

    if prompts:
        existing = getattr(st_model, "prompts", None)
        if existing is None:
            st_model.prompts = {}
            existing = st_model.prompts
        existing.update(prompts)

    if default_prompt_name is not None:
        st_model.default_prompt_name = default_prompt_name

    return prompts


def _task_name(task: Any) -> str:
    metadata = getattr(task, "metadata", None)
    for candidate in (
        getattr(task, "name", None),
        getattr(metadata, "name", None),
        getattr(task, "__class__", type(task)).__name__,
    ):
        if candidate:
            return str(candidate)
    return str(task)


def _json_ready(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _json_ready(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict"):
        try:
            return _json_ready(value.to_dict())
        except TypeError:
            pass
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(v) for v in value]
    if hasattr(value, "model_dump"):
        return _json_ready(value.model_dump())
    if hasattr(value, "dict"):
        try:
            return _json_ready(value.dict())
        except TypeError:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_ready(payload), f, ensure_ascii=False, indent=2)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mteb = _ensure_mteb()
    from src.evaluation.sentence_transformer_export import load_sentence_transformer_from_config

    st_model, export_meta, tmp_dir = load_sentence_transformer_from_config(
        config_path=args.config,
        router_ckpt=args.router_ckpt,
        pooling=args.pooling,
        max_seq_length=args.max_seq_length,
        safe_serialization=args.safe_serialization,
        device=args.device,
    )

    try:
        prompts = _set_sentence_transformer_prompts(
            st_model,
            query_prefix=args.query_prefix,
            passage_prefix=args.passage_prefix,
            default_prompt_name=args.default_prompt_name,
        )
        tasks = _resolve_tasks(mteb, args)
        task_names = [_task_name(task) for task in tasks]

        encode_kwargs: dict[str, Any] = {"batch_size": args.batch_size}
        if args.normalize_embeddings:
            encode_kwargs["normalize_embeddings"] = True

        eval_kwargs: dict[str, Any] = {
            "tasks": tasks,
            "encode_kwargs": encode_kwargs,
            "overwrite_strategy": args.overwrite_strategy,
        }
        if hasattr(mteb, "ResultCache"):
            eval_kwargs["cache"] = mteb.ResultCache(cache_path=str(output_dir / "cache"))

        prediction_folder = args.prediction_folder
        if prediction_folder is not None:
            prediction_dir = Path(prediction_folder)
            if not prediction_dir.is_absolute():
                prediction_dir = output_dir / prediction_dir
            prediction_dir.mkdir(parents=True, exist_ok=True)
            eval_kwargs["prediction_folder"] = str(prediction_dir)

        results = mteb.evaluate(st_model, **eval_kwargs)

        metadata = {
            "benchmark": None if args.tasks else args.benchmark,
            "explicit_tasks": args.tasks,
            "resolved_tasks": task_names,
            "languages": args.languages,
            "task_types": args.task_types,
            "pooling": args.pooling,
            "batch_size": args.batch_size,
            "max_seq_length": export_meta["max_seq_length"],
            "device": args.device,
            "normalize_embeddings": args.normalize_embeddings,
            "overwrite_strategy": args.overwrite_strategy,
            "prompts": prompts,
            "default_prompt_name": args.default_prompt_name,
            "prediction_folder": args.prediction_folder,
            "model_export": export_meta,
        }
        _write_json(output_dir / "mteb_eval_meta.json", metadata)
        _write_json(output_dir / "mteb_results.json", results)

        print(f"MTEB evaluation finished. Output directory: {output_dir}")
        print(f"Tasks evaluated: {len(task_names)}")
        print(f"Pooling: {args.pooling}")
        if prompts:
            print(f"Registered prompts: {', '.join(sorted(prompts))}")
        return 0
    finally:
        tmp_dir.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
