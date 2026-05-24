#!/usr/bin/env python3
"""Synthesize visual QA SFT data for the Dou Dizhu GUI environment."""

from __future__ import annotations

import argparse
import json
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_synthesis.doudizhu_qa.sampler import normalize_weights, synthesize_split  # noqa: E402
from data_synthesis.doudizhu_qa.schemas import GenerationConfig  # noqa: E402
from data_synthesis.doudizhu_qa.task_specs import build_task_specs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate verified Dou Dizhu QA SFT parquet files.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=15000)
    parser.add_argument("--val-samples", type=int, default=2000)
    parser.add_argument("--test-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--language", choices=("zh", "en"), default="zh")
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--max-bot-turns", type=int, default=256)
    parser.add_argument("--n-all", type=int, default=3, help="Task K emits all legal actions only when len(actions) <= this value.")
    parser.add_argument("--list-k", type=int, default=4, help="Task L asks for exactly this many legal actions.")
    parser.add_argument("--max-tasks-per-state", type=int, default=3)
    parser.add_argument("--max-episodes", type=int, default=100000)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument(
        "--disable-label-quotas",
        action="store_true",
        help="Disable within-task label quota steering for tasks F/H2/I/K/M.",
    )
    parser.add_argument(
        "--task-weights",
        type=str,
        default=None,
        help="Optional JSON object mapping task_id to synthesis proportion.",
    )
    return parser.parse_args()


def make_env_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "doudizhu": {
            "use_ray": False,
            "language": args.language,
            "chinese_mode": args.language == "zh",
            "image_width": args.image_width,
            "image_height": args.image_height,
            "max_clicks": 8,
            "max_bot_turns": args.max_bot_turns,
            "reward": {
                "projection_valid": 0.05,
                "click_valid": 0.05,
                "rule_action_valid": 0.10,
                "hand_depletion": 0.01,
                "win": 1.0,
                "loss": -1.0,
            },
        }
    }


def image_to_png_dict(image: np.ndarray) -> dict[str, bytes]:
    buffer = BytesIO()
    Image.fromarray(image.astype(np.uint8), mode="RGB").save(buffer, format="PNG")
    return {"bytes": buffer.getvalue()}


def write_split(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path, index=False)


def main() -> None:
    args = parse_args()
    if args.train_samples < 0 or args.val_samples < 0 or args.test_samples < 0:
        raise ValueError("Sample counts must be non-negative.")
    if args.n_all <= 0 or args.list_k <= 0:
        raise ValueError("--n-all and --list-k must be positive.")
    if args.max_tasks_per_state <= 0:
        raise ValueError("--max-tasks-per-state must be positive.")

    raw_task_weights = json.loads(args.task_weights) if args.task_weights else None
    task_specs = build_task_specs()
    task_weights = normalize_weights(task_specs, raw_task_weights)
    generation_config = GenerationConfig(
        language=args.language,
        n_all=args.n_all,
        list_k=args.list_k,
        label_quotas_enabled=not args.disable_label_quotas,
    )
    env_config = make_env_config(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split_specs = [
        ("train", args.train_samples, args.seed),
        ("val", args.val_samples, args.seed + 10_000_000),
        ("test", args.test_samples, args.seed + 20_000_000),
    ]
    all_metadata: dict[str, Any] = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "data_source": "doudizhu_qa_sft",
        "format": {
            "response": "answer-only tasks use <answer>; rule tasks use <plan> then <answer>; <answer> is JSON",
            "image_storage": "images=[{'bytes': PNG_BYTES}]",
            "n_all": args.n_all,
            "list_k": args.list_k,
            "label_quotas": "enabled by default for F/H2/I/K/M; use --disable-label-quotas to restore task-only quotas",
        },
        "task_weights": task_weights,
        "tasks": {spec.task_id: {"name": spec.task_name, "requires_plan": spec.requires_plan} for spec in task_specs},
        "splits": {},
    }

    for split, target_samples, seed_start in split_specs:
        if target_samples == 0:
            continue
        result = synthesize_split(
            split=split,
            target_samples=target_samples,
            seed_start=seed_start,
            env_config=env_config,
            generation_config=generation_config,
            image_encoder=image_to_png_dict,
            max_tasks_per_state=args.max_tasks_per_state,
            max_episodes=args.max_episodes,
            raw_task_weights=raw_task_weights,
            log_every=args.log_every,
            data_source="doudizhu_qa_sft",
        )
        write_split(result.rows, args.output_dir / f"{split}.parquet")
        all_metadata["splits"][split] = result.metadata
        with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(all_metadata, f, ensure_ascii=False, indent=2)
        print(f"{split}: wrote {len(result.rows)} rows to {args.output_dir / f'{split}.parquet'}", flush=True)

    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
