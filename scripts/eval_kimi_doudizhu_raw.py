#!/usr/bin/env python3
"""Evaluate Kimi K2.6 raw doudizhu synthesis data with doudizhu metrics.

This is an offline companion to scripts/eval_doudizhu_model.py. It consumes
raw JSONL trajectories that already contain model responses and environment
step infos, then writes the same doudizhu trajectory-level metric summary
without invoking vLLM or replaying the environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


DOUDIZHU_METRICS = (
    "won",
    "total_reward",
    "episode_length",
    "projection_valid_rate",
    "click_valid_ratio",
    "rule_action_valid_rate",
    "fallback_rate",
    "model_hand_depletion_rate",
    "legal_given_projection_rate",
    "fallback_pass_rate",
    "truncated",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline evaluation for Kimi K2.6 doudizhu raw synthesis data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw-dir", default="data_synthesis/doudizhu_end_to_end_sft/raw")
    parser.add_argument("--episodes-file", default="train_episodes.jsonl")
    parser.add_argument("--steps-file", default="train_steps.jsonl")
    parser.add_argument("--output-dir", default="outputs/doudizhu_model_eval/kimi_k26_raw")
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-name", default="kimi-k2.6")
    return parser.parse_args()


def json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object in {path} at line {line_number}; got {type(row).__name__}.")
            rows.append(row)
    return rows


def write_jsonl(handle, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y")
    return False


def is_pass_action(action: Any) -> bool:
    return str(action).strip().lower() in ("pass", "不要")


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def build_sample_row(raw_step: dict[str, Any]) -> dict[str, Any]:
    info = as_dict(raw_step.get("env_info"))
    projection = as_dict(raw_step.get("projection"))
    action_parse = as_dict(raw_step.get("action_parse"))
    api = as_dict(raw_step.get("api"))

    fallback_used = safe_bool(info.get("fallback_used"))
    hand_cards_reduced = safe_float(info.get("hand_cards_reduced"), 0.0)
    reward = safe_float(raw_step.get("reward"), safe_float(info.get("reward"), 0.0))
    game_action = info.get("game_action")

    return {
        "env_name": "doudizhu",
        "episode_index": raw_step.get("episode_index"),
        "episode_seed": raw_step.get("episode_seed"),
        "step_index": raw_step.get("step_index"),
        "raw_response": raw_step.get("raw_response", ""),
        "raw_action_text": projection.get("raw_action_text") or action_parse.get("raw_text", ""),
        "raw_tool_call_text": projection.get("raw_tool_call_text") or info.get("raw_tool_call_text", ""),
        "semantic_action": info.get("semantic_action") or projection.get("semantic_action", ""),
        "game_action": game_action,
        "selected_cards": info.get("selected_cards", ""),
        "submit_kind": info.get("submit_kind"),
        "projection_valid": safe_float(info.get("projection_valid"), safe_float(projection.get("projection_valid"), 0.0)),
        "click_valid_ratio": safe_float(info.get("click_valid_ratio"), 0.0),
        "rule_action_valid": safe_float(info.get("rule_action_valid"), 0.0),
        "fallback_used": float(fallback_used),
        "hand_cards_reduced": hand_cards_reduced,
        "model_hand_cards_reduced": hand_cards_reduced if not fallback_used else 0.0,
        "fallback_pass": float(fallback_used and is_pass_action(game_action)),
        "reward": reward,
        "done": safe_bool(raw_step.get("done")),
        "won": safe_float(info.get("won"), 0.0),
        "tool_calling": safe_float(info.get("tool_calling"), safe_float(projection.get("tool_calling"), 0.0)),
        "api_ok": safe_bool(api.get("ok")),
        "api_error_type": api.get("error_type", ""),
        "info": info,
    }


def group_steps_by_episode(steps: Iterable[dict[str, Any]]) -> dict[Any, list[dict[str, Any]]]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in steps:
        grouped[row.get("episode_index")].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda item: safe_int(item.get("step_index"), 0))
    return dict(grouped)


def last_nonempty_info(sample_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    for row in reversed(sample_rows):
        info = as_dict(row.get("info"))
        if info:
            return info
    return {}


def is_truncated(raw_episode: dict[str, Any], sample_rows: Sequence[dict[str, Any]]) -> bool:
    normal_end = safe_bool(raw_episode.get("normal_end"))
    terminated_reason = raw_episode.get("terminated_reason")
    bot_limit_reached = safe_bool(raw_episode.get("bot_limit_reached"))
    if bot_limit_reached:
        return True
    if terminated_reason is not None and terminated_reason != "done":
        return True
    if raw_episode and not normal_end:
        return True
    last_info = last_nonempty_info(sample_rows)
    return bool(sample_rows and sample_rows[-1].get("done") is False and last_info.get("winner_id") is None)


def build_episode_row(raw_episode: dict[str, Any], sample_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    length = safe_int(raw_episode.get("episode_length"), len(sample_rows)) if raw_episode else len(sample_rows)
    denom = max(length, 1)

    projection_valid_sum = sum(safe_float(row.get("projection_valid"), 0.0) for row in sample_rows)
    click_valid_sum = sum(safe_float(row.get("click_valid_ratio"), 0.0) for row in sample_rows)
    rule_action_valid_sum = sum(safe_float(row.get("rule_action_valid"), 0.0) for row in sample_rows)
    fallback_sum = sum(safe_float(row.get("fallback_used"), 0.0) for row in sample_rows)
    hand_depletion_sum = sum(safe_float(row.get("hand_cards_reduced"), 0.0) for row in sample_rows)
    model_hand_depletion_sum = sum(safe_float(row.get("model_hand_cards_reduced"), 0.0) for row in sample_rows)
    fallback_pass_sum = sum(safe_float(row.get("fallback_pass"), 0.0) for row in sample_rows)
    tool_calling_sum = sum(safe_float(row.get("tool_calling"), 0.0) for row in sample_rows)
    step_reward_sum = sum(safe_float(row.get("reward"), 0.0) for row in sample_rows)
    last_info = last_nonempty_info(sample_rows)

    total_reward = safe_float(raw_episode.get("total_reward"), step_reward_sum) if raw_episode else step_reward_sum
    won = safe_float(raw_episode.get("won"), safe_float(last_info.get("won"), 0.0)) if raw_episode else safe_float(last_info.get("won"), 0.0)

    return {
        "env_name": "doudizhu",
        "episode_index": raw_episode.get("episode_index") if raw_episode else sample_rows[0].get("episode_index"),
        "wave_index": 0,
        "episode_seed": raw_episode.get("episode_seed") if raw_episode else sample_rows[0].get("episode_seed"),
        "env_index": raw_episode.get("episode_index") if raw_episode else sample_rows[0].get("episode_index"),
        "won": won,
        "total_reward": total_reward,
        "episode_length": length,
        "projection_valid_rate": projection_valid_sum / denom,
        "click_valid_ratio": click_valid_sum / denom,
        "rule_action_valid_rate": rule_action_valid_sum / denom,
        "fallback_rate": fallback_sum / denom,
        "hand_depletion_rate": hand_depletion_sum / 20.0,
        "model_hand_depletion_rate": model_hand_depletion_sum / 20.0,
        "legal_given_projection_rate": rule_action_valid_sum / projection_valid_sum if projection_valid_sum > 0 else 0.0,
        "fallback_pass_rate": fallback_pass_sum / denom,
        "tool_calling": tool_calling_sum,
        "truncated": float(is_truncated(raw_episode, sample_rows)),
        "winner_id": raw_episode.get("winner_id") if raw_episode else last_info.get("winner_id"),
        "payoffs": raw_episode.get("payoffs") if raw_episode else last_info.get("payoffs"),
        "final_state_summary": last_info.get("state_summary"),
    }


def bootstrap_ci(values: Sequence[float], iters: int, rng: np.random.Generator) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return math.nan, math.nan
    if len(arr) == 1 or iters <= 0:
        return float(arr[0]), float(arr[0])
    samples = rng.choice(arr, size=(iters, len(arr)), replace=True)
    boot = samples.mean(axis=1)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def metric_summary(
    env_name: str,
    unit_name: str,
    rows: Sequence[dict[str, Any]],
    metrics: Sequence[str],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(args.seed)
    out = []
    for metric in metrics:
        values = [safe_float(row.get(metric), math.nan) for row in rows]
        finite_values = [value for value in values if math.isfinite(value)]
        ci_low, ci_high = bootstrap_ci(values, args.bootstrap_iters, rng)
        out.append(
            {
                "env_name": env_name,
                "unit": unit_name,
                "metric": metric,
                "num_units": len(finite_values),
                "mean": float(np.mean(finite_values)) if finite_values else math.nan,
                "std": float(np.std(finite_values)) if finite_values else math.nan,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
            }
        )
    return out


def validate_inputs(raw_dir: Path, episodes_path: Path, steps_path: Path) -> None:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory does not exist: {raw_dir}")
    if not episodes_path.exists():
        raise FileNotFoundError(f"Episodes file does not exist: {episodes_path}")
    if not steps_path.exists():
        raise FileNotFoundError(f"Steps file does not exist: {steps_path}")


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    episodes_path = raw_dir / args.episodes_file
    steps_path = raw_dir / args.steps_file
    validate_inputs(raw_dir, episodes_path, steps_path)

    raw_episodes = read_jsonl(episodes_path)
    raw_steps = read_jsonl(steps_path)
    steps_by_episode = group_steps_by_episode(raw_steps)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_rows = [build_sample_row(raw_step) for raw_step in raw_steps]
    sample_rows_by_episode = group_steps_by_episode(sample_rows)

    episode_rows: list[dict[str, Any]] = []
    seen_episode_indices = set()
    for raw_episode in raw_episodes:
        episode_index = raw_episode.get("episode_index")
        if episode_index in seen_episode_indices:
            raise ValueError(f"Duplicate episode_index in {episodes_path}: {episode_index}")
        seen_episode_indices.add(episode_index)
        episode_rows.append(build_episode_row(raw_episode, sample_rows_by_episode.get(episode_index, [])))

    for episode_index, rows in sample_rows_by_episode.items():
        if episode_index not in seen_episode_indices:
            episode_rows.append(build_episode_row({}, rows))

    config = {
        "model_name": args.model_name,
        "env": "doudizhu",
        "raw_dir": str(raw_dir),
        "episodes_file": str(episodes_path),
        "steps_file": str(steps_path),
        "output_dir": str(output_dir),
        "num_raw_episodes": len(raw_episodes),
        "num_raw_steps": len(raw_steps),
        "num_grouped_step_episodes": len(steps_by_episode),
        "bootstrap_iters": args.bootstrap_iters,
        "seed": args.seed,
        "metrics": DOUDIZHU_METRICS,
        "source_eval_script": "scripts/eval_doudizhu_model.py",
        "notes": "Offline projection/environment fields are read from raw Kimi synthesis JSONL; no model calls are made.",
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )

    with (output_dir / "samples.jsonl").open("w", encoding="utf-8") as sample_handle:
        for row in sample_rows:
            write_jsonl(sample_handle, row)

    with (output_dir / "episodes.jsonl").open("w", encoding="utf-8") as episode_handle:
        for row in episode_rows:
            write_jsonl(episode_handle, row)

    summary_rows = metric_summary("doudizhu", "trajectory", episode_rows, DOUDIZHU_METRICS, args)
    write_csv(output_dir / "episode_metrics.csv", episode_rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary_rows, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )

    print(json.dumps(summary_rows, indent=2, ensure_ascii=False, default=json_default), flush=True)
    print(f"Wrote samples to {output_dir / 'samples.jsonl'}", flush=True)
    print(f"Wrote episodes to {output_dir / 'episodes.jsonl'}", flush=True)
    print(f"Wrote summary to {output_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
