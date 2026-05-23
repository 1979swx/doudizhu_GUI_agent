#!/usr/bin/env python3
"""Synthesize SFT data for the doudizhu_grounding environment.

The generator is intentionally standalone: it uses the public environment and
renderer APIs, writes parquet files, and does not patch the training pipeline.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import uuid
from collections import Counter
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_system.environments.env_package.doudizhu.core.utils import CARD_TYPE  # noqa: E402
from agent_system.environments.env_package.doudizhu.renderer import HitBox  # noqa: E402
from agent_system.environments.env_package.doudizhu_grounding.envs import (  # noqa: E402
    DoudizhuGroundingSingleEnv,
    _pretty_action,
)
from agent_system.environments.env_package.doudizhu_grounding.projection import (  # noqa: E402
    doudizhu_grounding_projection,
)
from agent_system.environments.prompts.doudizhu_grounding import (  # noqa: E402
    DOUDIZHU_GROUNDING_TEMPLATE,
    DOUDIZHU_GROUNDING_TEMPLATE_ZH,
)


DEFAULT_CATEGORY_WEIGHTS = {
    "pass": 0.06,
    "solo": 0.25,
    "pair": 0.20,
    "trio": 0.14,
    "chain": 0.16,
    "bomb_rocket": 0.07,
    "other": 0.12,
}
PROJECTION_MAX_CLICKS = 128


@dataclass(frozen=True)
class GoldAction:
    target_action: str
    target_action_pretty: str
    action_category: str
    clicks: list[list[int]]
    selected_indices: list[int]
    response: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate verified doudizhu_grounding SFT parquet files.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=30000)
    parser.add_argument("--val-samples", type=int, default=5000)
    parser.add_argument("--test-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--language", choices=("zh", "en"), default="zh")
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--max-bot-turns", type=int, default=256)
    parser.add_argument(
        "--category-weights",
        type=str,
        default=None,
        help="Optional JSON object overriding action category weights.",
    )
    parser.add_argument("--jitter", type=float, default=0.20, help="Fraction of half hitbox size used for safe random jitter.")
    parser.add_argument("--log-every", type=int, default=1000, help="Print quota progress after this many accepted samples.")
    return parser.parse_args()


def make_env_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "doudizhu": {
            "use_ray": False,
            "language": args.language,
            "chinese_mode": args.language == "zh",
            "image_width": args.image_width,
            "image_height": args.image_height,
            "max_clicks": PROJECTION_MAX_CLICKS,
            "max_bot_turns": args.max_bot_turns,
            "reward": {
                "projection_valid": 0.05,
                "click_valid": 0.05,
                "rule_action_valid": 0.10,
                "hand_depletion": 0.01,
                "win": 1.0,
                "loss": -1.0,
            },
        },
        "doudizhu_grounding": {
            "use_ray": False,
            "teacher_policy": "rule_v1",
            "reward": {
                "projection_valid": 0.1,
                "click_valid": 0.2,
                "submit_correct": 0.2,
                "target_action_match": 1.0,
            },
        },
    }


def normalize_weights(raw_weights: dict[str, float]) -> dict[str, float]:
    weights = {key: float(raw_weights.get(key, 0.0)) for key in DEFAULT_CATEGORY_WEIGHTS}
    total = sum(value for value in weights.values() if value > 0)
    if total <= 0:
        raise ValueError("Category weights must contain at least one positive value.")
    return {key: max(value, 0.0) / total for key, value in weights.items()}


def quota_targets(total_samples: int, weights: dict[str, float]) -> dict[str, int]:
    if total_samples < 0:
        raise ValueError("total_samples must be non-negative.")
    if total_samples == 0:
        return {key: 0 for key in weights}

    raw_targets = {key: weights[key] * total_samples for key in weights}
    targets = {key: int(raw_targets[key]) for key in weights}
    remainder = total_samples - sum(targets.values())
    if remainder > 0:
        order = sorted(weights, key=lambda key: (raw_targets[key] - targets[key], weights[key], key), reverse=True)
        for key in order[:remainder]:
            targets[key] += 1
    return targets


def quota_complete(counts: Counter, targets: dict[str, int]) -> bool:
    return all(int(counts.get(category, 0)) >= target for category, target in targets.items())


def quota_remaining(counts: Counter, targets: dict[str, int]) -> dict[str, int]:
    return {category: max(0, target - int(counts.get(category, 0))) for category, target in targets.items()}


def action_category(action: str) -> str:
    if action == "pass":
        return "pass"

    card_types = CARD_TYPE[0].get(action)
    if not card_types:
        return "other"

    fine_type = str(card_types[0][0])
    if fine_type == "solo":
        return "solo"
    if fine_type == "pair":
        return "pair"
    if "chain" in fine_type:
        return "chain"
    if fine_type.startswith("trio"):
        return "trio"
    if fine_type in {"bomb", "rocket"}:
        return "bomb_rocket"
    return "other"


def norm_point(renderer: Any, px: float, py: float) -> list[int]:
    x = int(round(px / float(renderer.width - 1) * 1000.0))
    y = int(round(py / float(renderer.height - 1) * 1000.0))
    return [min(1000, max(0, x)), min(1000, max(0, y))]


def sample_point_in_box(renderer: Any, box: tuple[int, int, int, int], rng: random.Random, jitter: float) -> list[int]:
    x0, y0, x1, y1 = box
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    if jitter <= 0:
        return norm_point(renderer, cx, cy)

    safe = min(max(float(jitter), 0.0), 0.45)
    dx = (x1 - x0) * safe
    dy = (y1 - y0) * safe
    px = rng.uniform(cx - dx, cx + dx)
    py = rng.uniform(cy - dy, cy + dy)
    return norm_point(renderer, px, py)


def hitbox_by_kind(hitboxes: list[HitBox], kind: str) -> HitBox:
    for hitbox in hitboxes:
        if hitbox.kind == kind:
            return hitbox
    raise ValueError(f"Missing hitbox kind={kind!r}")


def hitbox_by_card_index(hitboxes: list[HitBox], index: int) -> HitBox:
    for hitbox in hitboxes:
        if hitbox.kind == "card" and hitbox.payload == index:
            return hitbox
    raise ValueError(f"Missing card hitbox for index={index}")


def selected_indices_for_action(hand: str, action: str) -> list[int]:
    used: set[int] = set()
    selected: list[int] = []
    for card in action:
        for idx, hand_card in enumerate(hand):
            if idx not in used and hand_card == card:
                used.add(idx)
                selected.append(idx)
                break
        else:
            raise ValueError(f"Action {action!r} is not contained in hand {hand!r}")
    return sorted(selected)


def gold_action_for_target(env: DoudizhuGroundingSingleEnv, target_action: str, rng: random.Random, jitter: float) -> GoldAction:
    renderer = env.renderer
    state = env.game.state
    hitboxes = renderer.get_hitboxes(state)

    if target_action == "pass":
        pass_box = hitbox_by_kind(hitboxes, "pass").box
        clicks = [sample_point_in_box(renderer, pass_box, rng, jitter)]
        selected_indices: list[int] = []
    else:
        hand = state.get("current_hand", "")
        selected_indices = selected_indices_for_action(hand, target_action)
        clicks = [
            sample_point_in_box(renderer, hitbox_by_card_index(hitboxes, idx).box, rng, jitter)
            for idx in selected_indices
        ]
        play_box = hitbox_by_kind(hitboxes, "play").box
        clicks.append(sample_point_in_box(renderer, play_box, rng, jitter))

    click_text = ",".join(f"[{x},{y}]" for x, y in clicks)
    response = f"<tool_call>left_click({click_text})</tool_call>"
    category = action_category(target_action)
    return GoldAction(
        target_action=target_action,
        target_action_pretty=_pretty_action(target_action, env.language),
        action_category=category,
        clicks=clicks,
        selected_indices=selected_indices,
        response=response,
    )


def legal_actions_by_needed_category(
    legal_actions: list[str],
    accepted_counts: Counter,
    targets: dict[str, int],
) -> dict[str, list[str]]:
    candidates_by_category: dict[str, list[str]] = {key: [] for key in targets}
    for action in legal_actions:
        category = action_category(action)
        if category not in candidates_by_category:
            category = "other"
        if int(accepted_counts.get(category, 0)) >= int(targets.get(category, 0)):
            continue
        candidates_by_category[category].append(action)
    return {key: value for key, value in candidates_by_category.items() if value}


def choose_quota_action(
    legal_actions: list[str],
    accepted_counts: Counter,
    targets: dict[str, int],
    rng: random.Random,
) -> str | None:
    available = legal_actions_by_needed_category(legal_actions, accepted_counts, targets)
    if not available:
        return None

    deficits = {}
    for category in available:
        deficits[category] = int(targets.get(category, 0)) - int(accepted_counts.get(category, 0))

    best_deficit = max(deficits.values())
    best_categories = [category for category, deficit in deficits.items() if deficit == best_deficit]
    chosen_category = rng.choice(sorted(best_categories))
    return rng.choice(sorted(available[chosen_category]))


def choose_transition_action(
    legal_actions: list[str],
    accepted_counts: Counter,
    targets: dict[str, int],
    rng: random.Random,
) -> str | None:
    if not legal_actions:
        return None

    available = legal_actions_by_needed_category(legal_actions, accepted_counts, targets)
    if available:
        deficits = {
            category: int(targets.get(category, 0)) - int(accepted_counts.get(category, 0))
            for category in available
        }
        best_deficit = max(deficits.values())
        best_categories = [category for category, deficit in deficits.items() if deficit == best_deficit]
        chosen_category = rng.choice(sorted(best_categories))
        return rng.choice(sorted(available[chosen_category]))

    return rng.choice(sorted(legal_actions))


def legal_category_set(legal_actions: list[str]) -> set[str]:
    categories = set()
    for action in legal_actions:
        category = action_category(action)
        if category not in DEFAULT_CATEGORY_WEIGHTS:
            category = "other"
        categories.add(category)
    return categories


def image_to_png_dict(image: np.ndarray) -> dict[str, bytes]:
    buffer = BytesIO()
    Image.fromarray(image.astype(np.uint8), mode="RGB").save(buffer, format="PNG")
    return {"bytes": buffer.getvalue()}


def build_prompt(language: str, command: str) -> str:
    template = DOUDIZHU_GROUNDING_TEMPLATE_ZH if language == "zh" else DOUDIZHU_GROUNDING_TEMPLATE
    return template.format(command=command)


def verify_response(env: DoudizhuGroundingSingleEnv, response: str) -> tuple[bool, dict[str, Any]]:
    actions, valids = doudizhu_grounding_projection([response], max_clicks=PROJECTION_MAX_CLICKS)
    if not valids[0]:
        return False, {"projection_valid": 0}

    reward, info = env._score_action(actions[0])
    info["reward"] = reward
    keep = (
        int(info.get("projection_valid", 0)) == 1
        and float(info.get("click_valid_ratio", 0.0)) == 1.0
        and float(info.get("submit_correct", 0.0)) == 1.0
        and float(info.get("target_action_match", 0.0)) == 1.0
        and info.get("predicted_action") == info.get("target_action")
        and not bool(info.get("done_before_scoring", False))
    )
    return keep, info


def compact_verifier_info(info: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "reward",
        "projection_valid",
        "click_valid_ratio",
        "submit_correct",
        "target_action_match",
        "predicted_action",
        "selected_cards",
        "selected_indices",
        "submit_kind",
        "tool_calling",
    )
    return {key: info.get(key) for key in keys}


def compact_trace(trace: list[Any]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in trace:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            compact.append({"player": int(item[0]), "action": str(item[1])})
        else:
            compact.append({"player": -1, "action": str(item)})
    return compact


def make_row(
    split: str,
    sample_index: int,
    episode_seed: int,
    episode_index: int,
    obs: np.ndarray,
    env: DoudizhuGroundingSingleEnv,
    gold: GoldAction,
    source_policy: str,
    verifier_info: dict[str, Any],
) -> dict[str, Any]:
    prompt = build_prompt(env.language, gold.target_action_pretty)
    state = env.game.state
    sample_id = f"doudizhu_grounding:{split}:{episode_seed}:{env.grounding_step}:{sample_index}:{uuid.uuid4().hex[:8]}"
    extra_info = {
        "sample_id": sample_id,
        "split": split,
        "sample_index": sample_index,
        "episode_seed": episode_seed,
        "episode_index": episode_index,
        "grounding_step": int(env.grounding_step),
        "source_policy": source_policy,
        "teacher_policy": "rule_v1" if source_policy == "rule" else source_policy,
        "target_action": gold.target_action,
        "target_action_pretty": gold.target_action_pretty,
        "action_category": gold.action_category,
        "hand": state.get("current_hand", ""),
        "num_cards_left": list(state.get("num_cards_left", [])),
        "legal_actions": list(state.get("actions", [])),
        "selected_indices": gold.selected_indices,
        "gold_clicks": gold.clicks,
        "trace_tail": compact_trace(list(state.get("trace", []))[-6:]),
        "verifier": compact_verifier_info(verifier_info),
    }
    return {
        "data_source": "doudizhu_grounding_sft",
        "prompt": [{"role": "user", "content": prompt}],
        "question": prompt,
        "images": [image_to_png_dict(obs)],
        "answer": gold.response,
        "ability": "agent_grounding",
        "reward_model": {"style": "rule", "ground_truth": gold.target_action},
        "extra_info": extra_info,
    }


def advance_or_reset(env: DoudizhuGroundingSingleEnv) -> tuple[np.ndarray, dict[str, Any]]:
    obs, info = env.advance_teacher()
    return obs, info


def write_split(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame(rows)
    dataframe.to_parquet(output_path, index=False)


def format_quota_progress(counts: Counter, targets: dict[str, int]) -> str:
    parts = []
    for category in targets:
        count = int(counts.get(category, 0))
        target = int(targets[category])
        suffix = " full" if count >= target else ""
        parts.append(f"{category}={count}/{target}{suffix}")
    return ", ".join(parts)


def synthesize_split(
    split: str,
    target_samples: int,
    seed_start: int,
    args: argparse.Namespace,
    weights: dict[str, float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed_start)
    env = DoudizhuGroundingSingleEnv(seed=seed_start, env_config=make_env_config(args))
    rows: list[dict[str, Any]] = []
    targets = quota_targets(target_samples, weights)
    category_counts: Counter = Counter()
    source_counts: Counter = Counter()
    rejection_counts: Counter = Counter()
    skip_counts: Counter = Counter()
    episode_index = 0
    rejected = 0
    skipped = 0
    last_logged_accepted = 0

    while not quota_complete(category_counts, targets):
        episode_seed = seed_start + episode_index
        obs, _info = env.reset(seed=episode_seed)

        while not env.done and not quota_complete(category_counts, targets):
            legal_actions = list(env.game.state.get("actions", []))
            target_action = choose_quota_action(
                legal_actions=legal_actions,
                accepted_counts=category_counts,
                targets=targets,
                rng=rng,
            )
            if target_action is None:
                legal_categories = legal_category_set(legal_actions)
                remaining = quota_remaining(category_counts, targets)
                needed_categories = {category for category, remain in remaining.items() if remain > 0}
                if not legal_actions:
                    reason = "no_legal_actions"
                elif legal_categories and legal_categories.issubset({category for category, remain in remaining.items() if remain <= 0}):
                    reason = "quota_full_for_legal_categories"
                elif legal_categories.isdisjoint(needed_categories):
                    reason = "no_needed_category"
                else:
                    reason = "no_candidate"
                skip_counts[reason] += 1
                skipped += 1
                transition_action = choose_transition_action(
                    legal_actions=legal_actions,
                    accepted_counts=category_counts,
                    targets=targets,
                    rng=rng,
                )
                if transition_action is not None:
                    env.target_action = transition_action
                obs, _next_info = advance_or_reset(env)
                continue

            try:
                env.target_action = target_action
                gold = gold_action_for_target(env, target_action, rng, args.jitter)
                keep, verifier_info = verify_response(env, gold.response)
            except Exception as exc:  # noqa: BLE001
                rejection_counts[f"exception:{type(exc).__name__}"] += 1
                rejected += 1
                obs, _next_info = advance_or_reset(env)
                continue

            if keep:
                row = make_row(
                    split=split,
                    sample_index=len(rows),
                    episode_seed=episode_seed,
                    episode_index=episode_index,
                    obs=obs,
                    env=env,
                    gold=gold,
                    source_policy="quota_sample",
                    verifier_info=verifier_info,
                )
                rows.append(row)
                category_counts[gold.action_category] += 1
                source_counts["quota_sample"] += 1
            else:
                reason = "verifier_reject"
                if int(verifier_info.get("projection_valid", 0)) != 1:
                    reason = "projection_invalid"
                elif float(verifier_info.get("click_valid_ratio", 0.0)) != 1.0:
                    reason = "click_invalid"
                elif float(verifier_info.get("submit_correct", 0.0)) != 1.0:
                    reason = "submit_incorrect"
                elif float(verifier_info.get("target_action_match", 0.0)) != 1.0:
                    reason = "target_mismatch"
                rejection_counts[reason] += 1
                rejected += 1

            transition_action = choose_transition_action(
                legal_actions=list(env.game.state.get("actions", [])),
                accepted_counts=category_counts,
                targets=targets,
                rng=rng,
            )
            if transition_action is not None:
                env.target_action = transition_action
            obs, _next_info = advance_or_reset(env)

            if args.log_every > 0 and len(rows) > 0 and len(rows) % args.log_every == 0 and len(rows) != last_logged_accepted:
                last_logged_accepted = len(rows)
                print(
                    f"{split}: accepted {len(rows)}/{target_samples}, rejected={rejected}, skipped={skipped}\n"
                    f"  quota: {format_quota_progress(category_counts, targets)}",
                    flush=True,
                )

        episode_index += 1

    metadata = {
        "split": split,
        "target_samples": target_samples,
        "accepted_samples": len(rows),
        "episodes_used": episode_index,
        "seed_start": seed_start,
        "quota_targets": targets,
        "quota_remaining": quota_remaining(category_counts, targets),
        "category_counts": dict(category_counts),
        "source_counts": dict(source_counts),
        "rejection_counts": dict(rejection_counts),
        "skip_counts": dict(skip_counts),
    }
    return rows, metadata


def main() -> None:
    args = parse_args()
    if args.train_samples < 0 or args.val_samples < 0 or args.test_samples < 0:
        raise ValueError("Sample counts must be non-negative.")

    raw_weights = DEFAULT_CATEGORY_WEIGHTS if args.category_weights is None else json.loads(args.category_weights)
    weights = normalize_weights(raw_weights)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split_specs = [
        ("train", args.train_samples, args.seed),
        ("val", args.val_samples, args.seed + 10_000_000),
        ("test", args.test_samples, args.seed + 20_000_000),
    ]

    all_metadata: dict[str, Any] = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "category_weights": weights,
        "format": {
            "response": "single <tool_call> tag only",
            "image_storage": "images=[{'bytes': PNG_BYTES}]",
            "projection_max_clicks": PROJECTION_MAX_CLICKS,
        },
        "splits": {},
    }

    for split, target_samples, seed_start in split_specs:
        if target_samples == 0:
            continue
        rows, metadata = synthesize_split(split, target_samples, seed_start, args, weights)
        write_split(rows, args.output_dir / f"{split}.parquet")
        all_metadata["splits"][split] = metadata
        with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(all_metadata, f, ensure_ascii=False, indent=2)
        print(f"{split}: wrote {len(rows)} rows to {args.output_dir / f'{split}.parquet'}", flush=True)

    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
