"""Rollout sampler for Dou Dizhu QA SFT synthesis."""

from __future__ import annotations

import random
import uuid
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from agent_system.environments.env_package.doudizhu.envs import DoudizhuSingleEnv

from .canonical import (
    action_sort_key,
    rarity_tags,
    raw_action_to_display_list,
)
from .schemas import GeneratedSample, GenerationConfig
from .task_specs import TaskSpec, build_task_specs
from .verifier import verify_response


@dataclass(frozen=True)
class SplitSynthesisResult:
    rows: list[dict[str, Any]]
    metadata: dict[str, Any]


def normalize_weights(task_specs: list[TaskSpec], raw_weights: dict[str, float] | None = None) -> dict[str, float]:
    configured = raw_weights or {spec.task_id: spec.weight for spec in task_specs}
    weights = {spec.task_id: float(configured.get(spec.task_id, 0.0)) for spec in task_specs}
    total = sum(value for value in weights.values() if value > 0)
    if total <= 0:
        raise ValueError("Task weights must contain at least one positive value.")
    return {task_id: max(weight, 0.0) / total for task_id, weight in weights.items()}


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


def quota_complete(counts: Counter[str], targets: dict[str, int]) -> bool:
    return all(int(counts.get(task_id, 0)) >= target for task_id, target in targets.items())


def quota_remaining(counts: Counter[str], targets: dict[str, int]) -> dict[str, int]:
    return {task_id: max(0, target - int(counts.get(task_id, 0))) for task_id, target in targets.items()}


def normalize_label_weights(raw_weights: dict[str, float]) -> dict[str, float]:
    weights = {bucket: max(0.0, float(weight)) for bucket, weight in raw_weights.items()}
    total = sum(weights.values())
    if total <= 0:
        return {}
    return {bucket: weight / total for bucket, weight in weights.items()}


def label_quota_weights(task_specs: list[TaskSpec], config: GenerationConfig) -> dict[str, dict[str, float]]:
    if not config.label_quotas_enabled:
        return {}
    weights: dict[str, dict[str, float]] = {}
    for spec in task_specs:
        normalized = normalize_label_weights(spec.label_bucket_weights(config))
        if normalized:
            weights[spec.task_id] = normalized
    return weights


def label_quota_targets(
    task_targets: dict[str, int],
    label_weights: dict[str, dict[str, float]],
) -> dict[str, dict[str, int]]:
    return {
        task_id: quota_targets(int(task_targets.get(task_id, 0)), weights)
        for task_id, weights in label_weights.items()
    }


def label_quota_remaining(
    label_counts: Counter[tuple[str, str]],
    label_targets: dict[str, dict[str, int]],
) -> dict[str, dict[str, int]]:
    return {
        task_id: {
            bucket: max(0, target - int(label_counts.get((task_id, bucket), 0)))
            for bucket, target in bucket_targets.items()
        }
        for task_id, bucket_targets in label_targets.items()
    }


def nested_label_counts(
    label_counts: Counter[tuple[str, str]],
    label_targets: dict[str, dict[str, int]],
) -> dict[str, dict[str, int]]:
    nested: dict[str, dict[str, int]] = {task_id: {bucket: 0 for bucket in targets} for task_id, targets in label_targets.items()}
    for (task_id, bucket), count in label_counts.items():
        nested.setdefault(task_id, {})[bucket] = int(count)
    return nested


def best_label_deficit_for_state(
    spec: TaskSpec,
    state: dict[str, Any],
    config: GenerationConfig,
    label_counts: Counter[tuple[str, str]],
    label_targets: dict[str, dict[str, int]],
) -> tuple[float, int]:
    bucket_targets = label_targets.get(spec.task_id, {})
    if not bucket_targets:
        return (0.0, 0)
    best_ratio = 0.0
    best_deficit = 0
    for bucket in spec.label_buckets_for_state(state, config):
        target = int(bucket_targets.get(bucket, 0))
        if target <= 0:
            continue
        deficit = target - int(label_counts.get((spec.task_id, bucket), 0))
        if deficit <= 0:
            continue
        ratio = float(deficit) / float(target)
        if (ratio, deficit) > (best_ratio, best_deficit):
            best_ratio = ratio
            best_deficit = deficit
    return (best_ratio, best_deficit)


def task_has_remaining_label_quota(
    spec: TaskSpec,
    label_counts: Counter[tuple[str, str]],
    label_targets: dict[str, dict[str, int]],
) -> bool:
    return any(
        int(label_counts.get((spec.task_id, bucket), 0)) < int(target)
        for bucket, target in label_targets.get(spec.task_id, {}).items()
    )


def state_labels_are_saturated(
    spec: TaskSpec,
    state: dict[str, Any],
    config: GenerationConfig,
    label_counts: Counter[tuple[str, str]],
    label_targets: dict[str, dict[str, int]],
) -> bool:
    if not task_has_remaining_label_quota(spec, label_counts, label_targets):
        return False
    possible_buckets = spec.label_buckets_for_state(state, config)
    if not possible_buckets:
        return False
    return best_label_deficit_for_state(spec, state, config, label_counts, label_targets)[1] <= 0


def choose_label_bucket_for_state(
    spec: TaskSpec,
    state: dict[str, Any],
    config: GenerationConfig,
    label_counts: Counter[tuple[str, str]],
    label_targets: dict[str, dict[str, int]],
    rng: random.Random,
) -> str | None:
    candidates: list[tuple[float, int, str]] = []
    for bucket in spec.label_buckets_for_state(state, config):
        target = int(label_targets.get(spec.task_id, {}).get(bucket, 0))
        if target <= 0:
            continue
        deficit = target - int(label_counts.get((spec.task_id, bucket), 0))
        if deficit <= 0:
            continue
        candidates.append((float(deficit) / float(target), deficit, bucket))
    if not candidates:
        return None
    best_score = max((ratio, deficit) for ratio, deficit, _bucket in candidates)
    best_buckets = sorted(bucket for ratio, deficit, bucket in candidates if (ratio, deficit) == best_score)
    return rng.choice(best_buckets)


def rare_preferred_task_ids(tags: dict[str, Any]) -> set[str]:
    preferred: set[str] = set()
    if tags.get("has_bomb") or tags.get("has_rocket"):
        preferred.add("E_bomb_rocket")
    if tags.get("has_straight"):
        preferred.add("F_straight")
    if tags.get("has_pair_chain"):
        preferred.add("G_pair_chain")
    if tags.get("has_plane_body"):
        preferred.add("H_plane_body")
        preferred.add("H2_plane_attachments")
    if tags.get("has_trio_solo") or tags.get("has_trio_pair"):
        preferred.add("H1_trio_attachments")
    if tags.get("has_four_two_solo") or tags.get("has_four_two_pair"):
        preferred.add("H3_four_attachments")
    if tags.get("can_pass"):
        preferred.add("I_can_pass")
    if int(tags.get("legal_action_count", 0)) <= 3:
        preferred.add("K_all_legal_actions")
    return preferred


def choose_task_specs_for_state(
    task_specs: list[TaskSpec],
    state: dict[str, Any],
    config: GenerationConfig,
    counts: Counter[str],
    targets: dict[str, int],
    label_counts: Counter[tuple[str, str]],
    label_targets: dict[str, dict[str, int]],
    rng: random.Random,
    max_tasks_per_state: int,
) -> list[TaskSpec]:
    applicable = [
        spec
        for spec in task_specs
        if int(counts.get(spec.task_id, 0)) < int(targets.get(spec.task_id, 0)) and spec.applies_to(state, config)
    ]
    if not applicable or max_tasks_per_state <= 0:
        return []

    tags = rarity_tags(state)
    preferred_ids = rare_preferred_task_ids(tags)
    selected: list[TaskSpec] = []
    remaining = applicable[:]

    def deficit_key(spec: TaskSpec) -> tuple[float, float, float, float, str]:
        target = max(1, int(targets.get(spec.task_id, 0)))
        deficit = int(targets.get(spec.task_id, 0)) - int(counts.get(spec.task_id, 0))
        label_ratio, label_deficit = best_label_deficit_for_state(spec, state, config, label_counts, label_targets)
        return (label_ratio, float(label_deficit), float(deficit) / float(target), float(deficit), spec.task_id)

    for pool_filter in (
        lambda spec: best_label_deficit_for_state(spec, state, config, label_counts, label_targets)[1] > 0,
        lambda spec: spec.task_id in preferred_ids
        and not state_labels_are_saturated(spec, state, config, label_counts, label_targets),
        lambda spec: not state_labels_are_saturated(spec, state, config, label_counts, label_targets),
        lambda spec: True,
    ):
        while len(selected) < max_tasks_per_state:
            pool = [spec for spec in remaining if pool_filter(spec)]
            if not pool:
                break
            best_score = max(deficit_key(spec)[:4] for spec in pool)
            best = [spec for spec in pool if deficit_key(spec)[:4] == best_score]
            chosen = rng.choice(sorted(best, key=lambda spec: spec.task_id))
            selected.append(chosen)
            remaining.remove(chosen)
    return selected


def choose_transition_action(legal_actions: list[str], rng: random.Random) -> str | None:
    if not legal_actions:
        return None
    non_pass = [action for action in legal_actions if action != "pass"]
    if non_pass and ("pass" not in legal_actions or rng.random() < 0.72):
        return rng.choice(sorted(non_pass, key=action_sort_key))
    return "pass" if "pass" in legal_actions else rng.choice(sorted(legal_actions, key=action_sort_key))


def advance_env_with_raw_action(env: DoudizhuSingleEnv, action: str | None) -> np.ndarray:
    if action is None or env.game is None or env.done:
        env.done = True
        return env._render()
    env.game.step(action)
    _bot_turns, bot_limit_reached = env._run_bots_until_player_turn()
    env.done = bool(env.game.is_over() or bot_limit_reached)
    if bot_limit_reached:
        env.last_message = "Bot 回合达到上限，当前 episode 已停止。" if env.language == "zh" else "Bot turn limit reached; episode stopped."
    else:
        env.last_message = ""
    return env._render()


def generated_sample_for_spec(
    spec: TaskSpec,
    state: dict[str, Any],
    rng: random.Random,
    config: GenerationConfig,
    label_bucket: str | None = None,
) -> GeneratedSample | None:
    gold = spec.build_gold_for_label(state, rng, config, label_bucket) if label_bucket else spec.build_gold(state, rng, config)
    if gold is None:
        return None
    response = spec.build_response(gold)
    verifier = verify_response(spec, response, gold)
    return GeneratedSample(gold=gold, response=response, verifier=verifier)


def make_row(
    *,
    data_source: str,
    split: str,
    sample_index: int,
    episode_seed: int,
    episode_index: int,
    step_index: int,
    obs: np.ndarray,
    image_encoder: Any,
    sample: GeneratedSample,
) -> dict[str, Any]:
    state_meta = sample.gold.metadata
    sample_id = f"{data_source}:{split}:{episode_seed}:{step_index}:{sample.gold.task_id}:{uuid.uuid4().hex[:8]}"
    legal_actions_raw = list(state_meta.get("legal_actions_raw", []))
    extra_info = {
        "sample_id": sample_id,
        "split": split,
        "sample_index": sample_index,
        "episode_seed": episode_seed,
        "episode_index": episode_index,
        "step_index": step_index,
        "task_id": sample.gold.task_id,
        "task_name": sample.gold.task_name,
        "label_bucket": state_meta.get("label_bucket"),
        "current_hand_raw": state_meta.get("current_hand_raw", ""),
        "current_hand_display": state_meta.get("current_hand_display", []),
        "legal_actions_raw": legal_actions_raw,
        "legal_actions_display": [raw_action_to_display_list(action) for action in legal_actions_raw],
        "trace_tail": state_meta.get("trace_tail", []),
        "gold": sample.gold.answer,
        "plan_aux": sample.gold.plan_aux,
        "task_metadata": {key: value for key, value in state_meta.items() if key not in {"trace_tail", "legal_actions_raw"}},
        "verifier": {
            "ok": sample.verifier.ok,
            "reason": sample.verifier.reason,
            "details": sample.verifier.details if sample.verifier.details else {"message": ""},
        },
    }
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": sample.gold.prompt}],
        "question": sample.gold.prompt,
        "images": [image_encoder(obs)],
        "answer": sample.response,
        "ability": "doudizhu_qa",
        "reward_model": {"style": "rule", "ground_truth": sample.gold.answer},
        "extra_info": extra_info,
    }


def synthesize_split(
    *,
    split: str,
    target_samples: int,
    seed_start: int,
    env_config: dict[str, Any],
    generation_config: GenerationConfig,
    image_encoder: Any,
    max_tasks_per_state: int = 3,
    max_episodes: int = 100000,
    raw_task_weights: dict[str, float] | None = None,
    log_every: int = 1000,
    data_source: str = "doudizhu_qa_sft",
) -> SplitSynthesisResult:
    task_specs = build_task_specs()
    task_by_id = {spec.task_id: spec for spec in task_specs}
    weights = normalize_weights(task_specs, raw_task_weights)
    targets = quota_targets(target_samples, weights)
    label_weights = label_quota_weights(task_specs, generation_config)
    label_targets = label_quota_targets(targets, label_weights)
    counts: Counter[str] = Counter()
    label_counts: Counter[tuple[str, str]] = Counter()
    rejection_counts: Counter[str] = Counter()
    skip_counts: Counter[str] = Counter()
    label_skip_counts: Counter[str] = Counter()
    rarity_counts: Counter[str] = Counter()
    hand_length_counts: Counter[str] = Counter()
    legal_action_count_counts: Counter[str] = Counter()
    can_pass_counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    rng = random.Random(seed_start)
    env = DoudizhuSingleEnv(seed=seed_start, env_config=env_config)
    episode_index = 0
    last_logged = 0

    while not quota_complete(counts, targets):
        if episode_index >= max_episodes:
            raise RuntimeError(
                f"{split}: exhausted max_episodes={max_episodes} before meeting task quotas; "
                f"remaining={quota_remaining(counts, targets)}"
            )
        episode_seed = seed_start + episode_index
        obs, _info = env.reset(seed=episode_seed)
        step_index = 0

        while not env.done and not quota_complete(counts, targets):
            state = dict(env.game.state)
            state_tags = rarity_tags(state)
            hand_length_counts[str(len(str(state.get("current_hand", ""))))] += 1
            legal_action_count_counts[str(len(state.get("actions", [])))] += 1
            can_pass_counts[str(bool(state_tags.get("can_pass"))).lower()] += 1
            for key, value in state_tags.items():
                if isinstance(value, bool) and value:
                    rarity_counts[key] += 1

            selected_specs = choose_task_specs_for_state(
                task_specs=task_specs,
                state=state,
                config=generation_config,
                counts=counts,
                targets=targets,
                label_counts=label_counts,
                label_targets=label_targets,
                rng=rng,
                max_tasks_per_state=max_tasks_per_state,
            )
            if not selected_specs:
                skip_counts["no_applicable_task_with_remaining_quota"] += 1

            for spec in selected_specs:
                if quota_complete(counts, targets) or int(counts.get(spec.task_id, 0)) >= int(targets.get(spec.task_id, 0)):
                    continue
                desired_label_bucket = choose_label_bucket_for_state(
                    spec,
                    state,
                    generation_config,
                    label_counts,
                    label_targets,
                    rng,
                )
                if desired_label_bucket is None and state_labels_are_saturated(
                    spec,
                    state,
                    generation_config,
                    label_counts,
                    label_targets,
                ):
                    label_skip_counts[f"{spec.task_id}:label_saturated_state"] += 1
                    continue
                sample = generated_sample_for_spec(spec, state, rng, generation_config, desired_label_bucket)
                if sample is None and desired_label_bucket:
                    label_skip_counts[f"{spec.task_id}:{desired_label_bucket}:no_gold"] += 1
                    sample = generated_sample_for_spec(spec, state, rng, generation_config)
                if sample is None:
                    skip_counts[f"{spec.task_id}:no_gold"] += 1
                    continue
                if not sample.verifier.ok:
                    rejection_counts[f"{spec.task_id}:{sample.verifier.reason}"] += 1
                    continue
                row = make_row(
                    data_source=data_source,
                    split=split,
                    sample_index=len(rows),
                    episode_seed=episode_seed,
                    episode_index=episode_index,
                    step_index=step_index,
                    obs=obs,
                    image_encoder=image_encoder,
                    sample=sample,
                )
                rows.append(row)
                counts[spec.task_id] += 1
                label_bucket = spec.label_bucket_for_gold(sample.gold)
                if label_bucket:
                    label_counts[(spec.task_id, label_bucket)] += 1
                if log_every > 0 and len(rows) > 0 and len(rows) % log_every == 0 and len(rows) != last_logged:
                    last_logged = len(rows)
                    print(
                        f"{split}: accepted {len(rows)}/{target_samples}; remaining={quota_remaining(counts, targets)}",
                        flush=True,
                    )

            transition_action = choose_transition_action(list(state.get("actions", [])), rng)
            obs = advance_env_with_raw_action(env, transition_action)
            step_index += 1

        episode_index += 1

    metadata = {
        "split": split,
        "target_samples": target_samples,
        "accepted_samples": len(rows),
        "episodes_used": episode_index,
        "seed_start": seed_start,
        "task_targets": targets,
        "task_counts": dict(counts),
        "task_names": {task_id: task_by_id[task_id].task_name for task_id in task_by_id},
        "task_weights": weights,
        "label_quotas_enabled": generation_config.label_quotas_enabled,
        "label_weights": label_weights,
        "label_targets": label_targets,
        "label_counts": nested_label_counts(label_counts, label_targets),
        "label_quota_remaining": label_quota_remaining(label_counts, label_targets),
        "quota_remaining": quota_remaining(counts, targets),
        "rejection_counts": dict(rejection_counts),
        "skip_counts": dict(skip_counts),
        "label_skip_counts": dict(label_skip_counts),
        "rarity_state_counts": dict(rarity_counts),
        "hand_length_counts": dict(hand_length_counts),
        "legal_action_count_counts": dict(legal_action_count_counts),
        "can_pass_state_counts": dict(can_pass_counts),
    }
    return SplitSynthesisResult(rows=rows, metadata=metadata)
