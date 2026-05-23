"""Canonical Dou Dizhu card/action conversions for QA synthesis."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Sequence

from agent_system.environments.env_package.doudizhu.core.judger import DoudizhuJudger
from agent_system.environments.env_package.doudizhu.core.utils import CARD_TYPE, INDEX, contains_cards

INTERNAL_RANKS = ["3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A", "2", "B", "R"]
DISPLAY_RANKS = ["3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2", "BJ", "RJ"]
INTERNAL_TO_DISPLAY = dict(zip(INTERNAL_RANKS, DISPLAY_RANKS, strict=True))
DISPLAY_TO_INTERNAL = dict(zip(DISPLAY_RANKS, INTERNAL_RANKS, strict=True))
RANK_TO_INDEX = {rank: idx for idx, rank in enumerate(INTERNAL_RANKS)}

TYPE_LABELS = {
    "solo": "单张",
    "pair": "对子",
    "trio": "三张",
    "trio_solo": "三带一",
    "trio_pair": "三带二",
    "bomb": "炸弹",
    "rocket": "王炸",
    "four_two_solo": "四带两单",
    "four_two_pair": "四带两对",
}


def rank_to_display(rank: str) -> str:
    if rank not in INTERNAL_TO_DISPLAY:
        raise ValueError(f"Unknown internal rank: {rank!r}")
    return INTERNAL_TO_DISPLAY[rank]


def rank_to_internal(rank: str) -> str:
    rank = str(rank).strip()
    if rank not in DISPLAY_TO_INTERNAL:
        raise ValueError(f"Unknown display rank: {rank!r}")
    return DISPLAY_TO_INTERNAL[rank]


def sort_internal_cards(cards: Iterable[str]) -> list[str]:
    return sorted(cards, key=lambda card: RANK_TO_INDEX[card])


def sorted_hand_raw(hand: str) -> str:
    return "".join(sort_internal_cards(hand))


def hand_to_display_list(hand: str) -> list[str]:
    return [rank_to_display(card) for card in sorted_hand_raw(hand)]


def display_cards_text(cards: Sequence[str]) -> str:
    return ", ".join(cards) if cards else "无"


def count_by_rank(hand: str) -> Counter[str]:
    return Counter(hand)


def count_map_display(hand: str) -> dict[str, int]:
    counts = count_by_rank(hand)
    return {
        rank_to_display(rank): int(counts[rank])
        for rank in INTERNAL_RANKS
        if counts[rank] > 0
    }


def display_rank_list_to_raw(ranks: Sequence[str]) -> str:
    return "".join(sort_internal_cards(rank_to_internal(rank) for rank in ranks))


def raw_action_to_display_list(action: str) -> list[str]:
    if action == "pass":
        return ["不要"]
    return [rank_to_display(card) for card in action]


def raw_action_to_display_text(action: str) -> str:
    return " ".join(raw_action_to_display_list(action))


def display_action_to_raw(action: Any) -> str:
    if not isinstance(action, list) or not action:
        raise ValueError("Action must be a non-empty list.")
    if len(action) == 1 and str(action[0]).strip() == "不要":
        return "pass"
    if any(str(item).strip() == "不要" for item in action):
        raise ValueError("'不要' cannot be mixed with card ranks.")
    return display_rank_list_to_raw([str(item).strip() for item in action])


def action_sort_key(action: str) -> tuple[Any, ...]:
    if action == "pass":
        return (999, 999, 999, "pass")
    ranks = [INDEX.get(card, 99) for card in action]
    return (len(action), max(ranks) if ranks else 99, sum(ranks), action)


def fine_type(action: str) -> str | None:
    if action == "pass":
        return None
    card_types = CARD_TYPE[0].get(action)
    if not card_types:
        return None
    return str(card_types[0][0])


def fine_type_to_label(card_type: str | None) -> str | None:
    if card_type is None:
        return None
    if card_type.startswith("trio_solo_chain"):
        return "飞机带单张"
    if card_type.startswith("trio_pair_chain"):
        return "飞机带对子"
    if card_type.startswith("solo_chain"):
        return "顺子"
    if card_type.startswith("pair_chain"):
        return "连对"
    if card_type.startswith("trio_chain"):
        return "飞机主体"
    return TYPE_LABELS.get(card_type)


def action_type_label(action: str) -> str | None:
    return fine_type_to_label(fine_type(action))


def contains_action(hand: str, action: str) -> bool:
    if action == "pass":
        return True
    return bool(contains_cards(hand, action))


def playable_actions_from_hand(hand: str) -> list[str]:
    return sorted(DoudizhuJudger.playable_cards_from_hand(hand), key=action_sort_key)


def playable_actions_by_label(hand: str, label: str) -> list[str]:
    return [action for action in playable_actions_from_hand(hand) if action_type_label(action) == label]


def longest_intervals(hand: str, min_count: int, min_length: int) -> tuple[list[list[str]], int]:
    counts = count_by_rank(hand)
    groups: list[list[str]] = []
    current: list[str] = []
    for rank in INTERNAL_RANKS[:12]:
        if counts[rank] >= min_count:
            current.append(rank)
        else:
            if current:
                groups.append(current)
                current = []
    if current:
        groups.append(current)

    longest_length = max((len(group) for group in groups), default=0)
    valid_groups = [group for group in groups if len(group) >= min_length]
    if not valid_groups:
        return [], longest_length
    valid_longest = max(len(group) for group in valid_groups)
    return [[rank_to_display(rank) for rank in group] for group in valid_groups if len(group) == valid_longest], longest_length


def interval_text(intervals: Sequence[Sequence[str]]) -> str:
    if not intervals:
        return "无"
    return "、".join("-".join(interval) for interval in intervals)


def current_trick_trace(trace: Sequence[Any]) -> list[tuple[int, str]]:
    normalized: list[tuple[int, str]] = []
    for item in trace:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            normalized.append((int(item[0]), str(item[1])))
    if not normalized:
        return []

    trick_start = 0
    for idx, (_player_id, action) in enumerate(normalized):
        if idx == 0:
            continue
        if action != "pass" and idx >= 2 and normalized[idx - 1][1] == "pass" and normalized[idx - 2][1] == "pass":
            trick_start = idx
    return normalized[trick_start:]


def target_to_respond(state: dict[str, Any]) -> tuple[int, str] | None:
    if "pass" not in set(state.get("actions", [])):
        return None
    for player_id, action in reversed(current_trick_trace(state.get("trace", []))):
        if action != "pass":
            return player_id, action
    return None


def round_context_text(state: dict[str, Any]) -> str:
    target = target_to_respond(state)
    if target is None:
        return "首发或新一轮，没有需要压过的上家牌"
    player_id, action = target
    return f"需要跟牌，上一手是玩家{player_id}出的{raw_action_to_display_text(action)}"


def compact_trace(trace: Sequence[Any]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in trace:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            compact.append({"player": int(item[0]), "action": str(item[1])})
        else:
            compact.append({"player": -1, "action": str(item)})
    return compact


def list_ranks_with_count(hand: str, min_count: int, exact: bool = False) -> list[str]:
    counts = count_by_rank(hand)
    result = []
    for rank in INTERNAL_RANKS:
        if (counts[rank] == min_count) if exact else (counts[rank] >= min_count):
            result.append(rank_to_display(rank))
    return result


def example_actions_to_display(actions: Sequence[str]) -> list[list[str]]:
    return [raw_action_to_display_list(action) for action in actions]


def first_or_none(actions: Sequence[str]) -> str | None:
    return actions[0] if actions else None


def ranks_with_at_least_from_counter(counts: Counter[str], threshold: int) -> list[str]:
    return [rank_to_display(rank) for rank in INTERNAL_RANKS if counts[rank] >= threshold]


def subtract_action_counts(hand: str, action: str) -> Counter[str]:
    counts = count_by_rank(hand)
    if action != "pass":
        for card in action:
            counts[card] -= 1
    return counts


def repeated_ranks(action: str, threshold: int) -> list[str]:
    counts = count_by_rank(action)
    return [rank for rank in INTERNAL_RANKS if counts[rank] >= threshold]


def attachment_summaries_for_body(hand: str, body_ranks: Sequence[str], body_count: int) -> tuple[list[str], list[str]]:
    remaining = count_by_rank(hand)
    for rank in body_ranks:
        remaining[rank] -= body_count
    single_cards: list[str] = []
    pair_ranks: list[str] = []
    for rank in INTERNAL_RANKS:
        if remaining[rank] > 0:
            single_cards.extend([rank_to_display(rank)] * int(remaining[rank]))
        if remaining[rank] >= 2:
            pair_ranks.append(rank_to_display(rank))
    return single_cards, pair_ranks


def plane_body_from_action(action: str) -> list[str]:
    counts = count_by_rank(action)
    ranks = [rank for rank in INTERNAL_RANKS[:12] if counts[rank] >= 3]
    if len(ranks) < 2:
        return []
    best: list[str] = []
    current: list[str] = []
    prev_idx = -10
    for rank in ranks:
        idx = RANK_TO_INDEX[rank]
        if idx == prev_idx + 1:
            current.append(rank)
        else:
            if len(current) > len(best):
                best = current
            current = [rank]
        prev_idx = idx
    if len(current) > len(best):
        best = current
    return best if len(best) >= 2 else []


def structure_description(action: str) -> str:
    label = action_type_label(action)
    display = raw_action_to_display_list(action)
    counts = count_by_rank(action)
    if label == "王炸":
        return "BJ和RJ同时打出"
    if label == "炸弹":
        rank = next(rank_to_display(rank) for rank in INTERNAL_RANKS if counts[rank] == 4)
        return f"四张{rank}"
    if label == "单张":
        return f"一张{display[0]}"
    if label == "对子":
        return f"两张{display[0]}"
    if label == "三张":
        rank = next(rank_to_display(rank) for rank in INTERNAL_RANKS if counts[rank] == 3)
        return f"三张{rank}"
    if label == "顺子":
        return f"{'-'.join(display)}连续单牌"
    if label == "连对":
        unique = [rank_to_display(rank) for rank in INTERNAL_RANKS if counts[rank] >= 2]
        return f"{'-'.join(unique)}连续对子"
    if label == "飞机主体":
        body = [rank_to_display(rank) for rank in plane_body_from_action(action)]
        return f"{'-'.join(body)}连续三张"
    if label in {"三带一", "三带二"}:
        main = next(rank for rank in INTERNAL_RANKS if counts[rank] == 3)
        others = [rank_to_display(rank) for rank in INTERNAL_RANKS if rank != main and counts[rank] > 0]
        return f"三张{rank_to_display(main)}带{'、'.join(others)}"
    if label in {"飞机带单张", "飞机带对子"}:
        body = plane_body_from_action(action)
        return f"{'-'.join(rank_to_display(rank) for rank in body)}连续三张带牌"
    if label in {"四带两单", "四带两对"}:
        main = next(rank for rank in INTERNAL_RANKS if counts[rank] == 4)
        others = [rank_to_display(rank) for rank in INTERNAL_RANKS if rank != main and counts[rank] > 0]
        return f"四张{rank_to_display(main)}带{'、'.join(others)}"
    return f"牌值为{' '.join(display)}"


def rarity_tags(state: dict[str, Any]) -> dict[str, Any]:
    hand = str(state.get("current_hand", ""))
    counts = count_by_rank(hand)
    straight, _ = longest_intervals(hand, min_count=1, min_length=5)
    pair_chain, _ = longest_intervals(hand, min_count=2, min_length=3)
    plane_body, _ = longest_intervals(hand, min_count=3, min_length=2)
    playable = playable_actions_from_hand(hand)
    labels = Counter(action_type_label(action) for action in playable)
    return {
        "has_bomb": any(counts[rank] == 4 for rank in INTERNAL_RANKS[:13]),
        "has_rocket": counts["B"] > 0 and counts["R"] > 0,
        "has_straight": bool(straight),
        "has_pair_chain": bool(pair_chain),
        "has_plane_body": bool(plane_body),
        "has_trio_solo": labels["三带一"] > 0,
        "has_trio_pair": labels["三带二"] > 0,
        "has_plane_solo": labels["飞机带单张"] > 0,
        "has_plane_pair": labels["飞机带对子"] > 0,
        "has_four_two_solo": labels["四带两单"] > 0,
        "has_four_two_pair": labels["四带两对"] > 0,
        "can_pass": "pass" in set(state.get("actions", [])),
        "legal_action_count": len(state.get("actions", [])),
    }
