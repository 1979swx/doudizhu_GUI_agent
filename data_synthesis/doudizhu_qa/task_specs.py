"""Task specifications for Dou Dizhu QA SFT synthesis."""

from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from typing import Any

from .canonical import (
    DISPLAY_RANKS,
    INTERNAL_RANKS,
    action_sort_key,
    action_type_label,
    attachment_summaries_for_body,
    compact_trace,
    contains_action,
    count_by_rank,
    count_map_display,
    display_action_to_raw,
    display_cards_text,
    example_actions_to_display,
    fine_type,
    hand_to_display_list,
    interval_text,
    list_ranks_with_count,
    longest_intervals,
    plane_body_from_action,
    playable_actions_by_label,
    playable_actions_from_hand,
    rank_to_display,
    rank_to_internal,
    raw_action_to_display_list,
    raw_action_to_display_text,
    repeated_ranks,
    round_context_text,
    structure_description,
    target_to_respond,
)
from .schemas import GenerationConfig, TaskGold, VerificationResult

RANK_SPEC_TEXT = "牌值使用 3,4,5,6,7,8,9,10,J,Q,K,A,2,BJ,RJ。"
COUNT_WORD = {1: "一", 2: "两", 3: "三", 4: "四", 5: "五"}


def json_compact(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def answer_only_prompt(question: str, schema: str) -> str:
    return (
        "当前游戏屏幕截图：<image>\n\n"
        f"问题：{question}\n\n"
        "输出要求：\n"
        "只输出一个 <answer> 标签。\n"
        f"<answer> 内必须是 JSON，格式为 {schema}。\n"
        f"{RANK_SPEC_TEXT}\n"
        "不要输出额外解释。"
    )


def plan_answer_prompt(question: str, schema: str) -> str:
    return (
        "当前游戏屏幕截图：<image>\n\n"
        f"问题：{question}\n\n"
        "输出要求：\n"
        "先输出 <plan>，用 3 到 5 个短句说明规则判断依据，plan 必须以“当前手牌有...”开头。\n"
        f"然后输出 <answer>，<answer> 内必须是 JSON，格式为 {schema}。\n"
        f"{RANK_SPEC_TEXT}\n"
        "不要输出其它内容。"
    )


def base_plan_aux(state: dict[str, Any]) -> dict[str, Any]:
    hand = str(state.get("current_hand", ""))
    hand_display = hand_to_display_list(hand)
    return {
        "当前手牌": display_cards_text(hand_display),
        "当前手牌列表": hand_display,
        "牌值数量": count_map_display(hand),
        "当前轮情况": round_context_text(state),
        "目标动作": "无" if target_to_respond(state) is None else raw_action_to_display_text(target_to_respond(state)[1]),
    }


def make_gold(
    spec: "TaskSpec",
    state: dict[str, Any],
    prompt: str,
    answer: dict[str, Any],
    plan_aux: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> TaskGold:
    hand = str(state.get("current_hand", ""))
    merged_aux = base_plan_aux(state)
    if plan_aux:
        merged_aux.update(plan_aux)
    merged_metadata = {
        "trace_tail": compact_trace(list(state.get("trace", []))[-6:]),
        "legal_actions_raw": list(state.get("actions", [])),
    }
    if metadata:
        merged_metadata.update(metadata)
    return TaskGold(
        task_id=spec.task_id,
        task_name=spec.task_name,
        prompt=prompt,
        answer=answer,
        plan_aux=merged_aux,
        metadata={
            "current_hand_raw": hand,
            "current_hand_display": hand_to_display_list(hand),
            **merged_metadata,
        },
        requires_plan=spec.requires_plan,
    )


def exact_answer(parsed: dict[str, Any], gold: TaskGold) -> VerificationResult:
    if parsed == gold.answer:
        return VerificationResult(True)
    return VerificationResult(False, "answer_mismatch", {"expected": gold.answer, "actual": parsed})


def validate_rank_list(value: Any, *, allow_duplicates: bool = False) -> tuple[list[str] | None, VerificationResult]:
    if not isinstance(value, list):
        return None, VerificationResult(False, "schema_error", {"message": "expected list"})
    normalized: list[str] = []
    seen: set[str] = set()
    last_index = -1
    for item in value:
        if not isinstance(item, str):
            return None, VerificationResult(False, "rank_format_error", {"rank": item})
        if item not in DISPLAY_RANKS:
            return None, VerificationResult(False, "rank_format_error", {"rank": item})
        index = DISPLAY_RANKS.index(item)
        if index < last_index:
            return None, VerificationResult(False, "order_error", {"ranks": value})
        if not allow_duplicates and item in seen:
            return None, VerificationResult(False, "schema_error", {"message": "duplicate rank", "rank": item})
        normalized.append(item)
        seen.add(item)
        last_index = index
    return normalized, VerificationResult(True)


def validate_action_list(value: Any) -> tuple[str | None, VerificationResult]:
    try:
        raw = display_action_to_raw(value)
    except ValueError as exc:
        return None, VerificationResult(False, "rank_format_error", {"message": str(exc), "action": value})
    return raw, VerificationResult(True)


class TaskSpec(ABC):
    task_id: str
    task_name: str
    weight: float
    requires_plan = True

    def applies_to(self, state: dict[str, Any], config: GenerationConfig) -> bool:
        return bool(state.get("current_hand")) and bool(state.get("actions") is not None)

    @abstractmethod
    def build_gold(self, state: dict[str, Any], rng: random.Random, config: GenerationConfig) -> TaskGold | None:
        raise NotImplementedError

    def label_bucket_weights(self, config: GenerationConfig) -> dict[str, float]:
        return {}

    def label_bucket_for_state(self, state: dict[str, Any], config: GenerationConfig) -> str | None:
        return None

    def label_buckets_for_state(self, state: dict[str, Any], config: GenerationConfig) -> list[str]:
        bucket = self.label_bucket_for_state(state, config)
        return [bucket] if bucket else []

    def label_bucket_for_gold(self, gold: TaskGold) -> str | None:
        bucket = gold.metadata.get("label_bucket")
        return str(bucket) if bucket else None

    def build_gold_for_label(
        self,
        state: dict[str, Any],
        rng: random.Random,
        config: GenerationConfig,
        label_bucket: str,
    ) -> TaskGold | None:
        if label_bucket not in set(self.label_buckets_for_state(state, config)):
            return None
        gold = self.build_gold(state, rng, config)
        if gold is None or self.label_bucket_for_gold(gold) != label_bucket:
            return None
        return gold

    def build_plan(self, gold: TaskGold) -> str:
        return f"当前手牌有{gold.plan_aux['当前手牌']}。根据题目要求进行规则判断。因此答案如 <answer> 所示。"

    def build_response(self, gold: TaskGold) -> str:
        answer_json = json_compact(gold.answer)
        if not gold.requires_plan:
            return f"<answer>{answer_json}</answer>"
        return f"<plan>{self.build_plan(gold)}</plan>\n<answer>{answer_json}</answer>"

    @abstractmethod
    def verify_answer(self, parsed: dict[str, Any], gold: TaskGold) -> VerificationResult:
        raise NotImplementedError


class CurrentHandTask(TaskSpec):
    task_id = "A_current_hand"
    task_name = "列出当前手牌"
    weight = 0.12
    requires_plan = False

    def build_gold(self, state: dict[str, Any], rng: random.Random, config: GenerationConfig) -> TaskGold:
        hand = str(state.get("current_hand", ""))
        answer = {"手牌": hand_to_display_list(hand)}
        prompt = answer_only_prompt("当前玩家的手牌都有哪些？请按牌值从小到大列出全部牌。", '{"手牌":[...]}')
        return make_gold(self, state, prompt, answer)

    def verify_answer(self, parsed: dict[str, Any], gold: TaskGold) -> VerificationResult:
        if set(parsed) != {"手牌"}:
            return VerificationResult(False, "schema_error", {"keys": list(parsed)})
        ranks, result = validate_rank_list(parsed.get("手牌"), allow_duplicates=True)
        if not result.ok:
            return result
        return exact_answer({"手牌": ranks}, gold)


class RankCountTask(TaskSpec):
    task_id = "B_rank_counts"
    task_name = "统计当前手牌各牌值数量"
    weight = 0.10
    requires_plan = False

    def build_gold(self, state: dict[str, Any], rng: random.Random, config: GenerationConfig) -> TaskGold:
        answer = {"牌值数量": count_map_display(str(state.get("current_hand", "")))}
        prompt = answer_only_prompt("当前玩家手牌中每种牌值分别有几张？只列出数量大于 0 的牌值。", '{"牌值数量":{"3":1}}')
        return make_gold(self, state, prompt, answer)

    def verify_answer(self, parsed: dict[str, Any], gold: TaskGold) -> VerificationResult:
        counts = parsed.get("牌值数量")
        if set(parsed) != {"牌值数量"} or not isinstance(counts, dict):
            return VerificationResult(False, "schema_error", {"keys": list(parsed)})
        normalized: dict[str, int] = {}
        last_index = -1
        for key, value in counts.items():
            if key not in DISPLAY_RANKS:
                return VerificationResult(False, "rank_format_error", {"rank": key})
            if DISPLAY_RANKS.index(key) < last_index:
                return VerificationResult(False, "order_error", {"ranks": list(counts)})
            if not isinstance(value, int) or value <= 0:
                return VerificationResult(False, "schema_error", {"rank": key, "count": value})
            normalized[key] = int(value)
            last_index = DISPLAY_RANKS.index(key)
        return exact_answer({"牌值数量": normalized}, gold)


class SpecificRankCountTask(TaskSpec):
    task_id = "C_specific_rank_count"
    task_name = "判断指定牌值数量"
    weight = 0.08
    requires_plan = False

    def build_gold(self, state: dict[str, Any], rng: random.Random, config: GenerationConfig) -> TaskGold:
        hand = str(state.get("current_hand", ""))
        present = sorted({rank_to_display(card) for card in hand}, key=DISPLAY_RANKS.index)
        if present and rng.random() < 0.6:
            rank = rng.choice(present)
        else:
            rank = rng.choice(DISPLAY_RANKS)
        count = count_by_rank(hand)[rank_to_internal(rank)]
        answer = {"牌值": rank, "是否存在": count > 0, "数量": int(count)}
        prompt = answer_only_prompt(f"当前玩家手牌中有没有 {rank}？如果有，有几张？", '{"牌值":"2","是否存在":true,"数量":3}')
        return make_gold(self, state, prompt, answer, metadata={"queried_rank": rank})

    def verify_answer(self, parsed: dict[str, Any], gold: TaskGold) -> VerificationResult:
        if set(parsed) != {"牌值", "是否存在", "数量"}:
            return VerificationResult(False, "schema_error", {"keys": list(parsed)})
        if parsed.get("牌值") not in DISPLAY_RANKS:
            return VerificationResult(False, "rank_format_error", {"rank": parsed.get("牌值")})
        if not isinstance(parsed.get("是否存在"), bool) or not isinstance(parsed.get("数量"), int):
            return VerificationResult(False, "schema_error")
        return exact_answer(parsed, gold)


class MultiplicityTask(TaskSpec):
    def __init__(self, task_id: str, task_name: str, answer_key: str, threshold: int, weight: float, exact: bool = False):
        self.task_id = task_id
        self.task_name = task_name
        self.answer_key = answer_key
        self.threshold = threshold
        self.weight = weight
        self.exact = exact

    def build_gold(self, state: dict[str, Any], rng: random.Random, config: GenerationConfig) -> TaskGold:
        ranks = list_ranks_with_count(str(state.get("current_hand", "")), self.threshold, exact=self.exact)
        answer = {self.answer_key: ranks}
        question = f"当前玩家手牌中有哪些{self.answer_key}？请列出所有能构成{self.answer_key}的牌值。"
        if self.answer_key == "四张":
            question = "当前玩家手牌中有哪些四张同牌？请列出所有四张牌值。"
        prompt = plan_answer_prompt(question, f'{{"{self.answer_key}":[...]}}')
        return make_gold(
            self,
            state,
            prompt,
            answer,
            plan_aux={"目标牌型": self.answer_key, "需要张数": self.threshold, "满足条件牌值": ranks},
        )

    def build_plan(self, gold: TaskGold) -> str:
        ranks = gold.answer[self.answer_key]
        need = COUNT_WORD.get(self.threshold, str(self.threshold))
        if ranks:
            return (
                f"当前手牌有{gold.plan_aux['当前手牌']}。判断{self.answer_key}先看每个牌值出现了几张："
                f"同一牌值达到{need}张就满足。这里{'和'.join(ranks)}达到了要求。因此答案中列出这些牌值。"
            )
        return (
            f"当前手牌有{gold.plan_aux['当前手牌']}。{self.answer_key}要求同一牌值至少{need}张。"
            f"逐项计数后，没有牌值达到这个数量。因此当前不能构成{self.answer_key}。"
        )

    def verify_answer(self, parsed: dict[str, Any], gold: TaskGold) -> VerificationResult:
        if set(parsed) != {self.answer_key}:
            return VerificationResult(False, "schema_error", {"keys": list(parsed)})
        ranks, result = validate_rank_list(parsed.get(self.answer_key), allow_duplicates=False)
        if not result.ok:
            return result
        return exact_answer({self.answer_key: ranks}, gold)


class BombRocketTask(TaskSpec):
    task_id = "E_bomb_rocket"
    task_name = "判断炸弹和王炸"
    weight = 0.06

    def build_gold(self, state: dict[str, Any], rng: random.Random, config: GenerationConfig) -> TaskGold:
        hand = str(state.get("current_hand", ""))
        counts = count_by_rank(hand)
        bombs = [rank_to_display(rank) for rank in INTERNAL_RANKS[:13] if counts[rank] == 4]
        rocket = counts["B"] > 0 and counts["R"] > 0
        answer = {"炸弹": bombs, "是否有王炸": rocket}
        bomb_observation = "、".join(f"{rank}有四张，可以作为普通炸弹" for rank in bombs) if bombs else "没有牌值达到四张"
        if rocket:
            rocket_observation = "BJ和RJ同时在手，可以组成王炸"
            missing = ""
        else:
            missing_parts = []
            if counts["B"] <= 0:
                missing_parts.append("BJ")
            if counts["R"] <= 0:
                missing_parts.append("RJ")
            missing = "缺少" + "和".join(missing_parts)
            rocket_observation = missing
        prompt = plan_answer_prompt("当前手牌中有没有炸弹或王炸？分别是什么？", '{"炸弹":["9"],"是否有王炸":true}')
        return make_gold(
            self,
            state,
            prompt,
            answer,
            plan_aux={"炸弹观察": bomb_observation, "王炸观察": rocket_observation, "王炸缺失原因": missing},
        )

    def build_plan(self, gold: TaskGold) -> str:
        if gold.answer["炸弹"] or gold.answer["是否有王炸"]:
            return (
                f"当前手牌有{gold.plan_aux['当前手牌']}。炸弹看是否有四张同牌，王炸看BJ和RJ是否同时在手。"
                f"当前{gold.plan_aux['炸弹观察']}；{gold.plan_aux['王炸观察']}。因此炸弹和王炸的判断如答案所示。"
            )
        return (
            f"当前手牌有{gold.plan_aux['当前手牌']}。普通炸弹需要四张同牌，王炸需要同时有BJ和RJ。"
            f"当前没有牌值达到四张，且{gold.plan_aux['王炸缺失原因']}。所以没有对应的炸弹或王炸。"
        )

    def verify_answer(self, parsed: dict[str, Any], gold: TaskGold) -> VerificationResult:
        if set(parsed) != {"炸弹", "是否有王炸"} or not isinstance(parsed.get("是否有王炸"), bool):
            return VerificationResult(False, "schema_error", {"keys": list(parsed)})
        bombs, result = validate_rank_list(parsed.get("炸弹"), allow_duplicates=False)
        if not result.ok:
            return result
        if "BJ" in bombs or "RJ" in bombs:
            return VerificationResult(False, "answer_mismatch", {"message": "rocket cannot be listed as normal bomb"})
        return exact_answer({"炸弹": bombs, "是否有王炸": parsed["是否有王炸"]}, gold)


class LongestIntervalTask(TaskSpec):
    def __init__(
        self,
        task_id: str,
        task_name: str,
        answer_bool_key: str,
        answer_list_key: str,
        target_name: str,
        min_count: int,
        min_length: int,
        weight: float,
    ):
        self.task_id = task_id
        self.task_name = task_name
        self.answer_bool_key = answer_bool_key
        self.answer_list_key = answer_list_key
        self.target_name = target_name
        self.min_count = min_count
        self.min_length = min_length
        self.weight = weight

    def label_bucket_weights(self, config: GenerationConfig) -> dict[str, float]:
        if self.task_id != "F_straight":
            return {}
        return {"has_straight:true": 0.5, "has_straight:false": 0.5}

    def label_bucket_for_state(self, state: dict[str, Any], config: GenerationConfig) -> str | None:
        if self.task_id != "F_straight":
            return None
        intervals, _longest_length = longest_intervals(str(state.get("current_hand", "")), self.min_count, self.min_length)
        return f"has_straight:{str(bool(intervals)).lower()}"

    def build_gold(self, state: dict[str, Any], rng: random.Random, config: GenerationConfig) -> TaskGold:
        intervals, longest_length = longest_intervals(str(state.get("current_hand", "")), self.min_count, self.min_length)
        answer = {self.answer_bool_key: bool(intervals), self.answer_list_key: intervals}
        prompt = plan_answer_prompt(
            f"当前手牌中是否存在{self.target_name}？请列出所有最长{self.target_name}区间。",
            json_compact({self.answer_bool_key: True, self.answer_list_key: [["4", "5", "6"]]}),
        )
        return make_gold(
            self,
            state,
            prompt,
            answer,
            plan_aux={"最长区间": intervals, "最长区间文本": interval_text(intervals), "最长长度": longest_length, "目标牌型": self.target_name},
            metadata={"label_bucket": self.label_bucket_for_state(state, config)} if self.task_id == "F_straight" else None,
        )

    def build_plan(self, gold: TaskGold) -> str:
        hand = gold.plan_aux["当前手牌"]
        exists = gold.answer[self.answer_bool_key]
        interval = gold.plan_aux["最长区间文本"]
        longest_length = gold.plan_aux["最长长度"]
        if self.target_name == "顺子":
            if exists:
                return (
                    f"当前手牌有{hand}。顺子只看3到A，要求至少5个连续牌值，每个牌值至少一张；2、BJ、RJ不能接进顺子。"
                    f"当前能连起来的最长区间是{interval}。因此存在答案中的最长顺子。"
                )
            return (
                f"当前手牌有{hand}。顺子只看3到A，至少要连续5个牌值；2、BJ、RJ不能参与。"
                f"当前3到A范围内最长连续区间只有{longest_length}个牌值，不足5个。因此当前没有顺子。"
            )
        if self.target_name == "连对":
            if exists:
                return (
                    f"当前手牌有{hand}。连对要找连续的对子，至少3个连续牌值，每个牌值至少两张；2、BJ、RJ不能参与。"
                    f"当前满足对子数量并且连续的区间是{interval}。因此存在答案中的最长连对。"
                )
            return (
                f"当前手牌有{hand}。连对要求至少3个连续牌值，每个牌值至少两张。"
                f"当前满足对子数量的连续区间最长只有{longest_length}个牌值，不足3个。因此当前没有连对。"
            )
        if exists:
            return (
                f"当前手牌有{hand}。飞机主体先找连续的三张，至少2个连续牌值，每个牌值至少三张；2、BJ、RJ不参与连续主体。"
                f"当前满足三张数量并且连续的区间是{interval}。因此存在答案中的最长飞机主体。"
            )
        return (
            f"当前手牌有{hand}。飞机主体要求至少2个连续牌值，每个牌值至少三张。"
            "当前满足三张数量的牌值不能形成长度至少2的连续区间。因此当前没有飞机主体。"
        )

    def verify_answer(self, parsed: dict[str, Any], gold: TaskGold) -> VerificationResult:
        if set(parsed) != {self.answer_bool_key, self.answer_list_key} or not isinstance(parsed.get(self.answer_bool_key), bool):
            return VerificationResult(False, "schema_error", {"keys": list(parsed)})
        intervals = parsed.get(self.answer_list_key)
        if not isinstance(intervals, list):
            return VerificationResult(False, "schema_error", {"message": "intervals must be a list"})
        normalized: list[list[str]] = []
        for interval in intervals:
            ranks, result = validate_rank_list(interval, allow_duplicates=False)
            if not result.ok:
                return result
            normalized.append(ranks)
        return exact_answer({self.answer_bool_key: parsed[self.answer_bool_key], self.answer_list_key: normalized}, gold)


class ExampleByTypeTask(TaskSpec):
    labels: tuple[str, str]
    bool_keys: tuple[str, str]
    example_keys: tuple[str, str]

    def _choose_example(self, hand: str, label: str, rng: random.Random) -> str | None:
        candidates = playable_actions_by_label(hand, label)
        if not candidates:
            return None
        return rng.choice(candidates)

    def _verify_example_group(
        self,
        parsed: dict[str, Any],
        gold: TaskGold,
        bool_key: str,
        example_key: str,
        label: str,
    ) -> VerificationResult:
        expected_exists = bool(gold.answer[bool_key])
        actual_exists = parsed.get(bool_key)
        examples = parsed.get(example_key)
        if not isinstance(actual_exists, bool) or not isinstance(examples, list):
            return VerificationResult(False, "schema_error", {"bool_key": bool_key, "example_key": example_key})
        if actual_exists != expected_exists:
            return VerificationResult(False, "answer_mismatch", {"key": bool_key, "expected": expected_exists, "actual": actual_exists})
        if not expected_exists:
            if examples:
                return VerificationResult(False, "answer_mismatch", {"key": example_key, "message": "false task must use empty examples"})
            return VerificationResult(True)
        if not examples:
            return VerificationResult(False, "schema_error", {"key": example_key, "message": "true task needs at least one example"})

        hand = str(gold.metadata["current_hand_raw"])
        for example in examples:
            raw, result = validate_action_list(example)
            if not result.ok:
                return result
            if not contains_action(hand, raw):
                return VerificationResult(False, "illegal_action", {"action": raw, "hand": hand})
            if action_type_label(raw) != label:
                return VerificationResult(False, "answer_mismatch", {"action": raw, "expected_type": label, "actual_type": action_type_label(raw)})
        return VerificationResult(True)

    def verify_answer(self, parsed: dict[str, Any], gold: TaskGold) -> VerificationResult:
        expected_keys = set(self.bool_keys + self.example_keys)
        if set(parsed) != expected_keys:
            return VerificationResult(False, "schema_error", {"keys": list(parsed), "expected_keys": sorted(expected_keys)})
        for bool_key, example_key, label in zip(self.bool_keys, self.example_keys, self.labels, strict=True):
            result = self._verify_example_group(parsed, gold, bool_key, example_key, label)
            if not result.ok:
                return result
        return VerificationResult(True)


class TrioAttachTask(ExampleByTypeTask):
    task_id = "H1_trio_attachments"
    task_name = "识别三带一和三带二"
    weight = 0.07
    labels = ("三带一", "三带二")
    bool_keys = ("是否有三带一", "是否有三带二")
    example_keys = ("三带一示例", "三带二示例")

    def build_gold(self, state: dict[str, Any], rng: random.Random, config: GenerationConfig) -> TaskGold:
        hand = str(state.get("current_hand", ""))
        solo = self._choose_example(hand, "三带一", rng)
        pair = self._choose_example(hand, "三带二", rng)
        answer = {
            "是否有三带一": solo is not None,
            "三带一示例": example_actions_to_display([solo] if solo else []),
            "是否有三带二": pair is not None,
            "三带二示例": example_actions_to_display([pair] if pair else []),
        }
        counts = count_by_rank(hand)
        trio_ranks = [rank for rank in INTERNAL_RANKS if counts[rank] >= 3]
        main = repeated_ranks(solo or pair or "", 3)[:1] or trio_ranks[:1]
        singles, pairs = attachment_summaries_for_body(hand, main, 3) if main else ([], [])
        failure = "没有三张主体"
        if trio_ranks:
            if not solo:
                failure = "主体之外没有足够单牌"
            elif not pair:
                failure = "主体之外没有足够对子"
        prompt = plan_answer_prompt("当前手牌中是否能组成三带一或三带二？如果可以，请分别列举一个合法示例。", '{"是否有三带一":true,"三带一示例":[["3","7","7","7"]],"是否有三带二":true,"三带二示例":[["4","4","7","7","7"]]}')
        return make_gold(
            self,
            state,
            prompt,
            answer,
            plan_aux={
                "三张主体": [rank_to_display(rank) for rank in trio_ranks],
                "可用单牌": singles,
                "可用对子": pairs,
                "失败原因": failure,
            },
        )

    def build_plan(self, gold: TaskGold) -> str:
        hand = gold.plan_aux["当前手牌"]
        trio = display_cards_text(gold.plan_aux["三张主体"])
        singles = display_cards_text(gold.plan_aux["可用单牌"])
        pairs = display_cards_text(gold.plan_aux["可用对子"])
        solo_ok = gold.answer["是否有三带一"]
        pair_ok = gold.answer["是否有三带二"]
        if solo_ok and pair_ok:
            return (
                f"当前手牌有{hand}。三带类牌型先找三张主体，再看主体之外能带什么牌：三带一带1张单牌，三带二带1个对子。"
                f"当前三张主体有{trio}，主体外可用单牌有{singles}，可用对子有{pairs}。因此可以组成答案中的示例。"
            )
        if solo_ok or pair_ok:
            exists = "三带一" if solo_ok else "三带二"
            missing = "三带二" if solo_ok else "三带一"
            return (
                f"当前手牌有{hand}。三带一和三带二都必须先有三张主体。"
                f"当前三张主体有{trio}；{exists}的带牌条件满足，但{missing}缺少{gold.plan_aux['失败原因']}。"
                "因此只存在答案中标为true的牌型。"
            )
        return (
            f"当前手牌有{hand}。三带一和三带二都需要三张主体，并且带牌不能从主体里重复拿。"
            f"当前{gold.plan_aux['失败原因']}。因此不能组成三带一或三带二。"
        )


class PlaneAttachTask(ExampleByTypeTask):
    task_id = "H2_plane_attachments"
    task_name = "识别飞机带单张和飞机带对子"
    weight = 0.05
    labels = ("飞机带单张", "飞机带对子")
    bool_keys = ("是否有飞机带单张", "是否有飞机带对子")
    example_keys = ("飞机带单张示例", "飞机带对子示例")

    def label_bucket_weights(self, config: GenerationConfig) -> dict[str, float]:
        return {"has_plane_attachment:true": 0.5, "has_plane_attachment:false": 0.5}

    def _available_attachment_labels(self, hand: str) -> tuple[bool, bool]:
        labels = {action_type_label(action) for action in playable_actions_from_hand(hand)}
        return "飞机带单张" in labels, "飞机带对子" in labels

    def label_bucket_for_state(self, state: dict[str, Any], config: GenerationConfig) -> str:
        solo_ok, pair_ok = self._available_attachment_labels(str(state.get("current_hand", "")))
        return f"has_plane_attachment:{str(solo_ok or pair_ok).lower()}"

    def build_gold(self, state: dict[str, Any], rng: random.Random, config: GenerationConfig) -> TaskGold:
        hand = str(state.get("current_hand", ""))
        solo = self._choose_example(hand, "飞机带单张", rng)
        pair = self._choose_example(hand, "飞机带对子", rng)
        if solo and pair:
            detail_bucket = "both"
        elif solo:
            detail_bucket = "solo_only"
        elif pair:
            detail_bucket = "pair_only"
        else:
            detail_bucket = "none"
        answer = {
            "是否有飞机带单张": solo is not None,
            "飞机带单张示例": example_actions_to_display([solo] if solo else []),
            "是否有飞机带对子": pair is not None,
            "飞机带对子示例": example_actions_to_display([pair] if pair else []),
        }
        body = plane_body_from_action(solo or pair or "")
        if not body:
            body_intervals, _ = longest_intervals(hand, min_count=3, min_length=2)
            body = [rank_to_internal(rank) for rank in body_intervals[0]] if body_intervals else []
        singles, pairs = attachment_summaries_for_body(hand, body, 3) if body else ([], [])
        if not body:
            failure = "没有长度至少2的连续三张主体"
        elif not solo and not pair:
            failure = "足够单牌和足够对子"
        elif not solo:
            failure = "主体之外没有足够单牌"
        elif not pair:
            failure = "主体之外没有足够对子"
        else:
            failure = ""
        prompt = plan_answer_prompt("当前手牌中是否能组成飞机带单张或飞机带对子？如果可以，请分别列举一个合法示例。", '{"是否有飞机带单张":true,"飞机带单张示例":[["3","6","6","6","7","7","7","9"]],"是否有飞机带对子":false,"飞机带对子示例":[]}')
        return make_gold(
            self,
            state,
            prompt,
            answer,
            plan_aux={
                "飞机主体": [rank_to_display(rank) for rank in body],
                "单牌数量": len(singles),
                "对子数量": len(pairs),
                "失败原因": failure,
            },
            metadata={
                "label_bucket": f"has_plane_attachment:{str(bool(solo or pair)).lower()}",
                "plane_attachment_detail": detail_bucket,
            },
        )

    def build_plan(self, gold: TaskGold) -> str:
        hand = gold.plan_aux["当前手牌"]
        body = display_cards_text(gold.plan_aux["飞机主体"])
        solo_ok = gold.answer["是否有飞机带单张"]
        pair_ok = gold.answer["是否有飞机带对子"]
        if not gold.plan_aux["飞机主体"]:
            return (
                f"当前手牌有{hand}。飞机带单张和飞机带对子都必须先有飞机主体，也就是至少两个连续的三张。"
                "当前没有长度至少2的连续三张主体。因此无法组成飞机带牌。"
            )
        if solo_ok and pair_ok:
            return (
                f"当前手牌有{hand}。飞机带牌先看有没有连续三张主体；主体长度为m时，带单张要有m张主体之外的单牌，带对子要有m个主体之外的对子。"
                f"当前飞机主体是{body}，可用单牌数量是{gold.plan_aux['单牌数量']}，可用对子数量是{gold.plan_aux['对子数量']}。因此对应牌型的存在性和示例如答案所示。"
            )
        return (
            f"当前手牌有{hand}。当前有飞机主体{body}，但带牌数量还要和主体长度对应：带单张需要同数量单牌，带对子需要同数量对子。"
            f"当前缺少{gold.plan_aux['失败原因']}。因此只有满足带牌数量的类别可以成立。"
        )


class FourAttachTask(ExampleByTypeTask):
    task_id = "H3_four_attachments"
    task_name = "识别四带两单和四带两对"
    weight = 0.03
    labels = ("四带两单", "四带两对")
    bool_keys = ("是否有四带两单", "是否有四带两对")
    example_keys = ("四带两单示例", "四带两对示例")

    def build_gold(self, state: dict[str, Any], rng: random.Random, config: GenerationConfig) -> TaskGold:
        hand = str(state.get("current_hand", ""))
        solo = self._choose_example(hand, "四带两单", rng)
        pair = self._choose_example(hand, "四带两对", rng)
        answer = {
            "是否有四带两单": solo is not None,
            "四带两单示例": example_actions_to_display([solo] if solo else []),
            "是否有四带两对": pair is not None,
            "四带两对示例": example_actions_to_display([pair] if pair else []),
        }
        counts = count_by_rank(hand)
        four_ranks = [rank for rank in INTERNAL_RANKS if counts[rank] >= 4]
        body = repeated_ranks(solo or pair or "", 4)[:1] or four_ranks[:1]
        singles, pairs = attachment_summaries_for_body(hand, body, 4) if body else ([], [])
        if not body:
            failure = "没有四张主体"
        elif not solo and not pair:
            failure = "主体之外没有足够单牌和足够对子"
        elif not solo:
            failure = "主体之外没有足够单牌"
        elif not pair:
            failure = "主体之外没有足够对子"
        else:
            failure = ""
        prompt = plan_answer_prompt("当前手牌中是否能组成四带两单或四带两对？如果可以，请分别列举一个合法示例。", '{"是否有四带两单":true,"四带两单示例":[["3","7","9","9","9","9"]],"是否有四带两对":true,"四带两对示例":[["4","4","9","9","9","9","J","J"]]}')
        return make_gold(
            self,
            state,
            prompt,
            answer,
            plan_aux={
                "四张主体": [rank_to_display(rank) for rank in four_ranks],
                "可用单牌": singles,
                "可用对子": pairs,
                "失败原因": failure,
            },
        )

    def build_plan(self, gold: TaskGold) -> str:
        hand = gold.plan_aux["当前手牌"]
        body = display_cards_text(gold.plan_aux["四张主体"])
        singles = display_cards_text(gold.plan_aux["可用单牌"])
        pairs = display_cards_text(gold.plan_aux["可用对子"])
        if gold.answer["是否有四带两单"] or gold.answer["是否有四带两对"]:
            return (
                f"当前手牌有{hand}。四带二先找四张主体，再看主体之外的两组带牌：四带两单带2张单牌，四带两对带2个对子。"
                f"当前四张主体是{body}，主体外可用单牌有{singles}，可用对子有{pairs}。因此可以组成答案中的示例。"
            )
        return (
            f"当前手牌有{hand}。四带两单和四带两对都必须先有四张主体，并且带牌要从主体之外选择。"
            f"当前{gold.plan_aux['失败原因']}。因此不能组成对应的四带二牌型。"
        )


class CanPassTask(TaskSpec):
    task_id = "I_can_pass"
    task_name = "当前能否过牌"
    weight = 0.04

    def label_bucket_weights(self, config: GenerationConfig) -> dict[str, float]:
        return {"can_pass:true": 0.5, "can_pass:false": 0.5}

    def label_bucket_for_state(self, state: dict[str, Any], config: GenerationConfig) -> str:
        return f"can_pass:{str('pass' in set(state.get('actions', []))).lower()}"

    def build_gold(self, state: dict[str, Any], rng: random.Random, config: GenerationConfig) -> TaskGold:
        can_pass = "pass" in set(state.get("actions", []))
        answer = {"是否可以不要": can_pass}
        prompt = plan_answer_prompt("当前局面下，玩家是否可以选择不要？", '{"是否可以不要":true}')
        return make_gold(self, state, prompt, answer, metadata={"label_bucket": self.label_bucket_for_state(state, config)})

    def build_plan(self, gold: TaskGold) -> str:
        hand = gold.plan_aux["当前手牌"]
        context = gold.plan_aux["当前轮情况"]
        target = gold.plan_aux["目标动作"]
        if gold.answer["是否可以不要"]:
            return (
                f"当前手牌有{hand}。斗地主里只有在需要跟别人出的牌时，才可以选择不要；如果轮到自己首发，就必须出牌。"
                f"当前出牌区显示{context}，需要回应的是{target}。因此这个局面可以选择不要。"
            )
        return (
            f"当前手牌有{hand}。斗地主里首发时不能不要，必须打出一个合法牌型；只有跟牌时才允许过牌。"
            f"当前出牌区显示{context}，没有需要回应的上家牌。因此这个局面不能选择不要。"
        )

    def verify_answer(self, parsed: dict[str, Any], gold: TaskGold) -> VerificationResult:
        if set(parsed) != {"是否可以不要"} or not isinstance(parsed.get("是否可以不要"), bool):
            return VerificationResult(False, "schema_error", {"keys": list(parsed)})
        return exact_answer(parsed, gold)


class AllLegalActionsTask(TaskSpec):
    task_id = "K_all_legal_actions"
    task_name = "列出全部合法动作"
    weight = 0.06

    def label_bucket_weights(self, config: GenerationConfig) -> dict[str, float]:
        return {f"legal_action_count:{count}": 1.0 for count in range(1, config.n_all + 1)}

    def label_bucket_for_state(self, state: dict[str, Any], config: GenerationConfig) -> str:
        return f"legal_action_count:{len(state.get('actions', []))}"

    def applies_to(self, state: dict[str, Any], config: GenerationConfig) -> bool:
        return super().applies_to(state, config) and 0 < len(state.get("actions", [])) <= config.n_all

    def build_gold(self, state: dict[str, Any], rng: random.Random, config: GenerationConfig) -> TaskGold:
        actions = list(state.get("actions", []))
        rng.shuffle(actions)
        answer = {"合法动作": [raw_action_to_display_list(action) for action in actions]}
        prompt = plan_answer_prompt("当前牌面下合法出牌动作较少。请列出全部合法动作。", '{"合法动作":[["不要"],["9"],["10"]]}')
        legal_action_count = len(state.get("actions", []))
        return make_gold(
            self,
            state,
            prompt,
            answer,
            plan_aux={"N_all": config.n_all},
            metadata={
                "n_all": config.n_all,
                "legal_action_count": legal_action_count,
                "label_bucket": self.label_bucket_for_state(state, config),
            },
        )

    def build_plan(self, gold: TaskGold) -> str:
        return (
            f"当前手牌有{gold.plan_aux['当前手牌']}。当前轮情况是{gold.plan_aux['当前轮情况']}；"
            f"若是首发，就从手牌能组成的合法牌型中枚举，若是跟牌，就只保留能按同牌型压过{gold.plan_aux['目标动作']}的动作；"
            f"若当前是允许过牌的跟牌局面，也把不要作为一个动作列出。这个样本的合法动作不超过{gold.plan_aux['N_all']}个，所以可以全部列出；动作之间不需要排序。"
        )

    def verify_answer(self, parsed: dict[str, Any], gold: TaskGold) -> VerificationResult:
        if set(parsed) != {"合法动作"} or not isinstance(parsed.get("合法动作"), list):
            return VerificationResult(False, "schema_error", {"keys": list(parsed)})
        raw_actions = []
        for action in parsed["合法动作"]:
            raw, result = validate_action_list(action)
            if not result.ok:
                return result
            raw_actions.append(raw)
        if len(raw_actions) != len(set(raw_actions)):
            return VerificationResult(False, "duplicate_action", {"actions": raw_actions})
        expected = set(gold.metadata["legal_actions_raw"])
        if set(raw_actions) != expected:
            return VerificationResult(False, "answer_mismatch", {"expected": sorted(expected), "actual": sorted(raw_actions)})
        return VerificationResult(True)


class SampleLegalActionsTask(TaskSpec):
    task_id = "L_sample_legal_actions"
    task_name = "任意列举K个合法动作"
    weight = 0.08

    def applies_to(self, state: dict[str, Any], config: GenerationConfig) -> bool:
        return super().applies_to(state, config) and len(state.get("actions", [])) >= config.list_k

    def build_gold(self, state: dict[str, Any], rng: random.Random, config: GenerationConfig) -> TaskGold:
        actions = rng.sample(list(state.get("actions", [])), config.list_k)
        answer = {"合法动作": [raw_action_to_display_list(action) for action in actions]}
        prompt = plan_answer_prompt(f"当前牌面下合法动作不少。请任意列举 {config.list_k} 种合法出牌。", '{"合法动作":[["3"],["4"],["5"],["6"]]}')
        return make_gold(self, state, prompt, answer, plan_aux={"K": config.list_k}, metadata={"K": config.list_k})

    def build_plan(self, gold: TaskGold) -> str:
        return (
            f"当前手牌有{gold.plan_aux['当前手牌']}。当前轮情况是{gold.plan_aux['当前轮情况']}；合法动作必须符合当前首发或跟牌规则。"
            f"题目只要求任意列举{gold.plan_aux['K']}个，所以选择{gold.plan_aux['K']}个互不重复、在当前局面下能出的动作即可，动作之间不需要排序。"
        )

    def verify_answer(self, parsed: dict[str, Any], gold: TaskGold) -> VerificationResult:
        if set(parsed) != {"合法动作"} or not isinstance(parsed.get("合法动作"), list):
            return VerificationResult(False, "schema_error", {"keys": list(parsed)})
        raw_actions = []
        for action in parsed["合法动作"]:
            raw, result = validate_action_list(action)
            if not result.ok:
                return result
            raw_actions.append(raw)
        expected_k = int(gold.metadata["K"])
        if len(raw_actions) != expected_k:
            return VerificationResult(False, "schema_error", {"expected_k": expected_k, "actual": len(raw_actions)})
        if len(raw_actions) != len(set(raw_actions)):
            return VerificationResult(False, "duplicate_action", {"actions": raw_actions})
        legal = set(gold.metadata["legal_actions_raw"])
        illegal = [action for action in raw_actions if action not in legal]
        if illegal:
            return VerificationResult(False, "illegal_action", {"illegal": illegal})
        return VerificationResult(True)


class CandidateLegalityTask(TaskSpec):
    task_id = "M_candidate_legality"
    task_name = "判断候选动作是否合法"
    weight = 0.05

    def label_bucket_weights(self, config: GenerationConfig) -> dict[str, float]:
        return {
            "candidate:legal": 0.45,
            "candidate:illegal_pass": 0.20,
            "candidate:illegal_missing_cards": 0.20,
            "candidate:illegal_rule": 0.15,
        }

    def _missing_card_candidates(self, hand: str, legal: set[str]) -> list[str]:
        counts = count_by_rank(hand)
        candidates: list[str] = []
        for rank in INTERNAL_RANKS:
            if counts[rank] == 0:
                candidates.append(rank)
            if counts[rank] < 2:
                candidates.append(rank * 2)
        return sorted({candidate for candidate in candidates if candidate not in legal}, key=action_sort_key)

    def _rule_negative_candidates(self, hand: str, legal: set[str]) -> list[str]:
        return sorted({action for action in playable_actions_from_hand(hand) if action not in legal}, key=action_sort_key)

    def _negative_candidates(self, state: dict[str, Any]) -> list[str]:
        hand = str(state.get("current_hand", ""))
        legal = set(state.get("actions", []))
        candidates = self._rule_negative_candidates(hand, legal)
        if "pass" not in legal:
            candidates.append("pass")
        candidates.extend(self._missing_card_candidates(hand, legal))
        return sorted({candidate for candidate in candidates if candidate not in legal}, key=action_sort_key)

    def _candidates_for_label_bucket(self, state: dict[str, Any], label_bucket: str) -> list[str]:
        hand = str(state.get("current_hand", ""))
        legal = list(state.get("actions", []))
        legal_set = set(legal)
        if label_bucket == "candidate:legal":
            return sorted(legal, key=action_sort_key)
        if label_bucket == "candidate:illegal_pass":
            return ["pass"] if "pass" not in legal_set else []
        if label_bucket == "candidate:illegal_missing_cards":
            return self._missing_card_candidates(hand, legal_set)
        if label_bucket == "candidate:illegal_rule":
            return self._rule_negative_candidates(hand, legal_set)
        return []

    def label_buckets_for_state(self, state: dict[str, Any], config: GenerationConfig) -> list[str]:
        buckets = [
            bucket
            for bucket in self.label_bucket_weights(config)
            if self._candidates_for_label_bucket(state, bucket)
        ]
        return buckets

    def _label_bucket_for_candidate(self, state: dict[str, Any], candidate: str, is_legal: bool) -> str:
        if is_legal:
            return "candidate:legal"
        if candidate == "pass":
            return "candidate:illegal_pass"
        if not contains_action(str(state.get("current_hand", "")), candidate):
            return "candidate:illegal_missing_cards"
        return "candidate:illegal_rule"

    def _make_gold_for_candidate(self, state: dict[str, Any], candidate: str, config: GenerationConfig) -> TaskGold:
        legal = list(state.get("actions", []))
        is_legal = candidate in set(legal)
        candidate_display = raw_action_to_display_list(candidate)
        answer = {"候选动作": candidate_display, "是否合法": is_legal}
        candidate_text = raw_action_to_display_text(candidate)
        if candidate == "pass":
            question = "当前局面下，选择不要是否是合法动作？"
        else:
            question = f"当前局面下，出 {candidate_text} 是否是合法动作？"
        prompt = plan_answer_prompt(question, '{"候选动作":["7","7"],"是否合法":true}')

        target = target_to_respond(state)
        candidate_label = action_type_label(candidate)
        if is_legal:
            if candidate == "pass":
                legal_reason = "当前需要跟牌，因此可以选择不要"
            elif target is None:
                legal_reason = f"这些牌能由手牌组成，并且本身是{candidate_label}"
            else:
                legal_reason = "这些牌能由手牌组成，并且能按当前跟牌规则压过目标动作"
            failure = ""
        else:
            hand = str(state.get("current_hand", ""))
            if candidate != "pass" and not contains_action(hand, candidate):
                failure = "候选牌不在当前手牌中"
            elif candidate == "pass":
                failure = "首发时不能选择不要"
            elif candidate_label is None:
                failure = "候选动作不是合法斗地主牌型"
            elif target is not None and action_type_label(target[1]) != candidate_label and candidate_label not in {"炸弹", "王炸"}:
                failure = "候选牌型不符合当前需要回应的牌型"
            else:
                failure = "候选动作无法压过当前需要回应的牌"
            legal_reason = ""
        label_bucket = self._label_bucket_for_candidate(state, candidate, is_legal)
        return make_gold(
            self,
            state,
            prompt,
            answer,
            plan_aux={
                "候选动作": candidate_display,
                "候选动作文本": candidate_text,
                "候选牌型": candidate_label or "无效牌型",
                "是否合法": is_legal,
                "合法原因": legal_reason,
                "失败原因": failure,
            },
            metadata={
                "candidate_action_raw": candidate,
                "candidate_action_display": candidate_display,
                "label_bucket": label_bucket,
                "candidate_legality_bucket": label_bucket,
            },
        )

    def build_gold_for_label(
        self,
        state: dict[str, Any],
        rng: random.Random,
        config: GenerationConfig,
        label_bucket: str,
    ) -> TaskGold | None:
        candidates = self._candidates_for_label_bucket(state, label_bucket)
        if not candidates:
            return None
        return self._make_gold_for_candidate(state, rng.choice(candidates), config)

    def build_gold(self, state: dict[str, Any], rng: random.Random, config: GenerationConfig) -> TaskGold | None:
        legal = list(state.get("actions", []))
        negatives = self._negative_candidates(state)
        choose_positive = bool(legal) and (not negatives or rng.random() < 0.5)
        if choose_positive:
            candidate = rng.choice(legal)
        elif negatives:
            candidate = rng.choice(negatives)
        elif legal:
            candidate = rng.choice(legal)
        else:
            return None
        return self._make_gold_for_candidate(state, candidate, config)

    def build_plan(self, gold: TaskGold) -> str:
        if gold.answer["是否合法"]:
            return (
                f"当前手牌有{gold.plan_aux['当前手牌']}。判断候选动作{gold.plan_aux['候选动作文本']}是否合法，"
                f"先看这些牌能否由当前手牌组成，再看它是否符合当前轮{gold.plan_aux['当前轮情况']}的出牌要求。"
                f"这里{gold.plan_aux['合法原因']}。因此它是合法动作。"
            )
        return (
            f"当前手牌有{gold.plan_aux['当前手牌']}。判断候选动作{gold.plan_aux['候选动作文本']}是否合法，"
            f"不能只看手里有没有这些牌，还要看当前轮{gold.plan_aux['当前轮情况']}是否允许这样出。"
            f"这里{gold.plan_aux['失败原因']}。因此它不是合法动作。"
        )

    def verify_answer(self, parsed: dict[str, Any], gold: TaskGold) -> VerificationResult:
        if set(parsed) != {"候选动作", "是否合法"} or not isinstance(parsed.get("是否合法"), bool):
            return VerificationResult(False, "schema_error", {"keys": list(parsed)})
        raw, result = validate_action_list(parsed.get("候选动作"))
        if not result.ok:
            return result
        if raw != gold.metadata["candidate_action_raw"]:
            return VerificationResult(False, "answer_mismatch", {"expected_candidate": gold.metadata["candidate_action_raw"], "actual": raw})
        return exact_answer({"候选动作": raw_action_to_display_list(raw), "是否合法": parsed["是否合法"]}, gold)


class CandidateTypeTask(TaskSpec):
    task_id = "N_candidate_type"
    task_name = "候选动作牌型识别"
    weight = 0.02

    def build_gold(self, state: dict[str, Any], rng: random.Random, config: GenerationConfig) -> TaskGold | None:
        hand = str(state.get("current_hand", ""))
        candidates = [action for action in playable_actions_from_hand(hand) if fine_type(action) is not None]
        if not candidates:
            return None
        candidate = rng.choice(candidates)
        label = action_type_label(candidate)
        if label is None:
            return None
        candidate_display = raw_action_to_display_list(candidate)
        answer = {"候选动作": candidate_display, "牌型": label}
        prompt = plan_answer_prompt(f"候选出牌 {raw_action_to_display_text(candidate)} 是什么牌型？", '{"候选动作":["3","3","4","4","5","5"],"牌型":"连对"}')
        return make_gold(
            self,
            state,
            prompt,
            answer,
            plan_aux={
                "候选动作": candidate_display,
                "候选动作文本": raw_action_to_display_text(candidate),
                "候选牌型": label,
                "牌型": label,
                "结构描述": structure_description(candidate),
            },
            metadata={"candidate_action_raw": candidate, "candidate_action_display": candidate_display},
        )

    def build_plan(self, gold: TaskGold) -> str:
        return (
            f"当前手牌有{gold.plan_aux['当前手牌']}。候选动作{gold.plan_aux['候选动作文本']}的牌型由牌值数量、连续关系和带牌结构决定。"
            f"它的结构是{gold.plan_aux['结构描述']}。因此对应斗地主牌型为{gold.plan_aux['牌型']}。"
        )

    def verify_answer(self, parsed: dict[str, Any], gold: TaskGold) -> VerificationResult:
        if set(parsed) != {"候选动作", "牌型"} or not isinstance(parsed.get("牌型"), str):
            return VerificationResult(False, "schema_error", {"keys": list(parsed)})
        raw, result = validate_action_list(parsed.get("候选动作"))
        if not result.ok:
            return result
        if raw != gold.metadata["candidate_action_raw"]:
            return VerificationResult(False, "answer_mismatch", {"expected_candidate": gold.metadata["candidate_action_raw"], "actual": raw})
        return exact_answer({"候选动作": raw_action_to_display_list(raw), "牌型": parsed["牌型"]}, gold)


def build_task_specs() -> list[TaskSpec]:
    return [
        CurrentHandTask(),
        RankCountTask(),
        SpecificRankCountTask(),
        MultiplicityTask("D_pairs", "列出对子", "对子", threshold=2, weight=0.03),
        MultiplicityTask("D_trios", "列出三张", "三张", threshold=3, weight=0.03),
        MultiplicityTask("D_fours", "列出四张", "四张", threshold=4, weight=0.03, exact=True),
        BombRocketTask(),
        LongestIntervalTask("F_straight", "识别顺子最长区间", "是否有顺子", "最长顺子", "顺子", min_count=1, min_length=5, weight=0.05),
        LongestIntervalTask("G_pair_chain", "识别连对最长区间", "是否有连对", "最长连对", "连对", min_count=2, min_length=3, weight=0.06),
        LongestIntervalTask("H_plane_body", "识别飞机主体最长区间", "是否有飞机主体", "最长飞机主体", "飞机主体", min_count=3, min_length=2, weight=0.04),
        TrioAttachTask(),
        PlaneAttachTask(),
        FourAttachTask(),
        CanPassTask(),
        AllLegalActionsTask(),
        SampleLegalActionsTask(),
        CandidateLegalityTask(),
        CandidateTypeTask(),
    ]
