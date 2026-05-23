"""Strict response parser and verifier dispatch for Dou Dizhu QA."""

from __future__ import annotations

import json
import re
from typing import Any

from .schemas import ParsedResponse, TaskGold, VerificationResult

TAG_RE = re.compile(r"<(plan|answer)>(.*?)</\1>", re.DOTALL)
FORBIDDEN_PLAN_FRAGMENTS = ("others_hand", "其它玩家真实手牌", "对手具体手牌", "点击坐标", "胜率")


def parse_response(text: str, requires_plan: bool) -> tuple[ParsedResponse | None, VerificationResult]:
    if not isinstance(text, str):
        return None, VerificationResult(False, "schema_error", {"message": "response is not a string"})

    matches = list(TAG_RE.finditer(text))
    if not matches:
        return None, VerificationResult(False, "missing_answer_tag")

    tags = [match.group(1) for match in matches]
    expected_tags = ["plan", "answer"] if requires_plan else ["answer"]
    if tags != expected_tags:
        if "answer" not in tags:
            return None, VerificationResult(False, "missing_answer_tag", {"tags": tags})
        return None, VerificationResult(False, "schema_error", {"tags": tags, "expected_tags": expected_tags})

    cursor = 0
    for match in matches:
        if text[cursor:match.start()].strip():
            return None, VerificationResult(False, "unexpected_extra_text")
        cursor = match.end()
    if text[cursor:].strip():
        return None, VerificationResult(False, "unexpected_extra_text")

    plan = matches[0].group(2).strip() if requires_plan else None
    answer_match = matches[-1]
    raw_answer = answer_match.group(2).strip()
    try:
        answer = json.loads(raw_answer)
    except json.JSONDecodeError as exc:
        return None, VerificationResult(False, "invalid_json", {"message": str(exc)})
    if not isinstance(answer, dict):
        return None, VerificationResult(False, "schema_error", {"message": "answer JSON must be an object"})
    return ParsedResponse(plan=plan, answer=answer, raw_answer_text=raw_answer), VerificationResult(True)


def verify_plan(plan: str | None, gold: TaskGold) -> VerificationResult:
    if not gold.requires_plan:
        if plan is not None:
            return VerificationResult(False, "schema_error", {"message": "plan is not expected"})
        return VerificationResult(True)
    if not isinstance(plan, str) or not plan:
        return VerificationResult(False, "schema_error", {"message": "missing plan"})

    current_hand = str(gold.plan_aux.get("当前手牌", ""))
    expected_prefix = f"当前手牌有{current_hand}。"
    if not plan.startswith(expected_prefix):
        return VerificationResult(
            False,
            "plan_contradiction",
            {"message": "plan does not start with current hand", "expected_prefix": expected_prefix},
        )
    for fragment in FORBIDDEN_PLAN_FRAGMENTS:
        if fragment in plan:
            return VerificationResult(False, "plan_contradiction", {"forbidden_fragment": fragment})
    answer = gold.answer
    false_keys = [key for key, value in answer.items() if key.startswith("是否") and value is False]
    if false_keys and "都可以组成" in plan:
        return VerificationResult(False, "plan_contradiction", {"false_keys": false_keys})
    return VerificationResult(True)


def verify_response(task_spec: Any, response: str, gold: TaskGold) -> VerificationResult:
    parsed, format_result = parse_response(response, requires_plan=gold.requires_plan)
    if not format_result.ok or parsed is None:
        return format_result

    plan_result = verify_plan(parsed.plan, gold)
    if not plan_result.ok:
        return plan_result

    return task_spec.verify_answer(parsed.answer, gold)
