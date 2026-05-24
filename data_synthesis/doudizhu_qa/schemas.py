"""Shared dataclasses for Dou Dizhu QA synthesis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GenerationConfig:
    language: str = "zh"
    n_all: int = 3
    list_k: int = 4
    label_quotas_enabled: bool = True


@dataclass(frozen=True)
class ParsedResponse:
    plan: str | None
    answer: Any
    raw_answer_text: str


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    reason: str = "ok"
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskGold:
    task_id: str
    task_name: str
    prompt: str
    answer: dict[str, Any]
    plan_aux: dict[str, Any]
    metadata: dict[str, Any]
    requires_plan: bool


@dataclass(frozen=True)
class GeneratedSample:
    gold: TaskGold
    response: str
    verifier: VerificationResult
