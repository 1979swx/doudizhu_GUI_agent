"""Dou Dizhu visual QA SFT synthesis helpers."""

from .schemas import GenerationConfig, GeneratedSample, TaskGold, VerificationResult
from .task_specs import build_task_specs

__all__ = [
    "GenerationConfig",
    "GeneratedSample",
    "TaskGold",
    "VerificationResult",
    "build_task_specs",
]
