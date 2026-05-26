#!/usr/bin/env python3
"""Collect and filter end-to-end Dou Dizhu SFT data.

The script deliberately keeps raw rollout collection separate from filtered SFT
materialization. Raw JSONL files are append-only audit assets; parquet files can
be rebuilt with ``--filter-only`` after changing filter thresholds.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
import json
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_system.environments.env_package.doudizhu.core.utils import CARD_TYPE  # noqa: E402
from agent_system.environments.env_package.doudizhu.envs import DoudizhuSingleEnv  # noqa: E402
from agent_system.environments.env_package.doudizhu.projection import (  # noqa: E402
    ACTION_INTERNAL_TO_CARD,
    doudizhu_projection,
    parse_doudizhu_action_tag,
)
from agent_system.environments.env_package.doudizhu.renderer import HitBox  # noqa: E402
from agent_system.environments.prompts.doudizhu import (  # noqa: E402
    DOUDIZHU_VISUAL_TEMPLATE,
    DOUDIZHU_VISUAL_TEMPLATE_ZH,
)

DATA_SOURCE = "doudizhu_end_to_end_sft"
ABILITY = "doudizhu_end_to_end"
REQUIRED_TAGS = ("plan", "action", "tool_call", "chat", "memory")
PARQUET_COLUMNS = ["data_source", "prompt", "question", "images", "answer", "ability", "reward_model", "extra_info"]
INITIAL_MEMORY_EN = "Initial turn. Read the screenshot, identify your hand, and plan the first landlord play."
INITIAL_MEMORY_ZH = "初始回合。阅读截图，识别你的手牌，并规划地主首轮出牌。"
NO_MEMORY_EN = "No previous memory."
NO_MEMORY_ZH = "没有上一轮记忆。"


@dataclass
class TokenLengths:
    prompt_tokens: int
    response_tokens: int
    full_sequence_tokens: int
    source: str
    prompt_text_tokens: int = 0
    prompt_image_tokens: int = 0


@dataclass
class ApiResult:
    ok: bool
    fatal: bool = False
    model: str = ""
    backend: str = ""
    response: str = ""
    request_id: str = ""
    latency_sec: float = 0.0
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    retry_count: int = 0
    error_type: str = ""
    error_message: str = ""
    http_status: int | None = None
    raw_response: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "fatal": self.fatal,
            "model": self.model,
            "backend": self.backend,
            "request_id": self.request_id,
            "latency_sec": self.latency_sec,
            "finish_reason": self.finish_reason,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "retry_count": self.retry_count,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "http_status": self.http_status,
        }


@dataclass
class CollectionTotals:
    api_calls: int = 0
    api_successes: int = 0
    api_failures: int = 0
    api_retries: int = 0
    http_429s: int = 0
    backoff_sleep_sec: float = 0.0
    estimated_cost: float = 0.0
    _lock: Any = field(default_factory=threading.Lock, repr=False, compare=False)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "api_calls": self.api_calls,
                "api_successes": self.api_successes,
                "api_failures": self.api_failures,
                "api_retries": self.api_retries,
                "http_429s": self.http_429s,
                "backoff_sleep_sec": self.backoff_sleep_sec,
                "estimated_cost": self.estimated_cost,
            }

    def should_stop(self, *, max_api_calls: int, max_cost: float) -> bool:
        with self._lock:
            if max_api_calls > 0 and self.api_calls >= max_api_calls:
                return True
            return bool(max_cost > 0 and self.estimated_cost >= max_cost)

    def add_api_call(self) -> None:
        with self._lock:
            self.api_calls += 1

    def add_success(self, *, prompt_tokens: int = 0, completion_tokens: int = 0, input_price: float = 0.0, output_price: float = 0.0) -> None:
        with self._lock:
            self.api_successes += 1
            self.estimated_cost += (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000.0

    def add_failure(self) -> None:
        with self._lock:
            self.api_failures += 1

    def add_retry(self) -> None:
        with self._lock:
            self.api_retries += 1

    def add_http_429(self) -> None:
        with self._lock:
            self.http_429s += 1

    def add_backoff_sleep(self, seconds: float) -> None:
        with self._lock:
            self.backoff_sleep_sec += float(seconds)


@dataclass
class EpisodeRunResult:
    split: str
    episode_seed: int
    episode_index: int
    steps: list[dict[str, Any]]
    summary: dict[str, Any]
    accepted_in_episode: int
    valid_steps_in_episode: int
    fatal_api_error: str = ""


@dataclass
class FilterConfig:
    max_response_tokens: int = 1024
    max_full_sequence_tokens: int = 2560
    max_prompt_tokens: int = 1536
    terminal_max_player0_hand: int = 2


@dataclass
class SplitBuildResult:
    rows: list[dict[str, Any]]
    stats: dict[str, Any]
    accepted_counts_by_episode: Counter[int] = field(default_factory=Counter)


class ApiRequestError(RuntimeError):
    def __init__(self, message: str, *, http_status: int | None = None, error_type: str = "api_error"):
        super().__init__(message)
        self.http_status = http_status
        self.error_type = error_type


class TokenCounter:
    """Best-effort target-model token counter.

    When a tokenizer is not provided, the fallback is intentionally conservative
    enough for progress accounting but is labelled as approximate in metadata.
    """

    def __init__(self, tokenizer_path: str | None, trust_remote_code: bool = True, enable_thinking: bool = False):
        self.tokenizer = None
        self.processor = None
        self.source = "approx_chars_div4"
        self.enable_thinking = bool(enable_thinking)
        self.lock = threading.Lock()
        self.image_token_cache: dict[tuple[int, int], int] = {}
        if tokenizer_path:
            self.tokenizer = self._load_tokenizer(tokenizer_path, trust_remote_code=trust_remote_code)
            self.processor = self._load_processor(tokenizer_path, trust_remote_code=trust_remote_code)
            self.source = str(tokenizer_path)

    @staticmethod
    def _load_tokenizer(path: str, trust_remote_code: bool):
        from transformers import AutoProcessor, AutoTokenizer

        try:
            return AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code, use_fast=True)
        except Exception as tokenizer_error:  # noqa: BLE001
            try:
                processor = AutoProcessor.from_pretrained(path, trust_remote_code=trust_remote_code, use_fast=True)
                tokenizer = getattr(processor, "tokenizer", None)
                if tokenizer is not None:
                    return tokenizer
            except Exception:
                pass
            raise tokenizer_error

    @staticmethod
    def _load_processor(path: str, trust_remote_code: bool):
        from transformers import AutoProcessor

        try:
            processor = AutoProcessor.from_pretrained(path, trust_remote_code=trust_remote_code, use_fast=True)
        except Exception:  # noqa: BLE001
            return None
        if getattr(processor, "image_processor", None) is None:
            return None
        return processor

    def _approx_count(self, text: str) -> int:
        text = text or ""
        return max(1 if text else 0, (len(text) + 3) // 4)

    def count_text(self, text: str) -> int:
        if self.tokenizer is None:
            return self._approx_count(text)
        with self.lock:
            return len(self.tokenizer.encode(text or "", add_special_tokens=False))

    @staticmethod
    def _tokenized_length(tokenized: Any) -> int:
        if hasattr(tokenized, "get"):
            input_ids = tokenized.get("input_ids")
            if input_ids is not None:
                return TokenCounter._tokenized_length(input_ids)
        if hasattr(tokenized, "tolist"):
            tokenized = tokenized.tolist()
        if isinstance(tokenized, (list, tuple)):
            if not tokenized:
                return 0
            first = tokenized[0]
            if hasattr(first, "tolist"):
                first = first.tolist()
            if isinstance(first, (list, tuple)):
                return len(first)
            return len(tokenized)
        return len(tokenized)

    def _chat_text(self, messages: list[dict[str, Any]], *, add_generation_prompt: bool) -> str:
        if self.tokenizer is None:
            return json.dumps(messages, ensure_ascii=False)
        try:
            with self.lock:
                return self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=add_generation_prompt,
                    tokenize=False,
                    enable_thinking=self.enable_thinking,
                )
        except TypeError:
            with self.lock:
                return self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=add_generation_prompt,
                    tokenize=False,
                )

    def _image_tokens_for_image(self, image: Any) -> int:
        if self.processor is None:
            return 0
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        elif not isinstance(image, Image.Image):
            return 0
        cache_key = (int(image.width), int(image.height))
        cached = self.image_token_cache.get(cache_key)
        if cached is not None:
            return cached
        with self.lock:
            image_inputs = self.processor.image_processor([image], return_tensors="pt")
        image_grid_thw = image_inputs.get("image_grid_thw")
        if image_grid_thw is None:
            return 0
        merge_size = int(getattr(self.processor.image_processor, "merge_size", 1))
        count = int(image_grid_thw[0].prod().item() // max(1, merge_size**2))
        self.image_token_cache[cache_key] = count
        return count

    def image_token_counts(self, images: list[Any] | None) -> list[int]:
        return [self._image_tokens_for_image(image) for image in images or []]

    def _vision_tokens(self) -> tuple[str, str, str]:
        if self.processor is None:
            return "<|vision_start|>", "<|image_pad|>", "<|vision_end|>"
        tokenizer = getattr(self.processor, "tokenizer", None) or self.tokenizer
        start = getattr(self.processor, "vision_start_token", None) or getattr(tokenizer, "vision_bos_token", None) or "<|vision_start|>"
        image = getattr(self.processor, "image_token", None) or getattr(tokenizer, "image_token", None) or "<|image_pad|>"
        end = getattr(self.processor, "vision_end_token", None) or getattr(tokenizer, "vision_eos_token", None) or "<|vision_end|>"
        return str(start), str(image), str(end)

    def _expand_image_placeholders(self, text: str, image_token_counts: list[int]) -> tuple[str, int]:
        if not image_token_counts:
            return text, 0
        start, image_token, end = self._vision_tokens()
        total_image_tokens = 0
        expanded = text
        for count in image_token_counts:
            total_image_tokens += int(count)
            expanded = expanded.replace("<image>", start + image_token * int(count) + end, 1)
        return expanded, total_image_tokens

    def count_messages(self, messages: list[dict[str, Any]], *, add_generation_prompt: bool, image_token_counts: list[int] | None = None) -> int:
        if self.tokenizer is None:
            return self._approx_count(json.dumps(messages, ensure_ascii=False))
        if image_token_counts:
            text = self._chat_text(messages, add_generation_prompt=add_generation_prompt)
            expanded, _image_tokens = self._expand_image_placeholders(text, image_token_counts)
            return self.count_text(expanded)
        try:
            with self.lock:
                token_ids = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=add_generation_prompt,
                    tokenize=True,
                    enable_thinking=self.enable_thinking,
                )
            return self._tokenized_length(token_ids)
        except TypeError:
            with self.lock:
                token_ids = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=add_generation_prompt,
                    tokenize=True,
                )
            return self._tokenized_length(token_ids)
        except Exception:  # noqa: BLE001
            text = "\n".join(str(message.get("content", "")) for message in messages)
            return self.count_text(text)

    def lengths(self, prompt: str, answer: str, images: list[Any] | None = None) -> TokenLengths:
        prompt_messages = [{"role": "user", "content": prompt}]
        full_messages = prompt_messages + [{"role": "assistant", "content": answer}]
        image_counts = self.image_token_counts(images)
        prompt_text_tokens = self.count_messages(prompt_messages, add_generation_prompt=True)
        prompt_tokens = self.count_messages(prompt_messages, add_generation_prompt=True, image_token_counts=image_counts)
        return TokenLengths(
            prompt_tokens=prompt_tokens,
            response_tokens=self.count_text(answer),
            full_sequence_tokens=self.count_messages(full_messages, add_generation_prompt=False, image_token_counts=image_counts),
            source=self.source,
            prompt_text_tokens=prompt_text_tokens,
            prompt_image_tokens=sum(image_counts),
        )


class SlidingWindowRateLimiter:
    def __init__(
        self,
        *,
        max_inflight_requests: int,
        requests_per_minute: int,
        tokens_per_minute: int,
        estimated_image_tokens: int,
    ):
        self.max_inflight_requests = int(max_inflight_requests)
        self.requests_per_minute = int(requests_per_minute)
        self.tokens_per_minute = int(tokens_per_minute)
        self.estimated_image_tokens = int(estimated_image_tokens)
        self.request_times: deque[float] = deque()
        self.token_events: deque[tuple[float, int]] = deque()
        self.total_sleep_sec = 0.0
        self.inflight_requests = 0
        self.condition = threading.Condition()

    @staticmethod
    def _trim(now: float, events: deque, *, tuple_events: bool) -> None:
        while events and now - (events[0][0] if tuple_events else events[0]) >= 60.0:
            events.popleft()

    def acquire(self, estimated_tokens: int) -> float:
        slept = 0.0
        estimated_tokens = max(0, int(estimated_tokens))
        while True:
            with self.condition:
                now = time.monotonic()
                self._trim(now, self.request_times, tuple_events=False)
                self._trim(now, self.token_events, tuple_events=True)

                wait_seconds = 0.0
                if self.max_inflight_requests > 0 and self.inflight_requests >= self.max_inflight_requests:
                    wait_seconds = 0.05
                if self.requests_per_minute > 0 and len(self.request_times) >= self.requests_per_minute:
                    wait_seconds = max(wait_seconds, 60.0 - (now - self.request_times[0]))
                if self.tokens_per_minute > 0:
                    used_tokens = sum(tokens for _ts, tokens in self.token_events)
                    if used_tokens + estimated_tokens > self.tokens_per_minute and self.token_events:
                        wait_seconds = max(wait_seconds, 60.0 - (now - self.token_events[0][0]))
                if wait_seconds <= 0:
                    self.inflight_requests += 1
                    self.request_times.append(now)
                    if estimated_tokens > 0:
                        self.token_events.append((now, estimated_tokens))
                    self.total_sleep_sec += slept
                    return slept
                self.condition.wait(timeout=wait_seconds)
                slept += wait_seconds

    def release(self) -> None:
        with self.condition:
            self.inflight_requests = max(0, self.inflight_requests - 1)
            self.condition.notify_all()


class OpenAICompatibleApiClient:
    def __init__(self, args: argparse.Namespace, token_counter: TokenCounter, totals: CollectionTotals):
        self.base_url = str(args.api_base_url).strip()
        self.model = str(args.api_model).strip()
        self.api_key_env = str(args.api_key_env).strip()
        self.timeout = float(args.api_timeout)
        self.thinking = str(args.api_thinking).strip()
        self.temperature = float(args.temperature)
        self.top_p = float(args.top_p)
        self.max_new_tokens = int(args.max_new_tokens)
        self.retry_max_attempts = int(args.retry_max_attempts)
        self.retry_backoff_min_seconds = float(args.retry_backoff_min_seconds)
        self.retry_backoff_max_seconds = float(args.retry_backoff_max_seconds)
        self.input_token_price_per_1m = float(args.input_token_price_per_1m)
        self.output_token_price_per_1m = float(args.output_token_price_per_1m)
        self.token_counter = token_counter
        self.totals = totals
        self.rate_limiter = SlidingWindowRateLimiter(
            max_inflight_requests=int(args.request_concurrency),
            requests_per_minute=int(args.request_rpm),
            tokens_per_minute=int(args.request_tpm),
            estimated_image_tokens=int(args.estimated_image_tokens),
        )
        if not self.model:
            raise ValueError("--api-model must be set for --model-backend api.")
        if not self.api_key_env:
            raise ValueError("--api-key-env cannot be empty.")
        if not os.environ.get(self.api_key_env):
            raise ValueError(f"API key env var {self.api_key_env!r} is not set.")
        if self.thinking not in {"default", "enabled", "disabled"}:
            raise ValueError("--api-thinking must be one of: default, enabled, disabled.")

    def _chat_url(self) -> str:
        base_url = self.base_url.rstrip("/")
        if not base_url:
            raise ValueError("--api-base-url cannot be empty.")
        if base_url.endswith("/chat/completions"):
            return base_url
        return f"{base_url}/chat/completions"

    @staticmethod
    def _message_content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(parts)
        return "" if content is None else str(content)

    @staticmethod
    def _image_data_url(obs: np.ndarray) -> str:
        png_bytes = image_to_png_bytes(obs)
        encoded = base64.b64encode(png_bytes).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def _payload(self, obs: np.ndarray, prompt: str) -> dict[str, Any]:
        prompt_text = (prompt or "").replace("<image>", "当前截图作为图片附件提供。")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": self._image_data_url(obs)}},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ],
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.thinking != "default":
            payload["thinking"] = {"type": self.thinking}
        return payload

    def _send(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {os.environ[self.api_key_env]}",
            "Content-Type": "application/json",
        }
        request = urllib_request.Request(self._chat_url(), data=data, headers=headers, method="POST")
        try:
            with urllib_request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 429:
                error_type = "api_rate_limited"
            elif exc.code in {400, 401, 403}:
                error_type = "api_invalid_request"
            else:
                error_type = "api_http_error"
            raise ApiRequestError(f"HTTP {exc.code}: {body}", http_status=exc.code, error_type=error_type) from exc
        except URLError as exc:
            raise ApiRequestError(str(exc), error_type="api_url_error") from exc
        except TimeoutError as exc:
            raise ApiRequestError(str(exc), error_type="api_timeout") from exc
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise ApiRequestError(f"Invalid JSON response: {body[:1000]}", error_type="api_invalid_json") from exc

    @staticmethod
    def _extract_response(parsed: dict[str, Any]) -> tuple[str, str]:
        choices = parsed.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ApiRequestError("API response has no choices.", error_type="api_invalid_response")
        first = choices[0]
        if not isinstance(first, dict):
            raise ApiRequestError("API response choice is not an object.", error_type="api_invalid_response")
        finish_reason = str(first.get("finish_reason") or "")
        message = first.get("message")
        if isinstance(message, dict):
            return OpenAICompatibleApiClient._message_content_to_text(message.get("content")).strip(), finish_reason
        return OpenAICompatibleApiClient._message_content_to_text(first.get("text")).strip(), finish_reason

    @staticmethod
    def _usage(parsed: dict[str, Any]) -> dict[str, int]:
        usage = parsed.get("usage") if isinstance(parsed, dict) else {}
        if not isinstance(usage, dict):
            usage = {}
        return {
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
        }

    @staticmethod
    def _request_id(parsed: dict[str, Any]) -> str:
        for key in ("id", "request_id"):
            value = parsed.get(key)
            if value:
                return str(value)
        return ""

    @staticmethod
    def _retriable(error: ApiRequestError) -> bool:
        if error.http_status in {429, 500, 502, 503, 504}:
            return True
        return error.error_type in {"api_timeout", "api_url_error"}

    @staticmethod
    def _fatal(error: ApiRequestError | None) -> bool:
        if error is None:
            return False
        return error.error_type == "api_invalid_request" or error.http_status in {400, 401, 403}

    def generate(self, obs: np.ndarray, prompt: str, state: dict[str, Any], env: DoudizhuSingleEnv) -> ApiResult:
        del state, env
        payload = self._payload(obs, prompt)
        estimated_tokens = self.token_counter.count_text(prompt) + self.max_new_tokens + self.rate_limiter.estimated_image_tokens
        started = time.monotonic()
        last_error: ApiRequestError | None = None
        attempts = max(1, self.retry_max_attempts)
        last_attempt_idx = 0
        for attempt_idx in range(attempts):
            last_attempt_idx = attempt_idx
            self.rate_limiter.acquire(estimated_tokens)
            self.totals.add_api_call()
            try:
                parsed = self._send(payload)
                response, finish_reason = self._extract_response(parsed)
                usage = self._usage(parsed)
                self.totals.add_success(
                    prompt_tokens=usage["prompt_tokens"],
                    completion_tokens=usage["completion_tokens"],
                    input_price=self.input_token_price_per_1m,
                    output_price=self.output_token_price_per_1m,
                )
                return ApiResult(
                    ok=True,
                    model=self.model,
                    backend="openai_compatible",
                    response=response,
                    request_id=self._request_id(parsed),
                    latency_sec=time.monotonic() - started,
                    finish_reason=finish_reason,
                    prompt_tokens=usage["prompt_tokens"],
                    completion_tokens=usage["completion_tokens"],
                    total_tokens=usage["total_tokens"],
                    retry_count=attempt_idx,
                    raw_response=parsed,
                )
            except ApiRequestError as exc:
                last_error = exc
                if exc.http_status == 429:
                    self.totals.add_http_429()
                if attempt_idx >= attempts - 1 or not self._retriable(exc):
                    break
                self.totals.add_retry()
                backoff = min(
                    self.retry_backoff_max_seconds,
                    self.retry_backoff_min_seconds * (2**attempt_idx),
                )
                backoff *= random.uniform(0.75, 1.25)
                time.sleep(backoff)
                self.totals.add_backoff_sleep(backoff)
            finally:
                self.rate_limiter.release()

        self.totals.add_failure()
        return ApiResult(
            ok=False,
            fatal=self._fatal(last_error),
            model=self.model,
            backend="openai_compatible",
            latency_sec=time.monotonic() - started,
            retry_count=last_attempt_idx,
            error_type=last_error.error_type if last_error else "api_error",
            error_message=str(last_error) if last_error else "unknown API error",
            http_status=last_error.http_status if last_error else None,
        )


class MockApiClient:
    """Local deterministic teacher used for dry-run and schema tests."""

    def __init__(self, args: argparse.Namespace, totals: CollectionTotals):
        self.policy = str(args.mock_policy)
        self.rng = random.Random(int(args.seed))
        self.lock = threading.Lock()
        self.totals = totals

    def _choose_action(self, state: dict[str, Any], env: DoudizhuSingleEnv) -> str:
        legal_actions = list(state.get("actions", []))
        if not legal_actions:
            return "pass"
        non_pass = sorted([action for action in legal_actions if action != "pass"], key=env._action_sort_key)
        if self.policy == "random":
            with self.lock:
                return self.rng.choice(sorted(legal_actions, key=env._action_sort_key))
        if self.policy == "first_non_pass" and non_pass:
            return non_pass[0]
        return env._fallback_action(state)

    def generate(self, obs: np.ndarray, prompt: str, state: dict[str, Any], env: DoudizhuSingleEnv) -> ApiResult:
        del obs, prompt
        started = time.monotonic()
        self.totals.add_api_call()
        try:
            action = self._choose_action(state, env)
            clicks = clicks_for_raw_action(env, state, action)
            action_text = raw_action_to_action_tag(action)
            click_text = ",".join(f"[{int(x)},{int(y)}]" for x, y in clicks)
            response = f"<plan>根据当前截图选择合法动作 {action_text}，并用界面点击提交。</plan><action>{action_text}</action><tool_call>left_click({click_text})</tool_call><chat>我按当前牌面稳妥处理。</chat><memory>上一手选择了 {action_text}，下轮继续以截图为准。</memory>"
            self.totals.add_success()
            return ApiResult(
                ok=True,
                model="mock",
                backend="mock",
                response=response,
                request_id=f"mock-{uuid.uuid4().hex[:8]}",
                latency_sec=time.monotonic() - started,
                finish_reason="stop",
            )
        except Exception as exc:  # noqa: BLE001
            self.totals.add_failure()
            return ApiResult(
                ok=False,
                model="mock",
                backend="mock",
                latency_sec=time.monotonic() - started,
                error_type=f"mock_error:{type(exc).__name__}",
                error_message=str(exc),
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect and filter end-to-end Dou Dizhu SFT data.")
    parser.add_argument("--output-dir", type=Path, default=Path("data_synthesis/doudizhu_end_to_end_sft"))
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--val-samples", type=int, default=500)
    parser.add_argument("--test-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--language", choices=("zh", "en"), default="zh")
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--max-clicks", type=int, default=20)
    parser.add_argument("--max-env-steps", type=int, default=30)
    parser.add_argument("--max-bot-turns", type=int, default=256)
    parser.add_argument("--max-raw-episodes", type=int, default=100000)
    parser.add_argument("--max-api-calls", type=int, default=0, help="0 means no global call cap.")
    parser.add_argument("--max-cost", type=float, default=0.0, help="0 means no estimated cost cap.")
    parser.add_argument("--max-consecutive-api-failures", type=int, default=3)
    parser.add_argument("--model-backend", default="api", choices=("api", "mock"))
    parser.add_argument("--mock-policy", default="first_non_pass", choices=("fallback", "first_non_pass", "random"))
    parser.add_argument("--api-base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.moonshot.cn/v1"))
    parser.add_argument("--api-model", default=os.environ.get("DOUDIZHU_API_MODEL", "kimi-k2.6"))
    parser.add_argument("--api-key-env", default="MOONSHOT_API_KEY")
    parser.add_argument("--api-timeout", type=float, default=90.0)
    parser.add_argument("--api-thinking", default="disabled", choices=("default", "enabled", "disabled"))
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Kimi K2.6 currently accepts only 0.6; override only for models that allow other values.",
    )
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--target-tokenizer-path", default=None)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-response-tokens", type=int, default=1024)
    parser.add_argument("--max-prompt-tokens", type=int, default=1536)
    parser.add_argument("--max-full-sequence-tokens", type=int, default=2560)
    parser.add_argument(
        "--terminal-max-hand",
        type=int,
        default=2,
        help="Accept normal-end episodes where player 0 has at most this many cards left at terminal. Use 0 for strict win/zero-hand filtering.",
    )
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel episode workers for collection.")
    parser.add_argument("--request-concurrency", type=int, default=64)
    parser.add_argument("--request-rpm", type=int, default=450)
    parser.add_argument("--request-tpm", type=int, default=2_700_000)
    parser.add_argument("--estimated-image-tokens", type=int, default=1024)
    parser.add_argument("--input-token-price-per-1m", type=float, default=0.0, help="Optional cost estimate for prompt tokens.")
    parser.add_argument("--output-token-price-per-1m", type=float, default=0.0, help="Optional cost estimate for completion tokens.")
    parser.add_argument("--rate-limit-tier", default="Tier2")
    parser.add_argument("--retry-max-attempts", type=int, default=4)
    parser.add_argument("--retry-backoff-min-seconds", type=float, default=2.0)
    parser.add_argument("--retry-backoff-max-seconds", type=float, default=60.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Remove generated files for requested splits before collection.")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--write-raw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-raw-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--filter-only", action="store_true")
    parser.add_argument("--review-samples", type=int, default=40)
    return parser.parse_args()


def make_env_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "doudizhu": {
            "use_ray": False,
            "language": args.language,
            "chinese_mode": args.language == "zh",
            "image_width": args.image_width,
            "image_height": args.image_height,
            "max_clicks": args.max_clicks,
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


def make_filter_config(args: argparse.Namespace) -> FilterConfig:
    return FilterConfig(
        max_response_tokens=int(args.max_response_tokens),
        max_full_sequence_tokens=int(args.max_full_sequence_tokens),
        max_prompt_tokens=int(args.max_prompt_tokens),
        terminal_max_player0_hand=int(args.terminal_max_hand),
    )


def split_specs(args: argparse.Namespace) -> list[tuple[str, int, int]]:
    return [
        ("train", int(args.train_samples), int(args.seed)),
        ("val", int(args.val_samples), int(args.seed) + 10_000_000),
        ("test", int(args.test_samples), int(args.seed) + 20_000_000),
    ]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:  # noqa: BLE001
        return ""
    return result.stdout.strip()


def text_sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def image_to_png_bytes(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image.astype(np.uint8), mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def image_to_png_dict(image: np.ndarray) -> dict[str, bytes]:
    return {"bytes": image_to_png_bytes(image)}


def image_sha256(png_bytes: bytes) -> str:
    return hashlib.sha256(png_bytes).hexdigest()


def write_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(to_jsonable(record), ensure_ascii=False, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Counter):
        return dict(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(to_jsonable(item) for item in value)
    return value


def compact_trace(trace: list[Any], limit: int = 8) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in trace[-limit:]:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            compact.append({"player": int(item[0]), "action": str(item[1])})
        else:
            compact.append({"player": -1, "action": str(item)})
    return compact


def compact_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "self": int(state.get("self", 0)),
        "landlord": int(state.get("landlord", 0)),
        "current_hand": str(state.get("current_hand", "")),
        "seen_cards": str(state.get("seen_cards", "")),
        "played_cards": list(state.get("played_cards", [])),
        "num_cards_left": list(state.get("num_cards_left", [])),
        "legal_actions": list(state.get("actions", [])),
        "trace_tail": compact_trace(list(state.get("trace", []))),
    }


def build_prompt(language: str, memory: str) -> str:
    template = DOUDIZHU_VISUAL_TEMPLATE_ZH if language == "zh" else DOUDIZHU_VISUAL_TEMPLATE
    no_memory = NO_MEMORY_ZH if language == "zh" else NO_MEMORY_EN
    return template.format(previous_memory=memory or no_memory)


def initial_memory(language: str) -> str:
    return INITIAL_MEMORY_ZH if language == "zh" else INITIAL_MEMORY_EN


def sanitize_memory(memory: str, max_chars: int) -> str:
    memory = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", memory or "").strip()
    if max_chars > 0 and len(memory) > max_chars:
        return memory[:max_chars].rstrip()
    return memory


def normalized_answer(response: str, normalized_action_text: str) -> str:
    if not response or not normalized_action_text:
        return response or ""

    def replace(match: re.Match[str]) -> str:
        return f"{match.group(1)}{normalized_action_text}{match.group(3)}"

    return re.sub(r"(<action>)(.*?)(</action>)", replace, response.strip(), count=1, flags=re.IGNORECASE | re.DOTALL)


def xml_tag_counts(response: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tag in REQUIRED_TAGS:
        counts[tag] = len(re.findall(rf"<{tag}>.*?</{tag}>", response or "", flags=re.IGNORECASE | re.DOTALL))
    return counts


def raw_action_to_action_tag(action: str) -> str:
    if action == "pass":
        return "[pass]"
    return "[" + ", ".join(ACTION_INTERNAL_TO_CARD.get(card, card) for card in action) + "]"


def action_category(action: str | None) -> str:
    if action == "pass":
        return "pass"
    if not action:
        return "unknown"
    card_types = CARD_TYPE[0].get(action)
    if not card_types:
        return "other"
    fine_type = str(card_types[0][0])
    if fine_type == "solo":
        return "solo"
    if fine_type == "pair":
        return "pair"
    if fine_type.startswith("trio"):
        return "trio"
    if "chain" in fine_type:
        return "chain"
    if fine_type in {"bomb", "rocket"}:
        return "bomb_rocket"
    return "other"


def bucket_count(value: int, buckets: tuple[int, ...] = (1, 2, 5, 10, 20)) -> str:
    value = int(value)
    lower = 0
    for upper in buckets:
        if value <= upper:
            return f"{lower + 1}-{upper}" if lower + 1 != upper else str(upper)
        lower = upper
    return f">{buckets[-1]}"


def game_phase(num_cards_left: list[int]) -> str:
    player_cards = int(num_cards_left[0]) if num_cards_left else 0
    if player_cards > 12:
        return "early"
    if player_cards > 6:
        return "mid"
    return "late"


def has_bomb_or_rocket(legal_actions: list[str]) -> bool:
    return any(action_category(action) == "bomb_rocket" for action in legal_actions)


def turn_type(legal_actions: list[str]) -> str:
    return "respond" if "pass" in set(legal_actions) else "lead"


def norm_point(renderer: Any, px: float, py: float) -> list[int]:
    x = int(round(px / float(renderer.width - 1) * 1000.0))
    y = int(round(py / float(renderer.height - 1) * 1000.0))
    return [min(1000, max(0, x)), min(1000, max(0, y))]


def center_of_hitbox(renderer: Any, hitbox: HitBox) -> list[int]:
    x0, y0, x1, y1 = hitbox.box
    return norm_point(renderer, (x0 + x1) / 2.0, (y0 + y1) / 2.0)


def hitbox_by_kind(hitboxes: list[HitBox], kind: str) -> HitBox:
    for hitbox in hitboxes:
        if hitbox.kind == kind:
            return hitbox
    raise ValueError(f"Missing hitbox kind={kind!r}")


def hitbox_by_card_index(hitboxes: list[HitBox], index: int) -> HitBox:
    for hitbox in hitboxes:
        if hitbox.kind == "card" and int(hitbox.payload) == int(index):
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


def clicks_for_raw_action(env: DoudizhuSingleEnv, state: dict[str, Any], action: str) -> list[list[int]]:
    hitboxes = env.renderer.get_hitboxes(state)
    if action == "pass":
        return [center_of_hitbox(env.renderer, hitbox_by_kind(hitboxes, "pass"))]
    indices = selected_indices_for_action(str(state.get("current_hand", "")), action)
    clicks = [center_of_hitbox(env.renderer, hitbox_by_card_index(hitboxes, idx)) for idx in indices]
    clicks.append(center_of_hitbox(env.renderer, hitbox_by_kind(hitboxes, "play")))
    return clicks


def episode_paths(output_dir: Path, split: str) -> tuple[Path, Path]:
    raw_dir = output_dir / "raw"
    return raw_dir / f"{split}_steps.jsonl", raw_dir / f"{split}_episodes.jsonl"


def generated_paths_for_split(output_dir: Path, split: str) -> list[Path]:
    steps_path, episodes_path = episode_paths(output_dir, split)
    return [
        steps_path,
        episodes_path,
        output_dir / f"{split}.parquet",
        output_dir / "reports" / f"{split}_stats.json",
    ]


def generated_dirs_for_split(output_dir: Path, split: str) -> list[Path]:
    return [output_dir / "raw" / "images" / split]


def generated_data_exists(output_dir: Path, splits: list[str]) -> bool:
    for split in splits:
        if any(path.exists() for path in generated_paths_for_split(output_dir, split)):
            return True
        if any(path.exists() for path in generated_dirs_for_split(output_dir, split)):
            return True
    return False


def remove_generated_data(output_dir: Path, splits: list[str]) -> None:
    for split in splits:
        for path in generated_paths_for_split(output_dir, split):
            if path.exists():
                path.unlink()
        for path in generated_dirs_for_split(output_dir, split):
            if path.exists():
                shutil.rmtree(path)


def image_path_for(output_dir: Path, split: str, episode_seed: int, step_index: int) -> Path:
    return output_dir / "raw" / "images" / split / f"seed_{episode_seed}_step_{step_index:03d}.png"


def relative_to_output(path: Path, output_dir: Path) -> str:
    try:
        return str(path.relative_to(output_dir))
    except ValueError:
        return str(path)


def write_raw_image(output_dir: Path, split: str, episode_seed: int, step_index: int, obs: np.ndarray) -> tuple[str, str]:
    png_bytes = image_to_png_bytes(obs)
    path = image_path_for(output_dir, split, episode_seed, step_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png_bytes)
    return relative_to_output(path, output_dir), image_sha256(png_bytes)


def resolve_raw_image_path(output_dir: Path, image_path: str) -> Path:
    path = Path(image_path)
    if path.is_absolute():
        return path
    return output_dir / path


def normalize_num_cards_left(value: Any) -> list[int]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and value:
        cards: list[int] = []
        for item in value:
            try:
                cards.append(int(item))
            except (TypeError, ValueError):
                return []
        return cards
    return []


def final_num_cards_left_from_step(step: dict[str, Any]) -> list[int]:
    env_cards = normalize_num_cards_left(step.get("env_info", {}).get("num_cards_left"))
    if env_cards:
        return env_cards
    return normalize_num_cards_left(step.get("state_before", {}).get("num_cards_left"))


def final_num_cards_left_from_episode(episode: dict[str, Any]) -> list[int]:
    cards = normalize_num_cards_left(episode.get("final_num_cards_left"))
    if cards:
        return cards
    return normalize_num_cards_left(episode.get("num_cards_left"))


def final_player0_num_cards_left(episode: dict[str, Any]) -> int | None:
    for key in ("final_player0_num_cards_left", "terminal_player0_num_cards_left"):
        value = episode.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    cards = final_num_cards_left_from_episode(episode)
    if cards:
        return int(cards[0])
    if bool(episode.get("won", False)):
        return 0
    return None


def enrich_episodes_with_terminal_counts(episodes: list[dict[str, Any]], steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    steps_by_seed: dict[int, list[dict[str, Any]]] = {}
    for step in steps:
        steps_by_seed.setdefault(int(step.get("episode_seed", -1)), []).append(step)
    enriched: list[dict[str, Any]] = []
    for episode in episodes:
        record = dict(episode)
        if final_player0_num_cards_left(record) is None:
            seed = int(record.get("episode_seed", -1))
            episode_steps = sorted(steps_by_seed.get(seed, []), key=lambda item: int(item.get("step_index", -1)))
            if episode_steps:
                final_cards = final_num_cards_left_from_step(episode_steps[-1])
                if final_cards:
                    record["final_num_cards_left"] = final_cards
                    record["final_player0_num_cards_left"] = int(final_cards[0])
        enriched.append(record)
    return enriched


def episode_terminal_qualified(episode: dict[str, Any], filter_config: FilterConfig) -> bool:
    final_player0 = final_player0_num_cards_left(episode)
    return final_player0 is not None and final_player0 <= int(filter_config.terminal_max_player0_hand)


def filter_reasons_for_step(
    step: dict[str, Any],
    episode: dict[str, Any],
    filter_config: FilterConfig,
) -> list[str]:
    reasons: list[str] = []
    api = step.get("api", {})
    if not api.get("ok", False):
        reasons.append(api.get("error_type") or "api_error")
    final_player0 = final_player0_num_cards_left(episode)
    if final_player0 is None:
        reasons.append("terminal_player0_hand_unknown")
    elif final_player0 > int(filter_config.terminal_max_player0_hand):
        reasons.append("terminal_player0_hand_gt_threshold")
    if not bool(episode.get("normal_end", False)):
        reasons.append("episode_truncated")
    projection = step.get("projection", {})
    action_parse = step.get("action_parse", {})
    if int(projection.get("projection_valid", 0)) != 1:
        reasons.append("projection_invalid")
        if action_parse.get("error") == "missing_required_tags":
            reasons.append("missing_required_tag")
        elif action_parse.get("parse_ok", False) and projection.get("raw_tool_call_text"):
            reasons.append("tool_call_parse_fail")
        elif not projection.get("raw_tool_call_text"):
            reasons.append("tool_call_empty")
    env_info = step.get("env_info", {})
    if float(env_info.get("click_valid_ratio", 0.0)) != 1.0:
        reasons.append("click_invalid")
    if env_info and env_info.get("submit_kind") is None:
        reasons.append("no_submit")
    if float(env_info.get("rule_action_valid", 0.0)) != 1.0:
        reasons.append("rule_invalid")
    if bool(env_info.get("fallback_used", False)):
        reasons.append("fallback_used")
    if not bool(action_parse.get("parse_ok", False)):
        reasons.append("action_tag_parse_fail")
    if action_parse.get("raw") != env_info.get("game_action"):
        reasons.append("action_mismatch")
    tokens = step.get("tokens", {})
    if int(tokens.get("response_tokens_target_model", 0)) > filter_config.max_response_tokens:
        reasons.append("response_too_long")
    if int(tokens.get("prompt_tokens_target_model", 0)) > filter_config.max_prompt_tokens:
        reasons.append("prompt_too_long")
    if int(tokens.get("full_sequence_tokens_target_model", 0)) > filter_config.max_full_sequence_tokens:
        reasons.append("full_sequence_too_long")
    if not step.get("image_path"):
        reasons.append("missing_image")
    return reasons


def is_accepted_step(step: dict[str, Any], episode: dict[str, Any], filter_config: FilterConfig) -> bool:
    return len(filter_reasons_for_step(step, episode, filter_config)) == 0


def image_for_step(output_dir: Path, step: dict[str, Any]) -> Image.Image | None:
    image_path = resolve_raw_image_path(output_dir, str(step.get("image_path", "")))
    if not image_path.exists():
        return None
    try:
        return Image.open(image_path).convert("RGB")
    except Exception:  # noqa: BLE001
        return None


def refresh_step_token_lengths(steps: list[dict[str, Any]], token_counter: TokenCounter, output_dir: Path) -> list[dict[str, Any]]:
    refreshed: list[dict[str, Any]] = []
    for step in steps:
        record = dict(step)
        answer = str(record.get("normalized_answer") or record.get("raw_response") or "")
        image = image_for_step(output_dir, record)
        token_lengths = token_counter.lengths(str(record.get("prompt", "")), answer, images=[image] if image is not None else None)
        tokens = dict(record.get("tokens", {}))
        tokens.update(
            {
                "prompt_tokens_target_model": token_lengths.prompt_tokens,
                "prompt_text_tokens_target_model": token_lengths.prompt_text_tokens,
                "prompt_image_tokens_target_model": token_lengths.prompt_image_tokens,
                "response_tokens_target_model": token_lengths.response_tokens,
                "full_sequence_tokens_target_model": token_lengths.full_sequence_tokens,
                "tokenizer": token_lengths.source,
                "response_tokens_le_512": token_lengths.response_tokens <= 512,
            }
        )
        record["tokens"] = tokens
        refreshed.append(record)
    return refreshed


def step_distribution_tags(step: dict[str, Any]) -> dict[str, str]:
    state = step.get("state_before", {})
    legal_actions = list(state.get("legal_actions", []))
    num_cards_left = list(state.get("num_cards_left", []))
    hand_len = len(str(state.get("current_hand", "")))
    opponent_counts = [int(x) for x in num_cards_left[1:]] if len(num_cards_left) >= 3 else []
    opponent_min = min(opponent_counts) if opponent_counts else 0
    game_action = step.get("env_info", {}).get("game_action")
    return {
        "action_category": action_category(game_action),
        "turn_type": turn_type(legal_actions),
        "hand_length_bucket": bucket_count(hand_len, buckets=(3, 6, 10, 14, 20)),
        "legal_action_count_bucket": bucket_count(len(legal_actions), buckets=(1, 3, 10, 30, 100)),
        "opponent_min_card_count_bucket": bucket_count(opponent_min, buckets=(1, 2, 5, 10, 20)),
        "pass_legal": str("pass" in set(legal_actions)).lower(),
        "bomb_or_rocket_available": str(has_bomb_or_rocket(legal_actions)).lower(),
        "game_phase": game_phase(num_cards_left),
    }


def make_step_record(
    *,
    split: str,
    episode_seed: int,
    episode_index: int,
    step_index: int,
    prompt: str,
    prompt_memory_before: str,
    obs: np.ndarray,
    output_dir: Path,
    args: argparse.Namespace,
    state_before: dict[str, Any],
    api_result: ApiResult,
    projected_action: dict[str, Any],
    env_info: dict[str, Any] | None,
    reward: float | None,
    done: bool,
    memory_after: str,
    token_lengths: TokenLengths,
) -> dict[str, Any]:
    image_path = ""
    image_hash = ""
    if args.write_raw_images:
        image_path, image_hash = write_raw_image(output_dir, split, episode_seed, step_index, obs)
    raw_action_text = projected_action.get("raw_action_text", "")
    action_parse = parse_doudizhu_action_tag(raw_action_text)
    answer = normalized_answer(api_result.response, action_parse.get("normalized_text", ""))
    selected_cards = (env_info or {}).get("selected_cards", "")
    selected_before_pass = action_parse.get("raw") == "pass" and bool(selected_cards)
    tag_counts = xml_tag_counts(api_result.response)
    projection = {
        "projection_valid": int(projected_action.get("projection_valid", 0)),
        "clicks": projected_action.get("clicks", []),
        "raw_tool_call_text": projected_action.get("raw_tool_call_text", ""),
        "tool_calls": projected_action.get("tool_calls", []),
        "tool_calling": projected_action.get("tool_calling", 0),
        "plan": projected_action.get("plan", ""),
        "chat": projected_action.get("chat", ""),
        "memory": projected_action.get("memory", ""),
        "semantic_action": projected_action.get("semantic_action", ""),
    }
    return {
        "schema_version": 1,
        "data_source": DATA_SOURCE,
        "split": split,
        "episode_seed": int(episode_seed),
        "episode_index": int(episode_index),
        "step_index": int(step_index),
        "sample_id": f"{DATA_SOURCE}:{split}:{episode_seed}:{step_index}:{uuid.uuid4().hex[:8]}",
        "created_at": utc_now(),
        "prompt": prompt,
        "prompt_memory_before": prompt_memory_before,
        "response_memory_raw": projected_action.get("memory", ""),
        "memory_after_sanitized": memory_after,
        "image_path": image_path,
        "image_sha256": image_hash,
        "raw_response": api_result.response,
        "normalized_answer": answer,
        "response_chars": len(api_result.response or ""),
        "api": api_result.to_dict(),
        "projection": projection,
        "action_parse": {
            "parse_ok": bool(action_parse.get("parse_ok", False)),
            "raw_text": raw_action_text,
            "raw": action_parse.get("raw", ""),
            "cards": action_parse.get("cards", []),
            "normalized_text": action_parse.get("normalized_text", ""),
            "error": action_parse.get("error", ""),
            "multiset_match": Counter(action_parse.get("raw", "")) == Counter(str((env_info or {}).get("game_action", ""))),
        },
        "env_info": compact_env_info(env_info or {}),
        "reward": None if reward is None else float(reward),
        "done": bool(done),
        "state_before": compact_state(state_before),
        "tokens": {
            "prompt_tokens_target_model": token_lengths.prompt_tokens,
            "prompt_text_tokens_target_model": token_lengths.prompt_text_tokens,
            "prompt_image_tokens_target_model": token_lengths.prompt_image_tokens,
            "response_tokens_target_model": token_lengths.response_tokens,
            "full_sequence_tokens_target_model": token_lengths.full_sequence_tokens,
            "tokenizer": token_lengths.source,
            "response_tokens_le_512": token_lengths.response_tokens <= 512,
        },
        "verifier_aux": {
            "selected_before_pass": bool(selected_before_pass),
            "click_count": len(projected_action.get("clicks", [])),
            "submit_kind": (env_info or {}).get("submit_kind"),
            "tag_counts": tag_counts,
            "all_required_tags_once": all(tag_counts.get(tag, 0) == 1 for tag in REQUIRED_TAGS),
        },
        "distribution": step_distribution_tags({"state_before": compact_state(state_before), "env_info": env_info or {}}),
    }


def compact_env_info(info: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "won",
        "task_score",
        "reward",
        "current_player",
        "landlord",
        "num_cards_left",
        "legal_actions",
        "trace",
        "winner_id",
        "payoffs",
        "state_summary",
        "click_valid_ratio",
        "rule_action_valid",
        "fallback_used",
        "game_action",
        "hand_cards_reduced",
        "hand_depletion_reward",
        "hand_depletion_reward_total",
        "win_reward",
        "selected_cards",
        "submit_kind",
        "bot_turns",
        "bot_limit_reached",
        "projection_valid",
        "validity_reward_delta",
        "validity_reward_average",
        "chat",
        "memory",
        "semantic_action",
        "tool_calls",
        "tool_calling",
    )
    return {key: to_jsonable(info.get(key)) for key in keys if key in info}


def episode_summary(
    *,
    split: str,
    episode_seed: int,
    episode_index: int,
    steps: list[dict[str, Any]],
    env: DoudizhuSingleEnv,
    terminated_reason: str,
    normal_end: bool,
) -> dict[str, Any]:
    last_info = steps[-1].get("env_info", {}) if steps else env._build_info(reward=0.0)
    final_cards_left = normalize_num_cards_left(last_info.get("num_cards_left"))
    fallback_steps = sum(1 for step in steps if bool(step.get("env_info", {}).get("fallback_used", False)))
    invalid_steps = sum(1 for step in steps if int(step.get("projection", {}).get("projection_valid", 0)) != 1)
    return {
        "schema_version": 1,
        "data_source": DATA_SOURCE,
        "split": split,
        "episode_seed": int(episode_seed),
        "episode_index": int(episode_index),
        "created_at": utc_now(),
        "normal_end": bool(normal_end),
        "terminated_reason": terminated_reason,
        "episode_length": len(steps),
        "winner_id": last_info.get("winner_id"),
        "won": bool(float(last_info.get("won", 0.0)) > 0),
        "payoffs": last_info.get("payoffs", [0, 0, 0]),
        "final_num_cards_left": final_cards_left,
        "final_player0_num_cards_left": int(final_cards_left[0]) if final_cards_left else None,
        "total_reward": float(sum(float(step.get("reward") or 0.0) for step in steps)),
        "fallback_steps": int(fallback_steps),
        "invalid_steps": int(invalid_steps),
        "api_failed_steps": sum(1 for step in steps if not bool(step.get("api", {}).get("ok", False))),
        "bot_limit_reached": any(bool(step.get("env_info", {}).get("bot_limit_reached", False)) for step in steps),
    }


def collection_budget_reached(args: argparse.Namespace, totals: CollectionTotals) -> bool:
    return totals.should_stop(max_api_calls=int(args.max_api_calls), max_cost=float(args.max_cost))


def valid_step_count(steps: list[dict[str, Any]]) -> int:
    return sum(
        1
        for step in steps
        if int(step.get("projection", {}).get("projection_valid", 0)) == 1 and float(step.get("env_info", {}).get("click_valid_ratio", 0.0)) == 1.0 and float(step.get("env_info", {}).get("rule_action_valid", 0.0)) == 1.0 and not bool(step.get("env_info", {}).get("fallback_used", False))
    )


def run_episode(
    *,
    split: str,
    episode_seed: int,
    episode_index: int,
    args: argparse.Namespace,
    api_client: Any,
    token_counter: TokenCounter,
    totals: CollectionTotals,
) -> EpisodeRunResult:
    filter_config = make_filter_config(args)
    env = DoudizhuSingleEnv(seed=episode_seed, env_config=make_env_config(args))
    obs, _reset_info = env.reset(seed=episode_seed)
    memory = initial_memory(args.language)
    episode_steps: list[dict[str, Any]] = []
    terminated_reason = "done"
    normal_end = False
    consecutive_api_failures = 0
    fatal_api_error = ""

    for step_index in range(int(args.max_env_steps)):
        if env.done:
            terminated_reason = "done"
            normal_end = True
            break
        if collection_budget_reached(args, totals):
            terminated_reason = "max_api_calls" if int(args.max_api_calls) > 0 else "max_cost"
            break

        state_before = dict(env.game.state)
        memory_before = memory
        prompt = build_prompt(args.language, memory_before)
        api_result = api_client.generate(obs, prompt, state_before, env)
        if not api_result.ok:
            consecutive_api_failures += 1
            projected_action = doudizhu_projection([""], max_clicks=args.max_clicks)[0][0]
            answer = ""
            token_lengths = token_counter.lengths(prompt, answer, images=[obs])
            step_record = make_step_record(
                split=split,
                episode_seed=episode_seed,
                episode_index=episode_index,
                step_index=step_index,
                prompt=prompt,
                prompt_memory_before=memory_before,
                obs=obs,
                output_dir=args.output_dir,
                args=args,
                state_before=state_before,
                api_result=api_result,
                projected_action=projected_action,
                env_info=None,
                reward=None,
                done=env.done,
                memory_after=memory_before,
                token_lengths=token_lengths,
            )
            episode_steps.append(step_record)
            if api_result.fatal:
                terminated_reason = api_result.error_type or "fatal_api_error"
                fatal_api_error = api_result.error_message
                break
            if consecutive_api_failures >= args.max_consecutive_api_failures:
                terminated_reason = "api_abort"
                break
            continue

        consecutive_api_failures = 0
        projected_actions, _valids = doudizhu_projection([api_result.response], max_clicks=args.max_clicks)
        projected_action = projected_actions[0]
        normalized = normalized_answer(api_result.response, projected_action.get("normalized_action_text", ""))
        token_lengths = token_counter.lengths(prompt, normalized, images=[obs])
        next_obs, reward, done, env_info = env.step(projected_action)
        response_memory = projected_action.get("memory", "")
        memory_after = memory_before
        if int(projected_action.get("projection_valid", 0)) == 1 and response_memory:
            memory_after = sanitize_memory(response_memory, max_chars=512)
        step_record = make_step_record(
            split=split,
            episode_seed=episode_seed,
            episode_index=episode_index,
            step_index=step_index,
            prompt=prompt,
            prompt_memory_before=memory_before,
            obs=obs,
            output_dir=args.output_dir,
            args=args,
            state_before=state_before,
            api_result=api_result,
            projected_action=projected_action,
            env_info=env_info,
            reward=reward,
            done=done,
            memory_after=memory_after,
            token_lengths=token_lengths,
        )
        memory = memory_after
        episode_steps.append(step_record)
        obs = next_obs
        if done:
            terminated_reason = "bot_limit_reached" if bool(env_info.get("bot_limit_reached", False)) else "done"
            normal_end = terminated_reason == "done"
            break
    else:
        terminated_reason = "max_env_steps"

    summary = episode_summary(
        split=split,
        episode_seed=episode_seed,
        episode_index=episode_index,
        steps=episode_steps,
        env=env,
        terminated_reason=terminated_reason,
        normal_end=normal_end,
    )
    accepted_in_episode = sum(1 for step in episode_steps if is_accepted_step(step, summary, filter_config))
    return EpisodeRunResult(
        split=split,
        episode_seed=episode_seed,
        episode_index=episode_index,
        steps=episode_steps,
        summary=summary,
        accepted_in_episode=accepted_in_episode,
        valid_steps_in_episode=valid_step_count(episode_steps),
        fatal_api_error=fatal_api_error,
    )


def write_episode_result(result: EpisodeRunResult, steps_path: Path, episodes_path: Path, args: argparse.Namespace) -> None:
    if not args.write_raw:
        return
    for step in result.steps:
        write_jsonl(steps_path, step)
    write_jsonl(episodes_path, result.summary)


def log_episode_progress(
    *,
    split: str,
    completed_episodes: int,
    accepted_count: int,
    target_samples: int,
    totals: CollectionTotals,
    result: EpisodeRunResult,
) -> None:
    totals_snapshot = totals.snapshot()
    print(
        f"{split}: episodes={completed_episodes}, accepted={accepted_count}/{target_samples}, "
        f"api_calls={totals_snapshot['api_calls']}, api_ok={totals_snapshot['api_successes']}, "
        f"last_seed={result.episode_seed}, last={result.summary.get('terminated_reason')}, "
        f"last_won={result.summary.get('won')}, last_len={result.summary.get('episode_length')}, "
        f"last_valid_steps={result.valid_steps_in_episode}, last_accepted={result.accepted_in_episode}",
        flush=True,
    )


def first_episode_index(seed_start: int, existing_episodes: list[dict[str, Any]]) -> int:
    if not existing_episodes:
        return 0
    return max(0, max(int(record.get("episode_seed", seed_start)) - seed_start for record in existing_episodes) + 1)


def collect_split(
    *,
    split: str,
    target_samples: int,
    seed_start: int,
    args: argparse.Namespace,
    api_client: Any,
    token_counter: TokenCounter,
    totals: CollectionTotals,
) -> None:
    if target_samples <= 0:
        return
    if int(args.num_workers) <= 1:
        collect_split_serial(
            split=split,
            target_samples=target_samples,
            seed_start=seed_start,
            args=args,
            api_client=api_client,
            token_counter=token_counter,
            totals=totals,
        )
        return
    collect_split_parallel(
        split=split,
        target_samples=target_samples,
        seed_start=seed_start,
        args=args,
        api_client=api_client,
        token_counter=token_counter,
        totals=totals,
    )


def collect_split_serial(
    *,
    split: str,
    target_samples: int,
    seed_start: int,
    args: argparse.Namespace,
    api_client: Any,
    token_counter: TokenCounter,
    totals: CollectionTotals,
) -> None:
    steps_path, episodes_path = episode_paths(args.output_dir, split)
    existing_episodes = load_jsonl(episodes_path) if args.resume else []
    completed_seeds = {int(record["episode_seed"]) for record in existing_episodes}
    filter_config = make_filter_config(args)
    existing_result = build_split_outputs(split=split, args=args, token_counter=token_counter, filter_config=filter_config, write_outputs=False) if args.resume and steps_path.exists() else SplitBuildResult(rows=[], stats={})
    accepted_count = len(existing_result.rows)
    episode_index = first_episode_index(seed_start, existing_episodes)
    completed_episodes = 0

    while accepted_count < target_samples:
        if args.max_raw_episodes > 0 and completed_episodes >= args.max_raw_episodes:
            break
        if collection_budget_reached(args, totals):
            break
        episode_seed = seed_start + episode_index
        episode_index += 1
        if episode_seed in completed_seeds:
            continue
        result = run_episode(split=split, episode_seed=episode_seed, episode_index=episode_index - 1, args=args, api_client=api_client, token_counter=token_counter, totals=totals)
        completed_episodes += 1
        write_episode_result(result, steps_path, episodes_path, args)
        if result.fatal_api_error:
            raise RuntimeError(f"Fatal API error while collecting {split}: {result.fatal_api_error}")
        accepted_count += result.accepted_in_episode
        if args.log_every > 0 and completed_episodes % args.log_every == 0:
            log_episode_progress(split=split, completed_episodes=completed_episodes, accepted_count=accepted_count, target_samples=target_samples, totals=totals, result=result)


def collect_split_parallel(
    *,
    split: str,
    target_samples: int,
    seed_start: int,
    args: argparse.Namespace,
    api_client: Any,
    token_counter: TokenCounter,
    totals: CollectionTotals,
) -> None:
    steps_path, episodes_path = episode_paths(args.output_dir, split)
    existing_episodes = load_jsonl(episodes_path) if args.resume else []
    completed_seeds = {int(record["episode_seed"]) for record in existing_episodes}
    filter_config = make_filter_config(args)
    existing_result = build_split_outputs(split=split, args=args, token_counter=token_counter, filter_config=filter_config, write_outputs=False) if args.resume and steps_path.exists() else SplitBuildResult(rows=[], stats={})
    accepted_count = len(existing_result.rows)
    episode_index = first_episode_index(seed_start, existing_episodes)
    scheduled_episodes = 0
    completed_episodes = 0
    max_workers = max(1, int(args.num_workers))
    futures: dict[Future, tuple[int, int]] = {}
    stop_scheduling = False

    def schedule_next(executor: ThreadPoolExecutor) -> bool:
        nonlocal episode_index, scheduled_episodes, stop_scheduling
        if stop_scheduling or accepted_count >= target_samples or collection_budget_reached(args, totals):
            return False
        if args.max_raw_episodes > 0 and scheduled_episodes >= args.max_raw_episodes:
            return False
        while True:
            episode_seed = seed_start + episode_index
            current_episode_index = episode_index
            episode_index += 1
            if episode_seed not in completed_seeds:
                break
        future = executor.submit(
            run_episode,
            split=split,
            episode_seed=episode_seed,
            episode_index=current_episode_index,
            args=args,
            api_client=api_client,
            token_counter=token_counter,
            totals=totals,
        )
        futures[future] = (episode_seed, current_episode_index)
        scheduled_episodes += 1
        return True

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=f"doudizhu-{split}") as executor:
        while len(futures) < max_workers and schedule_next(executor):
            pass
        while futures:
            done_futures, _pending = wait(futures, return_when=FIRST_COMPLETED)
            for future in done_futures:
                futures.pop(future)
                result = future.result()
                completed_episodes += 1
                write_episode_result(result, steps_path, episodes_path, args)
                if result.fatal_api_error:
                    stop_scheduling = True
                    for pending_future in futures:
                        pending_future.cancel()
                    raise RuntimeError(f"Fatal API error while collecting {split}: {result.fatal_api_error}")
                accepted_count += result.accepted_in_episode
                if args.log_every > 0 and completed_episodes % args.log_every == 0:
                    log_episode_progress(split=split, completed_episodes=completed_episodes, accepted_count=accepted_count, target_samples=target_samples, totals=totals, result=result)
            while len(futures) < max_workers and schedule_next(executor):
                pass


def rows_for_accepted_steps(
    *,
    steps: list[dict[str, Any]],
    episodes_by_seed: dict[int, dict[str, Any]],
    output_dir: Path,
    split: str,
    filter_config: FilterConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sample_index = 0
    for step in steps:
        episode = episodes_by_seed.get(int(step.get("episode_seed", -1)))
        if episode is None:
            continue
        if not is_accepted_step(step, episode, filter_config):
            continue
        image_path = resolve_raw_image_path(output_dir, str(step.get("image_path", "")))
        if not image_path.exists():
            continue
        image_bytes = image_path.read_bytes()
        env_info = step.get("env_info", {})
        action_parse = step.get("action_parse", {})
        final_cards_left = final_num_cards_left_from_episode(episode)
        final_player0_cards_left = final_player0_num_cards_left(episode)
        prior_steps = [prior for prior in steps if int(prior.get("episode_seed", -1)) == int(step.get("episode_seed", -2)) and int(prior.get("step_index", -1)) < int(step.get("step_index", -1))]
        prior_invalid_steps = sum(1 for prior in prior_steps if int(prior.get("projection", {}).get("projection_valid", 0)) != 1)
        prior_fallback_steps = sum(1 for prior in prior_steps if bool(prior.get("env_info", {}).get("fallback_used", False)))
        verifier = {
            "projection_valid": int(step.get("projection", {}).get("projection_valid", 0)),
            "click_valid_ratio": float(env_info.get("click_valid_ratio", 0.0)),
            "rule_action_valid": float(env_info.get("rule_action_valid", 0.0)),
            "fallback_used": bool(env_info.get("fallback_used", False)),
            "action_tag_parse_ok": bool(action_parse.get("parse_ok", False)),
            "action_match": action_parse.get("raw") == env_info.get("game_action"),
            "response_tokens_target_model": int(step.get("tokens", {}).get("response_tokens_target_model", 0)),
            "prompt_tokens_target_model": int(step.get("tokens", {}).get("prompt_tokens_target_model", 0)),
            "prompt_text_tokens_target_model": int(step.get("tokens", {}).get("prompt_text_tokens_target_model", 0)),
            "prompt_image_tokens_target_model": int(step.get("tokens", {}).get("prompt_image_tokens_target_model", 0)),
            "full_sequence_tokens_target_model": int(step.get("tokens", {}).get("full_sequence_tokens_target_model", 0)),
            "selected_before_pass": bool(step.get("verifier_aux", {}).get("selected_before_pass", False)),
        }
        extra_info = {
            "sample_id": step.get("sample_id") or f"{DATA_SOURCE}:{split}:{sample_index}:{uuid.uuid4().hex[:8]}",
            "split": split,
            "sample_index": sample_index,
            "episode_seed": int(step.get("episode_seed", 0)),
            "episode_index": int(step.get("episode_index", 0)),
            "step_index": int(step.get("step_index", 0)),
            "source_model": step.get("api", {}).get("model", ""),
            "source_backend": step.get("api", {}).get("backend", ""),
            "language": "zh" if "当前游戏屏幕" in str(step.get("prompt", "")) else "en",
            "prompt_memory_before": step.get("prompt_memory_before", ""),
            "memory_after_sanitized": step.get("memory_after_sanitized", ""),
            "game_action": env_info.get("game_action"),
            "action_category": step.get("distribution", {}).get("action_category", action_category(env_info.get("game_action"))),
            "action_tag_raw_text": action_parse.get("raw_text", ""),
            "action_tag_raw": action_parse.get("raw", ""),
            "action_tag_cards": action_parse.get("cards", []),
            "action_match": action_parse.get("raw") == env_info.get("game_action"),
            "selected_cards": env_info.get("selected_cards", ""),
            "selected_indices": selected_indices_from_step(step),
            "tool_clicks": step.get("projection", {}).get("clicks", []),
            "submit_kind": env_info.get("submit_kind"),
            "legal_actions": step.get("state_before", {}).get("legal_actions", []),
            "current_hand": step.get("state_before", {}).get("current_hand", ""),
            "num_cards_left": step.get("state_before", {}).get("num_cards_left", []),
            "trace_tail": step.get("state_before", {}).get("trace_tail", []),
            "episode": {
                "won": bool(episode.get("won", False)),
                "episode_length": int(episode.get("episode_length", 0)),
                "winner_id": episode.get("winner_id"),
                "payoffs": episode.get("payoffs", [0, 0, 0]),
                "total_reward": float(episode.get("total_reward", 0.0)),
                "final_num_cards_left": final_cards_left,
                "final_player0_num_cards_left": final_player0_cards_left,
                "terminal_max_player0_hand": int(filter_config.terminal_max_player0_hand),
                "terminal_player0_hand_qualified": episode_terminal_qualified(episode, filter_config),
                "prior_invalid_steps": int(prior_invalid_steps),
                "prior_fallback_steps": int(prior_fallback_steps),
                "prior_has_invalid_or_fallback": bool(prior_invalid_steps or prior_fallback_steps),
            },
            "verifier": verifier,
            "api": step.get("api", {}),
            "tokens": step.get("tokens", {}),
            "distribution": step.get("distribution", {}),
            "raw_response": step.get("raw_response", ""),
        }
        rows.append(
            {
                "data_source": DATA_SOURCE,
                "prompt": [{"role": "user", "content": step.get("prompt", "")}],
                "question": step.get("prompt", ""),
                "images": [{"bytes": image_bytes}],
                "answer": step.get("normalized_answer", step.get("raw_response", "")),
                "ability": ABILITY,
                "reward_model": {"style": "rule", "ground_truth": env_info.get("game_action")},
                "extra_info": extra_info,
            }
        )
        sample_index += 1
    return rows


def selected_indices_from_step(step: dict[str, Any]) -> list[int]:
    selected_cards = str(step.get("env_info", {}).get("selected_cards", ""))
    hand = str(step.get("state_before", {}).get("current_hand", ""))
    if not selected_cards:
        return []
    try:
        return selected_indices_for_action(hand, selected_cards)
    except ValueError:
        return []


def write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=PARQUET_COLUMNS).to_parquet(path, index=False)


def safe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def safe_median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def summarize_stats(
    *,
    split: str,
    steps: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    filter_config: FilterConfig,
) -> dict[str, Any]:
    episodes_by_seed = {int(episode["episode_seed"]): episode for episode in episodes}
    completed = [episode for episode in episodes if bool(episode.get("normal_end", False))]
    won = [episode for episode in completed if bool(episode.get("won", False))]
    terminal_qualified = [episode for episode in completed if episode_terminal_qualified(episode, filter_config)]
    final_player0_counts = [final_player0_num_cards_left(episode) for episode in completed]
    final_player0_counts = [count for count in final_player0_counts if count is not None]
    payoffs = [float((episode.get("payoffs") or [0])[0]) for episode in completed]
    lengths = [float(episode.get("episode_length", 0)) for episode in completed]
    rejection_counts: Counter[str] = Counter()
    for step in steps:
        episode = episodes_by_seed.get(int(step.get("episode_seed", -1)), {})
        rejection_counts.update(filter_reasons_for_step(step, episode, filter_config))
    distribution_counts: dict[str, Counter[str]] = {
        "action_category": Counter(),
        "turn_type": Counter(),
        "game_phase": Counter(),
        "hand_length_bucket": Counter(),
        "legal_action_count_bucket": Counter(),
        "opponent_min_card_count_bucket": Counter(),
        "pass_legal": Counter(),
        "bomb_or_rocket_available": Counter(),
    }
    accepted_final_player0_counts: Counter[str] = Counter()
    accepted_prior_context_counts: Counter[str] = Counter()
    for row in rows:
        distribution = row.get("extra_info", {}).get("distribution", {})
        for key in distribution_counts:
            distribution_counts[key][str(distribution.get(key, "unknown"))] += 1
        episode_info = row.get("extra_info", {}).get("episode", {})
        accepted_final_player0_counts[str(episode_info.get("final_player0_num_cards_left", "unknown"))] += 1
        accepted_prior_context_counts["prior_dirty" if episode_info.get("prior_has_invalid_or_fallback") else "prior_clean"] += 1
    projection_valid = [float(step.get("projection", {}).get("projection_valid", 0)) for step in steps]
    click_ratios = [float(step.get("env_info", {}).get("click_valid_ratio", 0.0)) for step in steps if step.get("env_info")]
    rule_valid = [float(step.get("env_info", {}).get("rule_action_valid", 0.0)) for step in steps if step.get("env_info")]
    fallback = [float(bool(step.get("env_info", {}).get("fallback_used", False))) for step in steps if step.get("env_info")]
    action_parse_ok = [float(bool(step.get("action_parse", {}).get("parse_ok", False))) for step in steps]
    action_match = [float(step.get("action_parse", {}).get("raw") == step.get("env_info", {}).get("game_action")) for step in steps if step.get("env_info")]
    response_tokens = [int(step.get("tokens", {}).get("response_tokens_target_model", 0)) for step in steps]
    full_tokens = [int(step.get("tokens", {}).get("full_sequence_tokens_target_model", 0)) for step in steps]
    return {
        "split": split,
        "episode_metrics": {
            "num_episodes": len(episodes),
            "num_completed_episodes": len(completed),
            "num_won_episodes": len(won),
            "num_terminal_qualified_episodes": len(terminal_qualified),
            "win_rate": float(len(won) / len(completed)) if completed else 0.0,
            "terminal_qualified_rate": float(len(terminal_qualified) / len(completed)) if completed else 0.0,
            "terminal_max_player0_hand": int(filter_config.terminal_max_player0_hand),
            "avg_final_player0_num_cards_left": safe_mean([float(count) for count in final_player0_counts]),
            "final_player0_num_cards_left_distribution": dict(Counter(str(count) for count in final_player0_counts)),
            "avg_episode_length": safe_mean(lengths),
            "median_episode_length": safe_median(lengths),
            "truncated_rate": float(sum(not bool(ep.get("normal_end", False)) for ep in episodes) / len(episodes)) if episodes else 0.0,
            "bot_limit_rate": float(sum(ep.get("terminated_reason") == "bot_limit_reached" for ep in episodes) / len(episodes)) if episodes else 0.0,
            "api_abort_rate": float(sum(ep.get("terminated_reason") == "api_abort" for ep in episodes) / len(episodes)) if episodes else 0.0,
            "avg_payoff_player0": safe_mean(payoffs),
            "winner_id_distribution": dict(Counter(str(ep.get("winner_id")) for ep in completed)),
        },
        "step_metrics": {
            "num_steps": len(steps),
            "projection_valid_rate": safe_mean(projection_valid),
            "click_valid_ratio_mean": safe_mean(click_ratios),
            "click_valid_all_rate": safe_mean([float(value == 1.0) for value in click_ratios]),
            "rule_action_valid_rate": safe_mean(rule_valid),
            "fallback_rate": safe_mean(fallback),
            "action_tag_parse_rate": safe_mean(action_parse_ok),
            "action_match_rate": safe_mean(action_match),
            "accepted_steps": len(rows),
            "accepted_step_rate": float(len(rows) / len(steps)) if steps else 0.0,
            "response_tokens_avg": safe_mean([float(x) for x in response_tokens]),
            "response_tokens_p50": safe_median([float(x) for x in response_tokens]),
            "full_sequence_tokens_avg": safe_mean([float(x) for x in full_tokens]),
            "full_sequence_tokens_p50": safe_median([float(x) for x in full_tokens]),
        },
        "rejection_counts": dict(rejection_counts),
        "accepted_distribution": {key: dict(counter) for key, counter in distribution_counts.items()},
        "accepted_episode_distribution": {
            "final_player0_num_cards_left": dict(accepted_final_player0_counts),
            "prior_context": dict(accepted_prior_context_counts),
        },
    }


def build_split_outputs(
    *,
    split: str,
    args: argparse.Namespace,
    token_counter: TokenCounter,
    filter_config: FilterConfig,
    write_outputs: bool = True,
) -> SplitBuildResult:
    steps_path, episodes_path = episode_paths(args.output_dir, split)
    steps = refresh_step_token_lengths(load_jsonl(steps_path), token_counter, args.output_dir)
    episodes = enrich_episodes_with_terminal_counts(load_jsonl(episodes_path), steps)
    episodes_by_seed = {int(record["episode_seed"]): record for record in episodes}
    rows = rows_for_accepted_steps(
        steps=steps,
        episodes_by_seed=episodes_by_seed,
        output_dir=args.output_dir,
        split=split,
        filter_config=filter_config,
    )
    stats = summarize_stats(split=split, steps=steps, episodes=episodes, rows=rows, filter_config=filter_config)
    if write_outputs:
        write_parquet(rows, args.output_dir / f"{split}.parquet")
        report_path = args.output_dir / "reports" / f"{split}_stats.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(to_jsonable(stats), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{split}: wrote {len(rows)} filtered rows to {args.output_dir / f'{split}.parquet'}", flush=True)
    counts = Counter(int(row["extra_info"]["episode_seed"]) for row in rows)
    return SplitBuildResult(rows=rows, stats=stats, accepted_counts_by_episode=counts)


def build_metadata(args: argparse.Namespace, split_results: dict[str, SplitBuildResult], totals: CollectionTotals, token_counter: TokenCounter) -> dict[str, Any]:
    totals_snapshot = totals.snapshot()
    return {
        "data_source": DATA_SOURCE,
        "created_at": utc_now(),
        "git_commit": git_commit(),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "env_config": make_env_config(args),
        "model_config": {
            "backend": args.model_backend,
            "api_base_url": args.api_base_url if args.model_backend == "api" else "",
            "api_model": args.api_model if args.model_backend == "api" else "",
            "api_key_env": args.api_key_env if args.model_backend == "api" else "",
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "api_thinking": args.api_thinking,
            "api_timeout": args.api_timeout,
        },
        "prompt": {
            "language": args.language,
            "template_sha256": text_sha256(DOUDIZHU_VISUAL_TEMPLATE_ZH if args.language == "zh" else DOUDIZHU_VISUAL_TEMPLATE),
            "template_zh_sha256": text_sha256(DOUDIZHU_VISUAL_TEMPLATE_ZH),
            "template_en_sha256": text_sha256(DOUDIZHU_VISUAL_TEMPLATE),
        },
        "rate_limit": {
            "tier": args.rate_limit_tier,
            "request_concurrency": args.request_concurrency,
            "request_rpm": args.request_rpm,
            "request_tpm": args.request_tpm,
            "estimated_image_tokens": args.estimated_image_tokens,
            "http_429s": totals_snapshot["http_429s"],
            "backoff_sleep_sec": totals_snapshot["backoff_sleep_sec"],
        },
        "api_totals": to_jsonable(totals_snapshot),
        "cost_config": {
            "max_cost": args.max_cost,
            "input_token_price_per_1m": args.input_token_price_per_1m,
            "output_token_price_per_1m": args.output_token_price_per_1m,
            "estimated_cost": totals_snapshot["estimated_cost"],
        },
        "filter_config": {
            "episode_acceptance": "terminal_player0_num_cards_left_le_threshold",
            "terminal_player_id": 0,
            "terminal_max_player0_hand": int(args.terminal_max_hand),
            "strict_win_equivalent_terminal_max_hand": 0,
            "episode_normal_end": True,
            "projection_valid": 1,
            "click_valid_ratio": 1.0,
            "rule_action_valid": 1.0,
            "fallback_used": False,
            "action_tag_raw_equals_game_action": True,
            "max_response_tokens": args.max_response_tokens,
            "max_prompt_tokens": args.max_prompt_tokens,
            "max_full_sequence_tokens": args.max_full_sequence_tokens,
        },
        "format": {
            "parquet_columns": PARQUET_COLUMNS,
            "image_storage": "images=[{'bytes': PNG_BYTES}]",
            "response": "five XML tags: <plan>, <action>, <tool_call>, <chat>, <memory>",
            "action_contract": "list of display cards: [pass], [3, 3], [10, J], [BJ, RJ]",
        },
        "tokenizer": {"target_tokenizer_path": args.target_tokenizer_path, "token_length_source": token_counter.source},
        "splits": {split: result.stats for split, result in split_results.items()},
    }


def png_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def load_image_from_row(row: dict[str, Any]) -> Image.Image:
    entry = row["images"][0]
    return Image.open(io.BytesIO(entry["bytes"])).convert("RGB")


def annotate_image(image: Image.Image, clicks: list[list[Any]]) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    width, height = annotated.size
    points: list[tuple[float, float]] = []
    for click in clicks:
        if isinstance(click, list) and len(click) == 2:
            points.append((float(click[0]) * width / 1000.0, float(click[1]) * height / 1000.0))
    if len(points) >= 2:
        draw.line(points, fill=(255, 202, 40), width=3)
    for index, (px, py) in enumerate(points, start=1):
        radius = 10
        fill = (29, 142, 64) if index == len(points) else (220, 38, 38)
        draw.ellipse([px - radius, py - radius, px + radius, py + radius], fill=fill, outline=(255, 255, 255), width=3)
        label = str(index)
        bbox = draw.textbbox((0, 0), label)
        draw.rectangle([px + radius + 4, py - radius, px + radius + (bbox[2] - bbox[0]) + 12, py - radius + (bbox[3] - bbox[1]) + 8], fill=(20, 24, 31))
        draw.text((px + radius + 8, py - radius + 4), label, fill=(255, 255, 255))
    return annotated


def write_review(output_dir: Path, rows_by_split: dict[str, list[dict[str, Any]]], num_samples: int, seed: int) -> None:
    all_rows: list[tuple[str, dict[str, Any]]] = []
    for split, rows in rows_by_split.items():
        all_rows.extend((split, row) for row in rows)
    rng = random.Random(seed)
    rng.shuffle(all_rows)
    selected = all_rows[: max(0, min(num_samples, len(all_rows)))]
    blocks: list[str] = []
    for index, (split, row) in enumerate(selected):
        extra = row.get("extra_info", {})
        image = annotate_image(load_image_from_row(row), extra.get("tool_clicks", []))
        image_url = png_data_url(image)
        action = html.escape(str(extra.get("game_action", "")))
        answer = html.escape(str(row.get("answer", "")))
        question = html.escape(str(row.get("question", "")))
        pretty_extra = html.escape(json.dumps(to_jsonable(extra), ensure_ascii=False, indent=2))
        blocks.append(
            f"""
            <section class="sample">
              <div class="sample-head">
                <h2>#{index} {html.escape(split)} action={action}</h2>
                <div class="badges">
                  <span>{html.escape(str(extra.get("action_category", "unknown")))}</span>
                  <span>seed {html.escape(str(extra.get("episode_seed", "")))}</span>
                  <span>step {html.escape(str(extra.get("step_index", "")))}</span>
                </div>
              </div>
              <img src="{image_url}" />
              <div class="grid">
                <div><h3>Answer</h3><pre>{answer}</pre></div>
                <div><h3>Extra Info</h3><pre>{pretty_extra}</pre></div>
              </div>
              <h3>Question</h3><pre>{question}</pre>
            </section>
            """
        )
    summary = {
        "rows": len(all_rows),
        "sampled": len(selected),
        "splits": {split: len(rows) for split, rows in rows_by_split.items()},
    }
    html_text = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <title>Dou Dizhu End-to-End SFT Samples</title>
      <style>
        body {{ font-family: sans-serif; margin: 24px; background: #f6f7f9; color: #20242a; }}
        .sample {{ background: white; border: 1px solid #d8dee8; border-radius: 8px; padding: 16px; margin-bottom: 20px; }}
        .sample-head {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; }}
        .badges {{ display: flex; gap: 8px; flex-wrap: wrap; }}
        .badges span {{ background: #e9eef6; border: 1px solid #d3dbe8; border-radius: 999px; padding: 4px 10px; font-size: 13px; }}
        img {{ width: 640px; max-width: 100%; display: block; border: 1px solid #ccd3dd; }}
        pre {{ white-space: pre-wrap; word-break: break-word; background: #f0f2f5; padding: 12px; border-radius: 6px; }}
        .grid {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 12px; }}
        @media (max-width: 860px) {{ .grid {{ grid-template-columns: 1fr; }} }}
      </style>
    </head>
    <body>
      <h1>Dou Dizhu End-to-End SFT Samples</h1>
      <p>Output: {html.escape(str(output_dir))}</p>
      <h2>Summary</h2>
      <pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre>
      {"".join(blocks)}
    </body>
    </html>
    """
    review_path = output_dir / "reports" / "review.html"
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text(html_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.train_samples < 0 or args.val_samples < 0 or args.test_samples < 0:
        raise ValueError("Sample counts must be non-negative.")
    if args.terminal_max_hand < 0:
        raise ValueError("--terminal-max-hand must be non-negative.")
    if not args.filter_only and (not args.write_raw or not args.write_raw_images):
        raise ValueError("Collection mode requires --write-raw and --write-raw-images so filtered parquet can be rebuilt.")
    active_splits = [split for split, target_samples, _seed_start in split_specs(args) if target_samples > 0]
    if not args.filter_only and active_splits:
        if args.overwrite:
            remove_generated_data(args.output_dir, active_splits)
        elif not args.resume and generated_data_exists(args.output_dir, active_splits):
            raise ValueError(f"{args.output_dir} already contains generated raw/parquet files for {active_splits}. Use --resume, --overwrite, --filter-only, or a new --output-dir.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    token_counter = TokenCounter(args.target_tokenizer_path, trust_remote_code=args.trust_remote_code)
    filter_config = make_filter_config(args)
    totals = CollectionTotals()

    if not args.filter_only:
        client: Any
        if args.model_backend == "mock":
            client = MockApiClient(args, totals)
        else:
            client = OpenAICompatibleApiClient(args, token_counter, totals)
        for split, target_samples, seed_start in split_specs(args):
            collect_split(
                split=split,
                target_samples=target_samples,
                seed_start=seed_start,
                args=args,
                api_client=client,
                token_counter=token_counter,
                totals=totals,
            )

    split_results: dict[str, SplitBuildResult] = {}
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split, target_samples, _seed_start in split_specs(args):
        if target_samples <= 0 and not (args.output_dir / "raw" / f"{split}_steps.jsonl").exists():
            continue
        result = build_split_outputs(
            split=split,
            args=args,
            token_counter=token_counter,
            filter_config=filter_config,
            write_outputs=True,
        )
        split_results[split] = result
        rows_by_split[split] = result.rows

    metadata = build_metadata(args, split_results, totals, token_counter)
    (args.output_dir / "metadata.json").write_text(json.dumps(to_jsonable(metadata), ensure_ascii=False, indent=2), encoding="utf-8")
    write_review(args.output_dir, rows_by_split, args.review_samples, args.seed)


if __name__ == "__main__":
    main()
