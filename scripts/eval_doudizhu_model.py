#!/usr/bin/env python3
"""Evaluate a model on doudizhu and doudizhu_grounding without training.

The two tasks use different statistical units:
- doudizhu: independent stochastic trajectories.
- doudizhu_grounding: multiple samples for the same canonical state, then
  aggregate by state before aggregating globally.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import multiprocessing as mp
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from agent_system.environments.env_package.doudizhu import build_doudizhu_envs, doudizhu_projection
from agent_system.environments.env_package.doudizhu.projection import extract_tag
from agent_system.environments.env_package.doudizhu_grounding import (
    build_doudizhu_grounding_envs,
    doudizhu_grounding_projection,
)
from agent_system.environments.prompts.doudizhu import DOUDIZHU_VISUAL_TEMPLATE, DOUDIZHU_VISUAL_TEMPLATE_ZH
from agent_system.environments.prompts.doudizhu_grounding import (
    DOUDIZHU_GROUNDING_TEMPLATE,
    DOUDIZHU_GROUNDING_TEMPLATE_ZH,
)
from agent_system.multi_turn_rollout.utils import process_image


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

GROUNDING_STATE_METRICS = (
    "target_action_match",
    "click_valid_ratio",
    "submit_correct",
    "projection_valid",
    "reward",
    "tool_calling",
    "nonpass_target_action_match",
    "nonpass_selected_cards_match",
)

MEMORY_MULTIMODAL_TOKENS = (
    "<image>",
    "<video>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<|placeholder|>",
)


@dataclass(frozen=True)
class RequestMeta:
    env_index: int
    sample_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a model on doudizhu and/or doudizhu_grounding.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", "Qwen/Qwen3.5-4B"))
    parser.add_argument("--env", choices=("doudizhu", "doudizhu_grounding", "both"), default="both")
    parser.add_argument("--output-dir", default="outputs/doudizhu_model_eval")
    parser.add_argument("--num-episodes", type=int, default=128, help="Default model trajectory count for each selected env.")
    parser.add_argument("--doudizhu-num-episodes", type=int, default=None, help="Doudizhu independent model trajectory count.")
    parser.add_argument(
        "--grounding-num-episodes",
        type=int,
        default=None,
        help="Grounding model trajectory count; canonical teacher games equal this value divided by --grounding-samples-per-state.",
    )
    parser.add_argument("--num-envs", type=int, default=16, help="Concurrent canonical environments.")
    parser.add_argument("--grounding-samples-per-state", type=int, default=8, help="Number of grounding model trajectories sharing each canonical teacher state.")
    parser.add_argument("--max-env-steps", type=int, default=30)
    parser.add_argument("--env-seed-start", type=int, default=100000)
    parser.add_argument("--teacher-policy", default="rule_v1")
    parser.add_argument("--language", choices=("zh", "en"), default="zh")
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--max-clicks", type=int, default=20)

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--max-response-length", type=int, default=512)
    parser.add_argument("--use-vllm-n", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0, help="vLLM sampling seed and bootstrap seed.")

    parser.add_argument("--data-parallel-size", type=int, default=int(os.environ.get("DATA_PARALLEL_SIZE", "1")))
    parser.add_argument("--data-parallel-backend", choices=("mp",), default=os.environ.get("DATA_PARALLEL_BACKEND", "mp"))
    parser.add_argument("--tensor-model-parallel-size", type=int, default=int(os.environ.get("ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE", "1")))
    parser.add_argument("--gpu-memory-utilization", type=float, default=float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.6")))
    parser.add_argument("--max-num-batched-tokens", type=int, default=int(os.environ.get("MAX_NUM_BATCHED_TOKENS", "8192")))
    parser.add_argument("--max-num-seqs", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable-chunked-prefill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mm-processor-cache-gb", type=float, default=0.0)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--flush-every", type=int, default=1)
    return parser.parse_args()


def import_vllm():
    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError(
            "vLLM is not importable in this Python environment. Activate the Qwen3.5/vLLM environment, "
            "e.g. `conda activate verl-agent-bw-exp`."
        ) from exc
    return LLM, SamplingParams


def import_tokenizer(model_path: str, trust_remote_code: bool):
    from transformers import AutoProcessor, AutoTokenizer

    processor = None
    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    except Exception:
        processor = None
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    return tokenizer


def supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in sig.parameters}


def build_single_llm(args: argparse.Namespace):
    LLM, _SamplingParams = import_vllm()
    engine_kwargs = {
        "model": args.model_path,
        "tensor_parallel_size": args.tensor_model_parallel_size,
        "dtype": args.dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "max_model_len": args.max_model_len,
        "trust_remote_code": args.trust_remote_code,
        "enforce_eager": args.enforce_eager,
        "enable_chunked_prefill": args.enable_chunked_prefill,
        "enable_prefix_caching": args.enable_prefix_caching,
        "mm_processor_cache_gb": args.mm_processor_cache_gb,
    }
    engine_kwargs = {key: value for key, value in engine_kwargs.items() if value is not None}
    engine_kwargs = supported_kwargs(LLM, engine_kwargs)
    return LLM(**engine_kwargs)


def _output_texts(outputs: Sequence[Any]) -> list[list[str]]:
    return [[candidate.text.strip() for candidate in output.outputs] for output in outputs]


def _worker_visible_devices(rank: int, data_parallel_size: int, tensor_parallel_size: int) -> str:
    total_slots = data_parallel_size * tensor_parallel_size
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        devices = [device.strip() for device in visible.split(",") if device.strip()]
    else:
        devices = [str(idx) for idx in range(total_slots)]
    if len(devices) < total_slots:
        raise ValueError(
            f"Need at least {total_slots} visible CUDA devices for data_parallel_size={data_parallel_size}, "
            f"tensor_model_parallel_size={tensor_parallel_size}; got CUDA_VISIBLE_DEVICES={visible!r}."
        )
    start = rank * tensor_parallel_size
    return ",".join(devices[start : start + tensor_parallel_size])


def _vllm_worker_main(rank: int, args_dict: dict[str, Any], input_queue: mp.Queue, output_queue: mp.Queue, visible_devices: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    args = argparse.Namespace(**args_dict)
    args.data_parallel_size = 1
    try:
        llm = build_single_llm(args)
        while True:
            item = input_queue.get()
            if item is None:
                break
            task_id, requests, sampling_n = item
            try:
                sampling_params = build_sampling_params(args, n=sampling_n)
                outputs = llm.generate(requests, sampling_params=sampling_params, use_tqdm=False)
                output_queue.put((task_id, _output_texts(outputs), None))
            except Exception:
                output_queue.put((task_id, None, traceback.format_exc()))
    except Exception:
        output_queue.put(("__init__", None, traceback.format_exc()))


class LocalVLLMRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.llm = build_single_llm(args)

    def generate_texts(self, requests: list[dict[str, Any]], sampling_n: int) -> list[list[str]]:
        sampling_params = build_sampling_params(self.args, n=sampling_n)
        outputs = self.llm.generate(requests, sampling_params=sampling_params, use_tqdm=False)
        return _output_texts(outputs)

    def close(self) -> None:
        return None


class MultiProcessVLLMRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.ctx = mp.get_context("spawn")
        self.output_queue = self.ctx.Queue()
        self.input_queues: list[mp.Queue] = []
        self.processes: list[mp.Process] = []
        self.task_id = 0
        args_dict = vars(args).copy()

        for rank in range(args.data_parallel_size):
            visible_devices = _worker_visible_devices(rank, args.data_parallel_size, args.tensor_model_parallel_size)
            input_queue = self.ctx.Queue()
            process = self.ctx.Process(
                target=_vllm_worker_main,
                args=(rank, args_dict, input_queue, self.output_queue, visible_devices),
            )
            process.start()
            self.input_queues.append(input_queue)
            self.processes.append(process)

    def generate_texts(self, requests: list[dict[str, Any]], sampling_n: int) -> list[list[str]]:
        if not requests:
            return []

        chunks: list[list[dict[str, Any]]] = [[] for _ in self.processes]
        index_maps: list[list[int]] = [[] for _ in self.processes]
        for request_index, request in enumerate(requests):
            worker_index = request_index % len(self.processes)
            chunks[worker_index].append(request)
            index_maps[worker_index].append(request_index)

        pending: dict[int, list[int]] = {}
        for worker_index, chunk in enumerate(chunks):
            if not chunk:
                continue
            self.task_id += 1
            task_id = self.task_id
            pending[task_id] = index_maps[worker_index]
            self.input_queues[worker_index].put((task_id, chunk, sampling_n))

        results: list[list[str] | None] = [None for _ in requests]
        while pending:
            task_id, texts, error = self.output_queue.get()
            if error:
                raise RuntimeError(f"vLLM worker task {task_id} failed:\n{error}")
            if task_id == "__init__":
                raise RuntimeError(f"vLLM worker failed during initialization:\n{error}")
            indices = pending.pop(task_id)
            if len(indices) != len(texts):
                raise RuntimeError(f"Worker task {task_id} returned {len(texts)} outputs for {len(indices)} requests.")
            for request_index, output_texts in zip(indices, texts, strict=True):
                results[request_index] = output_texts

        return [texts if texts is not None else [] for texts in results]

    def close(self) -> None:
        for input_queue in self.input_queues:
            input_queue.put(None)
        for process in self.processes:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)


def build_llm_runner(args: argparse.Namespace):
    if args.data_parallel_size == 1:
        return LocalVLLMRunner(args)
    return MultiProcessVLLMRunner(args)


def build_sampling_params(args: argparse.Namespace, n: int):
    _LLM, SamplingParams = import_vllm()
    kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_response_length,
        "n": n,
        "seed": args.seed,
    }
    kwargs = supported_kwargs(SamplingParams, kwargs)
    return SamplingParams(**kwargs)


def apply_chat_template(tokenizer: Any, prompt: str, enable_thinking: bool) -> str:
    chat = [{"role": "user", "content": prompt}]
    kwargs = {
        "add_generation_prompt": True,
        "tokenize": False,
        "enable_thinking": bool(enable_thinking),
    }
    try:
        text = tokenizer.apply_chat_template(chat, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        text = tokenizer.apply_chat_template(chat, **kwargs)

    text = text.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
    if "<|image_pad|>" not in text:
        text = "<|vision_start|><|image_pad|><|vision_end|>\n" + text
    return text


def json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


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


def sanitize_memory(memory: Any) -> str:
    if not isinstance(memory, str):
        return ""
    for token in MEMORY_MULTIMODAL_TOKENS:
        memory = memory.replace(token, "")
    return memory.strip()


def make_env_config(args: argparse.Namespace) -> dict[str, Any]:
    chinese_mode = args.language == "zh"
    return {
        "doudizhu": {
            "use_ray": False,
            "language": args.language,
            "chinese_mode": chinese_mode,
            "image_width": args.image_width,
            "image_height": args.image_height,
            "max_clicks": args.max_clicks,
        },
        "doudizhu_grounding": {
            "use_ray": False,
            "teacher_policy": args.teacher_policy,
        },
    }


def empty_action(raw_text: Any = "") -> dict[str, Any]:
    return {
        "clicks": [],
        "plan": "",
        "semantic_action": "",
        "chat": "",
        "memory": "",
        "raw_action_text": "",
        "raw_tool_call_text": "",
        "tool_calls": [],
        "tool_calling": 0,
        "raw_response": raw_text if isinstance(raw_text, str) else "",
        "projection_valid": 0,
    }


def command_from_info(info: dict[str, Any], language: str) -> str:
    return (
        info.get("next_target_action_pretty")
        or info.get("target_action_pretty")
        or info.get("target_action")
        or ("pass" if language == "en" else "不要")
    )


def canonical_index(env_index: int, group_n: int) -> int:
    return env_index * group_n


def build_request(tokenizer: Any, prompt: str, image: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    prompt_text = apply_chat_template(tokenizer, prompt, args.enable_thinking)
    return {
        "prompt": prompt_text,
        "multi_modal_data": {"image": [process_image(image)]},
    }


def generate_grouped(
    llm_runner: Any,
    requests: list[dict[str, Any]],
    metas: list[RequestMeta],
    sampling_n: int,
    use_vllm_n: bool,
) -> dict[int, list[str]]:
    responses: dict[int, list[str]] = {meta.env_index: [] for meta in metas}
    if not requests:
        return responses
    outputs = llm_runner.generate_texts(requests, sampling_n=sampling_n if use_vllm_n else 1)
    for texts, meta in zip(outputs, metas, strict=True):
        responses.setdefault(meta.env_index, []).extend(texts)
    for meta in metas:
        if len(responses.get(meta.env_index, [])) != meta.sample_count:
            raise RuntimeError(
                f"Expected {meta.sample_count} samples for env_index={meta.env_index}, "
                f"got {len(responses.get(meta.env_index, []))}."
            )
    return responses


def project_doudizhu_response(response: str, max_clicks: int) -> dict[str, Any]:
    actions, valids = doudizhu_projection([response], max_clicks=max_clicks)
    action = actions[0]
    action["projection_valid"] = int(valids[0])
    return action


def project_grounding_response(response: str, max_clicks: int) -> dict[str, Any]:
    actions, valids = doudizhu_grounding_projection([response], max_clicks=max_clicks)
    action = actions[0]
    action["projection_valid"] = int(valids[0])
    return action


def mean_float(rows: Iterable[dict[str, Any]], key: str) -> float:
    vals = [float(row.get(key, 0.0)) for row in rows]
    vals = [val for val in vals if math.isfinite(val)]
    return float(np.mean(vals)) if vals else math.nan


def is_pass_action(action: Any) -> bool:
    return str(action).strip().lower() in ("pass", "不要")


def normalize_cards(text: Any) -> str:
    if text is None:
        return ""
    return str(text).replace(" ", "").strip()


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


def metric_summary(env_name: str, unit_name: str, rows: Sequence[dict[str, Any]], metrics: Sequence[str], args: argparse.Namespace) -> list[dict[str, Any]]:
    rng = np.random.default_rng(args.seed)
    out = []
    for metric in metrics:
        values = [float(row.get(metric, 0.0)) for row in rows]
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


def selected_envs(args: argparse.Namespace) -> list[str]:
    if args.env == "both":
        return ["doudizhu", "doudizhu_grounding"]
    return [args.env]


def episode_count_for(env_name: str, args: argparse.Namespace) -> int:
    if env_name == "doudizhu":
        return int(args.doudizhu_num_episodes or args.num_episodes)
    if env_name == "doudizhu_grounding":
        trajectory_count = int(args.grounding_num_episodes or args.num_episodes)
        return trajectory_count // int(args.grounding_samples_per_state)
    raise ValueError(f"Unknown env_name: {env_name}")


def trajectory_count_for(env_name: str, args: argparse.Namespace) -> int:
    if env_name == "doudizhu":
        return int(args.doudizhu_num_episodes or args.num_episodes)
    if env_name == "doudizhu_grounding":
        return int(args.grounding_num_episodes or args.num_episodes)
    raise ValueError(f"Unknown env_name: {env_name}")


def initial_memory(language: str) -> str:
    if language == "zh":
        return "初始回合。阅读截图，识别你的手牌，并规划地主首轮出牌。"
    return "Initial turn. Read the screenshot, identify your hand, and plan the first landlord play."


def doudizhu_prompt(language: str, memory: str) -> str:
    template = DOUDIZHU_VISUAL_TEMPLATE_ZH if language == "zh" else DOUDIZHU_VISUAL_TEMPLATE
    return template.format(previous_memory=memory)


def grounding_prompt(language: str, command: str) -> str:
    template = DOUDIZHU_GROUNDING_TEMPLATE_ZH if language == "zh" else DOUDIZHU_GROUNDING_TEMPLATE
    return template.format(command=command)


def run_doudizhu_wave(
    llm_runner: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    wave_index: int,
    episode_seeds: Sequence[int],
    sample_handle,
    episode_handle,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    env_count = len(episode_seeds)
    env = build_doudizhu_envs(
        seed=args.env_seed_start + wave_index * args.num_envs,
        env_num=env_count,
        group_n=1,
        is_train=False,
        env_config=make_env_config(args),
        resources_per_worker={"num_cpus": 0.1, "num_gpus": 0},
    )
    episode_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []

    try:
        images, infos = env.reset(kwargs=[{"seed": int(seed)} for seed in episode_seeds])
        active = [True for _ in range(env_count)]
        memories = [initial_memory(args.language) for _ in range(env_count)]
        totals = [
            {
                "total_reward": 0.0,
                "episode_length": 0,
                "projection_valid_sum": 0.0,
                "click_valid_sum": 0.0,
                "rule_action_valid_sum": 0.0,
                "fallback_sum": 0.0,
                "hand_depletion_sum": 0.0,
                "model_hand_depletion_sum": 0.0,
                "fallback_pass_sum": 0.0,
                "tool_calling_sum": 0.0,
                "last_info": {},
                "raw_responses": [],
            }
            for _ in range(env_count)
        ]

        for step_index in range(args.max_env_steps):
            if not any(active):
                break

            requests: list[dict[str, Any]] = []
            metas: list[RequestMeta] = []
            for env_index in range(env_count):
                if not active[env_index]:
                    continue
                prompt = doudizhu_prompt(args.language, memories[env_index])
                requests.append(build_request(tokenizer, prompt, images[env_index], args))
                metas.append(RequestMeta(env_index=env_index, sample_count=1))

            responses = generate_grouped(
                llm_runner=llm_runner,
                requests=requests,
                metas=metas,
                sampling_n=1,
                use_vllm_n=True,
            )

            actions: list[dict[str, Any]] = []
            raw_responses = ["" for _ in range(env_count)]
            for env_index in range(env_count):
                if active[env_index]:
                    response = responses.get(env_index, [""])[0]
                    raw_responses[env_index] = response
                    actions.append(project_doudizhu_response(response, args.max_clicks))
                else:
                    actions.append(empty_action())

            next_images, rewards, dones, step_infos = env.step(actions)
            for env_index in range(env_count):
                if not active[env_index]:
                    continue
                info = dict(step_infos[env_index])
                action = actions[env_index]
                reward = float(rewards[env_index])
                totals[env_index]["total_reward"] += reward
                totals[env_index]["episode_length"] += 1
                totals[env_index]["projection_valid_sum"] += float(info.get("projection_valid", action.get("projection_valid", 0.0)))
                totals[env_index]["click_valid_sum"] += float(info.get("click_valid_ratio", 0.0))
                totals[env_index]["rule_action_valid_sum"] += float(info.get("rule_action_valid", 0.0))
                fallback_used = bool(info.get("fallback_used", 0.0))
                hand_cards_reduced = float(info.get("hand_cards_reduced", 0.0))
                totals[env_index]["fallback_sum"] += float(fallback_used)
                totals[env_index]["hand_depletion_sum"] += hand_cards_reduced
                if not fallback_used:
                    totals[env_index]["model_hand_depletion_sum"] += hand_cards_reduced
                if fallback_used and is_pass_action(info.get("game_action")):
                    totals[env_index]["fallback_pass_sum"] += 1.0
                totals[env_index]["tool_calling_sum"] += float(info.get("tool_calling", 0.0))
                totals[env_index]["last_info"] = info
                totals[env_index]["raw_responses"].append(raw_responses[env_index])

                memory = sanitize_memory(action.get("memory", ""))
                if memory:
                    memories[env_index] = memory[:512]

                row = {
                    "env_name": "doudizhu",
                    "wave_index": wave_index,
                    "episode_seed": int(episode_seeds[env_index]),
                    "env_index": env_index,
                    "step_index": step_index,
                    "raw_response": raw_responses[env_index],
                    "raw_action_text": action.get("raw_action_text", ""),
                    "raw_tool_call_text": action.get("raw_tool_call_text", ""),
                    "semantic_action": action.get("semantic_action", ""),
                    "game_action": info.get("game_action"),
                    "selected_cards": info.get("selected_cards", ""),
                    "submit_kind": info.get("submit_kind"),
                    "projection_valid": float(info.get("projection_valid", action.get("projection_valid", 0.0))),
                    "click_valid_ratio": float(info.get("click_valid_ratio", 0.0)),
                    "rule_action_valid": float(info.get("rule_action_valid", 0.0)),
                    "fallback_used": float(fallback_used),
                    "hand_cards_reduced": hand_cards_reduced,
                    "model_hand_cards_reduced": hand_cards_reduced if not fallback_used else 0.0,
                    "fallback_pass": float(fallback_used and is_pass_action(info.get("game_action"))),
                    "reward": reward,
                    "done": bool(dones[env_index]),
                    "won": float(info.get("won", 0.0)),
                    "tool_calling": float(info.get("tool_calling", 0.0)),
                    "info": info,
                }
                write_jsonl(sample_handle, row)
                sample_rows.append(row)

            active = [active[idx] and not bool(dones[idx]) for idx in range(env_count)]
            images, infos = next_images, step_infos
            if args.flush_every > 0 and (step_index + 1) % args.flush_every == 0:
                sample_handle.flush()

        for env_index in range(env_count):
            total = totals[env_index]
            length = int(total["episode_length"])
            last_info = dict(total["last_info"])
            truncated = bool(length >= args.max_env_steps and last_info.get("winner_id") is None)
            denom = max(length, 1)
            projection_valid_count = float(total["projection_valid_sum"])
            episode_row = {
                "env_name": "doudizhu",
                "wave_index": wave_index,
                "episode_seed": int(episode_seeds[env_index]),
                "env_index": env_index,
                "won": float(last_info.get("won", 0.0)),
                "total_reward": float(total["total_reward"]),
                "episode_length": length,
                "projection_valid_rate": float(total["projection_valid_sum"]) / denom,
                "click_valid_ratio": float(total["click_valid_sum"]) / denom,
                "rule_action_valid_rate": float(total["rule_action_valid_sum"]) / denom,
                "fallback_rate": float(total["fallback_sum"]) / denom,
                "hand_depletion_rate": float(total["hand_depletion_sum"]) / 20.0,
                "model_hand_depletion_rate": float(total["model_hand_depletion_sum"]) / 20.0,
                "legal_given_projection_rate": float(total["rule_action_valid_sum"]) / projection_valid_count if projection_valid_count > 0 else 0.0,
                "fallback_pass_rate": float(total["fallback_pass_sum"]) / denom,
                "tool_calling": float(total["tool_calling_sum"]),
                "truncated": float(truncated),
                "winner_id": last_info.get("winner_id"),
                "payoffs": last_info.get("payoffs"),
                "final_state_summary": last_info.get("state_summary"),
            }
            write_jsonl(episode_handle, episode_row)
            episode_rows.append(episode_row)
    finally:
        env.close()

    return sample_rows, episode_rows


def run_grounding_wave(
    llm_runner: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    wave_index: int,
    episode_seeds: Sequence[int],
    sample_handle,
    episode_handle,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    env_count = len(episode_seeds)
    group_n = int(args.grounding_samples_per_state)
    env = build_doudizhu_grounding_envs(
        seed=args.env_seed_start + 1_000_000 + wave_index * args.num_envs,
        env_num=env_count,
        group_n=group_n,
        is_train=False,
        env_config=make_env_config(args),
        resources_per_worker={"num_cpus": 0.1, "num_gpus": 0},
    )
    sample_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []

    try:
        images, infos = env.reset(kwargs=[{"seed": int(seed)} for seed in episode_seeds])
        active = [True for _ in range(env_count)]
        trajectory_acc = [
            [
                {
                    "num_states": 0,
                    **{f"{metric}_sum": 0.0 for metric in GROUNDING_STATE_METRICS},
                    **{f"{metric}_count": 0 for metric in GROUNDING_STATE_METRICS},
                    "last_info": {},
                }
                for _ in range(group_n)
            ]
            for _ in range(env_count)
        ]

        for step_index in range(args.max_env_steps):
            if not any(active):
                break

            requests: list[dict[str, Any]] = []
            metas: list[RequestMeta] = []
            pre_infos = [dict(info) for info in infos]
            for env_index in range(env_count):
                if not active[env_index]:
                    continue
                base = canonical_index(env_index, group_n)
                command = command_from_info(pre_infos[base], args.language)
                prompt = grounding_prompt(args.language, command)
                if args.use_vllm_n:
                    requests.append(build_request(tokenizer, prompt, images[base], args))
                    metas.append(RequestMeta(env_index=env_index, sample_count=group_n))
                else:
                    for _ in range(group_n):
                        requests.append(build_request(tokenizer, prompt, images[base], args))
                        metas.append(RequestMeta(env_index=env_index, sample_count=group_n))

            responses = generate_grouped(
                llm_runner=llm_runner,
                requests=requests,
                metas=metas,
                sampling_n=group_n,
                use_vllm_n=args.use_vllm_n,
            )

            actions: list[dict[str, Any]] = []
            for env_index in range(env_count):
                env_responses = responses.get(env_index, [])
                for sample_index in range(group_n):
                    if active[env_index] and sample_index < len(env_responses):
                        actions.append(project_grounding_response(env_responses[sample_index], args.max_clicks))
                    else:
                        actions.append(empty_action())

            next_images, _rewards, dones, step_infos = env.step(actions)

            for env_index in range(env_count):
                if not active[env_index]:
                    continue
                base = canonical_index(env_index, group_n)
                state_uid = f"grounding:seed:{int(episode_seeds[env_index])}:step:{step_index}"
                env_responses = responses.get(env_index, [])
                per_sample_rows = []
                canonical_episode_index = wave_index * args.num_envs + env_index
                trajectory_start_index = canonical_episode_index * group_n
                for sample_index in range(group_n):
                    idx = base + sample_index
                    info = dict(step_infos[idx])
                    response = env_responses[sample_index] if sample_index < len(env_responses) else ""
                    action = actions[idx]
                    trajectory_index = trajectory_start_index + sample_index
                    target_action = info.get("target_action")
                    nonpass_target = not is_pass_action(target_action)
                    selected_cards_match = float(
                        nonpass_target and normalize_cards(info.get("selected_cards")) == normalize_cards(target_action)
                    )
                    row = {
                        "env_name": "doudizhu_grounding",
                        "wave_index": wave_index,
                        "episode_seed": int(episode_seeds[env_index]),
                        "env_index": env_index,
                        "canonical_episode_index": canonical_episode_index,
                        "trajectory_index": trajectory_index,
                        "state_uid": state_uid,
                        "step_index": step_index,
                        "sample_index": sample_index,
                        "target_action": target_action,
                        "target_action_pretty": info.get("target_action_pretty") or pre_infos[base].get("target_action_pretty"),
                        "nonpass_target": float(nonpass_target),
                        "predicted_action": info.get("predicted_action"),
                        "selected_cards": info.get("selected_cards"),
                        "selected_cards_match": selected_cards_match,
                        "selected_indices": info.get("selected_indices"),
                        "submit_kind": info.get("submit_kind"),
                        "target_action_match": float(info.get("target_action_match", 0.0)),
                        "click_valid_ratio": float(info.get("click_valid_ratio", 0.0)),
                        "submit_correct": float(info.get("submit_correct", 0.0)),
                        "projection_valid": float(info.get("projection_valid", action.get("projection_valid", 0.0))),
                        "reward": float(info.get("reward", 0.0)),
                        "tool_calling": float(info.get("tool_calling", 0.0)),
                        "nonpass_target_action_match": float(info.get("target_action_match", 0.0)) if nonpass_target else math.nan,
                        "nonpass_selected_cards_match": selected_cards_match if nonpass_target else math.nan,
                        "raw_tool_call_text": info.get("raw_tool_call_text") or extract_tag(response, "tool_call") or "",
                        "raw_response": response,
                        "done": bool(dones[idx]),
                        "info": info,
                    }
                    write_jsonl(sample_handle, row)
                    sample_rows.append(row)
                    per_sample_rows.append(row)

                    acc = trajectory_acc[env_index][sample_index]
                    acc["num_states"] += 1
                    for metric in GROUNDING_STATE_METRICS:
                        metric_value = float(row[metric])
                        if math.isfinite(metric_value):
                            acc[f"{metric}_sum"] += metric_value
                            acc[f"{metric}_count"] += 1
                    acc["last_info"] = info

                state_row = {
                    "env_name": "doudizhu_grounding",
                    "wave_index": wave_index,
                    "episode_seed": int(episode_seeds[env_index]),
                    "env_index": env_index,
                    "canonical_episode_index": canonical_episode_index,
                    "trajectory_start_index": trajectory_start_index,
                    "state_uid": state_uid,
                    "step_index": step_index,
                    "num_samples": group_n,
                    "num_trajectories": group_n,
                    "target_action": per_sample_rows[0].get("target_action") if per_sample_rows else None,
                    "target_action_pretty": per_sample_rows[0].get("target_action_pretty") if per_sample_rows else None,
                }
                for metric in GROUNDING_STATE_METRICS:
                    state_row[metric] = mean_float(per_sample_rows, metric)
                state_rows.append(state_row)

            active = [active[idx] and not bool(dones[canonical_index(idx, group_n)]) for idx in range(env_count)]
            images, infos = next_images, step_infos
            if args.flush_every > 0 and (step_index + 1) % args.flush_every == 0:
                sample_handle.flush()

        for env_index in range(env_count):
            canonical_episode_index = wave_index * args.num_envs + env_index
            trajectory_start_index = canonical_episode_index * group_n
            for sample_index in range(group_n):
                acc = trajectory_acc[env_index][sample_index]
                num_states = int(acc["num_states"])
                denom = max(num_states, 1)
                last_info = dict(acc["last_info"])
                truncated = bool(num_states >= args.max_env_steps and last_info.get("winner_id") is None)
                episode_row = {
                    "env_name": "doudizhu_grounding",
                    "wave_index": wave_index,
                    "episode_seed": int(episode_seeds[env_index]),
                    "env_index": env_index,
                    "canonical_episode_index": canonical_episode_index,
                    "trajectory_index": trajectory_start_index + sample_index,
                    "sample_index": sample_index,
                    "num_states": num_states,
                    "episode_length": num_states,
                    "truncated": float(truncated),
                    "winner_id": last_info.get("winner_id"),
                    "final_state_summary": last_info.get("state_summary"),
                }
                for metric in GROUNDING_STATE_METRICS:
                    metric_count = int(acc[f"{metric}_count"])
                    episode_row[metric] = (
                        float(acc[f"{metric}_sum"]) / metric_count
                        if metric_count > 0
                        else math.nan
                    )
                write_jsonl(episode_handle, episode_row)
                episode_rows.append(episode_row)
    finally:
        env.close()

    return sample_rows, state_rows, episode_rows


def run_env(
    env_name: str,
    llm_runner: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    sample_handle,
    episode_handle,
) -> dict[str, list[dict[str, Any]]]:
    num_canonical_episodes = episode_count_for(env_name, args)
    num_model_trajectories = trajectory_count_for(env_name, args)
    remaining = num_canonical_episodes
    next_seed = args.env_seed_start + (0 if env_name == "doudizhu" else 1_000_000)
    wave_index = 0
    all_samples: list[dict[str, Any]] = []
    all_states: list[dict[str, Any]] = []
    all_episodes: list[dict[str, Any]] = []

    while remaining > 0:
        env_count = min(args.num_envs, remaining)
        episode_seeds = list(range(next_seed, next_seed + env_count))
        print(
            f"[{env_name} wave {wave_index}] canonical_envs={env_count} seeds={episode_seeds[0]}..{episode_seeds[-1]} "
            f"model_trajectory_budget={num_model_trajectories}",
            flush=True,
        )
        if env_name == "doudizhu":
            sample_rows, episode_rows = run_doudizhu_wave(
                llm_runner=llm_runner,
                tokenizer=tokenizer,
                args=args,
                wave_index=wave_index,
                episode_seeds=episode_seeds,
                sample_handle=sample_handle,
                episode_handle=episode_handle,
            )
            all_samples.extend(sample_rows)
            all_episodes.extend(episode_rows)
        elif env_name == "doudizhu_grounding":
            sample_rows, state_rows, episode_rows = run_grounding_wave(
                llm_runner=llm_runner,
                tokenizer=tokenizer,
                args=args,
                wave_index=wave_index,
                episode_seeds=episode_seeds,
                sample_handle=sample_handle,
                episode_handle=episode_handle,
            )
            all_samples.extend(sample_rows)
            all_states.extend(state_rows)
            all_episodes.extend(episode_rows)
        else:
            raise ValueError(f"Unsupported env_name: {env_name}")

        remaining -= env_count
        next_seed += env_count
        wave_index += 1

    return {
        "samples": all_samples,
        "states": all_states,
        "episodes": all_episodes,
    }


def build_summary(results: dict[str, dict[str, list[dict[str, Any]]]], args: argparse.Namespace) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    if "doudizhu" in results:
        summary.extend(metric_summary("doudizhu", "trajectory", results["doudizhu"]["episodes"], DOUDIZHU_METRICS, args))
    if "doudizhu_grounding" in results:
        summary.extend(
            metric_summary(
                "doudizhu_grounding",
                "state",
                results["doudizhu_grounding"]["states"],
                GROUNDING_STATE_METRICS,
                args,
            )
        )
        summary.extend(
            metric_summary(
                "doudizhu_grounding",
                "episode",
                results["doudizhu_grounding"]["episodes"],
                (
                    "target_action_match",
                    "click_valid_ratio",
                    "submit_correct",
                    "projection_valid",
                    "reward",
                    "nonpass_target_action_match",
                    "nonpass_selected_cards_match",
                    "truncated",
                ),
                args,
            )
        )
    return summary


def validate_args(args: argparse.Namespace) -> None:
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive")
    if args.num_episodes <= 0:
        raise ValueError("--num-episodes must be positive")
    if args.doudizhu_num_episodes is not None and args.doudizhu_num_episodes <= 0:
        raise ValueError("--doudizhu-num-episodes must be positive")
    if args.grounding_num_episodes is not None and args.grounding_num_episodes <= 0:
        raise ValueError("--grounding-num-episodes must be positive")
    if args.grounding_samples_per_state <= 0:
        raise ValueError("--grounding-samples-per-state must be positive")
    if "doudizhu_grounding" in selected_envs(args):
        grounding_trajectories = int(args.grounding_num_episodes or args.num_episodes)
        if grounding_trajectories % int(args.grounding_samples_per_state) != 0:
            raise ValueError(
                "--grounding-num-episodes/--num-episodes must be divisible by --grounding-samples-per-state "
                "so grounding trajectory budget maps cleanly to canonical teacher games."
            )
    if args.data_parallel_size <= 0 or args.tensor_model_parallel_size <= 0:
        raise ValueError("--data-parallel-size and --tensor-model-parallel-size must be positive")


def main() -> None:
    args = parse_args()
    validate_args(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )

    tokenizer = import_tokenizer(args.model_path, args.trust_remote_code)
    print(
        f"vLLM parallelism: data_parallel_size={args.data_parallel_size}, "
        f"tensor_model_parallel_size={args.tensor_model_parallel_size}, "
        f"total_gpu_slots={args.data_parallel_size * args.tensor_model_parallel_size}",
        flush=True,
    )
    llm_runner = build_llm_runner(args)

    sample_jsonl = output_dir / "samples.jsonl"
    episode_jsonl = output_dir / "episodes.jsonl"
    results: dict[str, dict[str, list[dict[str, Any]]]] = {}
    try:
        with sample_jsonl.open("w", encoding="utf-8") as sample_handle, episode_jsonl.open("w", encoding="utf-8") as episode_handle:
            for env_name in selected_envs(args):
                results[env_name] = run_env(
                    env_name=env_name,
                    llm_runner=llm_runner,
                    tokenizer=tokenizer,
                    args=args,
                    sample_handle=sample_handle,
                    episode_handle=episode_handle,
                )
    finally:
        llm_runner.close()

    state_rows = results.get("doudizhu_grounding", {}).get("states", [])
    summary_rows = build_summary(results, args)
    write_csv(output_dir / "grounding_state_metrics.csv", state_rows)
    write_csv(output_dir / "episode_metrics.csv", [row for result in results.values() for row in result["episodes"]])
    write_csv(output_dir / "summary.csv", summary_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary_rows, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )

    print(json.dumps(summary_rows, indent=2, ensure_ascii=False, default=json_default), flush=True)
    print(f"Wrote samples to {sample_jsonl}", flush=True)
    print(f"Wrote episodes to {episode_jsonl}", flush=True)
    print(f"Wrote summary to {output_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
