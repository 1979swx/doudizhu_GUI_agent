#!/usr/bin/env python3
"""Evaluate whether requiring <plan> changes doudizhu_grounding click accuracy.

This script intentionally does not modify or depend on training entry points.  It
uses the canonical doudizhu grounding environment directly: model generations are
only parsed and scored, while the environment is advanced by the teacher policy.
For each canonical game state it samples M plan responses and M no_plan
responses, then scores all 2M samples against the same screenshot and command.
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

from agent_system.environments.env_package.doudizhu.projection import (
    extract_tag,
    parse_left_click_tool_call,
)
from agent_system.environments.env_package.doudizhu_grounding import (
    build_doudizhu_grounding_envs,
    doudizhu_grounding_projection,
)
from agent_system.environments.prompts.doudizhu_grounding import (
    DOUDIZHU_GROUNDING_TEMPLATE,
    DOUDIZHU_GROUNDING_TEMPLATE_ZH,
)
from agent_system.multi_turn_rollout.utils import process_image

NO_PLAN_TEMPLATE = """<image>
You are controlling the Dou Dizhu GUI by normalized left clicks.

Commanded card action: {command}

Your task is only to execute the commanded action on the screenshot. Do not choose a different card action.
- If the commanded action is pass, click only the PASS button.
- Otherwise, click each matching card in your bottom hand, then click the PLAY button.
- Coordinates must be normalized numbers from 0 to 1000, where [0,0] is the top-left corner and [1000,1000] is the bottom-right corner.
- Output one turn only.

Return exactly one XML tag and do not output <plan> or any explanation:
<tool_call>left_click([x1,y1],[x2,y2])</tool_call>
"""


NO_PLAN_TEMPLATE_ZH = """
你正在通过鼠标点击来控制斗地主 GUI。

指挥动作：{command}

<image>你的任务是在截图中执行这个指挥动作。不要自行选择其它出牌。
- 你通过 [x,y] 坐标来进行点击动作，坐标必须是 0 到 1000 范围内的归一化数字，[0,0] 代表左上角，[1000,1000] 代表右下角。
- 游戏页面的底部有手牌，其上方有‘出牌’和‘不要’按钮，这是主要交互区域。
- 如果指挥动作是“不要”或 pass，只点击“不要”按钮。
- 如果指挥动作是出牌，则依次点击底部手牌中与指挥动作匹配的每张牌，然后点击“出牌”按钮。
- 每一轮的动作必须以点击“出牌”或“不要”两个按钮之一结尾。

只输出一个名为 <tool_call> </tool_call> 的 XML 标签，标签内是一个 left_click([x1,y1],[x2,y2],...,[xN,yN]) 调用。
不要输出 <plan>，不要输出解释或其它标签。

示例：
指挥动作：不要
输出：<tool_call>left_click([566,764])</tool_call>

指挥动作：3 3
输出：<tool_call>left_click([55,850],[100,860],[430,755])</tool_call>

当前指挥动作：{command}
"""


MODES = ("plan", "no_plan")
METRIC_KEYS = (
    "target_action_match",
    "click_valid_ratio",
    "submit_correct",
    "projection_valid",
    "reward",
)


@dataclass(frozen=True)
class RequestMeta:
    wave_index: int
    env_index: int
    mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired plan/no_plan sampling eval for doudizhu_grounding.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", "Qwen/Qwen3.5-4B"))
    parser.add_argument("--output-dir", default="outputs/doudizhu_grounding_plan_ablation")
    parser.add_argument("--num-episodes", type=int, default=128)
    parser.add_argument("--num-envs", type=int, default=16, help="Concurrent canonical game environments.")
    parser.add_argument("--samples-per-mode", type=int, default=8, help="M samples for plan and M for no_plan per state.")
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
    parser.add_argument("--disable-nccl-for-dp-synchronization", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--bootstrap-unit", choices=("episode", "state"), default="episode")
    parser.add_argument("--flush-every", type=int, default=1, help="Flush JSONL every N env steps.")
    return parser.parse_args()


def import_vllm():
    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError("vLLM is not importable in this Python environment. Activate the Qwen3.5/vLLM environment, e.g. `conda activate verl-agent-bw-exp`.") from exc
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
        raise ValueError(f"Need at least {total_slots} visible CUDA devices for data_parallel_size={data_parallel_size}, tensor_model_parallel_size={tensor_parallel_size}; got CUDA_VISIBLE_DEVICES={visible!r}.")
    start = rank * tensor_parallel_size
    end = start + tensor_parallel_size
    return ",".join(devices[start:end])


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


def build_prompt(mode: str, command: str, language: str) -> str:
    if mode == "plan":
        template = DOUDIZHU_GROUNDING_TEMPLATE_ZH if language == "zh" else DOUDIZHU_GROUNDING_TEMPLATE
    elif mode == "no_plan":
        template = NO_PLAN_TEMPLATE_ZH if language == "zh" else NO_PLAN_TEMPLATE
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return template.format(command=command)


def relaxed_no_plan_projection(text_actions: list[str], max_clicks: int) -> tuple[list[dict[str, Any]], list[int]]:
    structured_actions: list[dict[str, Any]] = []
    valids: list[int] = []
    for response in text_actions:
        if not isinstance(response, str):
            structured_actions.append(empty_action(response))
            valids.append(0)
            continue

        tool_call_text = extract_tag(response, "tool_call")
        if tool_call_text:
            clicks, tool_calls, action_valid = parse_left_click_tool_call(tool_call_text, max_clicks=max_clicks)
        else:
            clicks, tool_calls, action_valid = [], [], False
        valid = bool(action_valid)
        structured_actions.append(
            {
                "clicks": clicks if valid else [],
                "plan": "",
                "raw_tool_call_text": tool_call_text or "",
                "tool_calls": tool_calls if valid else [],
                "tool_calling": len(tool_calls) if valid else 0,
                "raw_response": response,
                "projection_valid": int(valid),
            }
        )
        valids.append(int(valid))
    return structured_actions, valids


def empty_action(raw_text: Any = "") -> dict[str, Any]:
    return {
        "clicks": [],
        "plan": "",
        "raw_tool_call_text": "",
        "tool_calls": [],
        "tool_calling": 0,
        "raw_response": raw_text if isinstance(raw_text, str) else "",
        "projection_valid": 0,
    }


def project_one(mode: str, response: str, max_clicks: int) -> dict[str, Any]:
    if mode == "plan":
        actions, valids = doudizhu_grounding_projection([response], max_clicks=max_clicks)
    elif mode == "no_plan":
        actions, valids = relaxed_no_plan_projection([response], max_clicks=max_clicks)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    action = actions[0]
    action["projection_valid"] = int(valids[0])
    return action


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


def canonical_index(group_index: int, group_n: int) -> int:
    return group_index * group_n


def command_from_info(info: dict[str, Any], language: str) -> str:
    return info.get("next_target_action_pretty") or info.get("target_action_pretty") or info.get("target_action") or ("不要" if language == "zh" else "pass")


def generate_for_states(
    llm_runner: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    images: np.ndarray,
    infos: Sequence[dict[str, Any]],
    active: Sequence[bool],
    wave_index: int,
    env_count: int,
    group_n: int,
) -> dict[tuple[int, str], list[str]]:
    requests: list[dict[str, Any]] = []
    request_meta: list[RequestMeta] = []

    for env_index in range(env_count):
        if not active[env_index]:
            continue
        base = canonical_index(env_index, group_n)
        image = process_image(images[base])
        command = command_from_info(infos[base], args.language)
        for mode in MODES:
            prompt = build_prompt(mode, command, args.language)
            prompt_text = apply_chat_template(tokenizer, prompt, args.enable_thinking)
            if args.use_vllm_n:
                requests.append({"prompt": prompt_text, "multi_modal_data": {"image": [image]}})
                request_meta.append(RequestMeta(wave_index=wave_index, env_index=env_index, mode=mode))
            else:
                for _ in range(args.samples_per_mode):
                    requests.append({"prompt": prompt_text, "multi_modal_data": {"image": [image]}})
                    request_meta.append(RequestMeta(wave_index=wave_index, env_index=env_index, mode=mode))

    responses: dict[tuple[int, str], list[str]] = {(env_index, mode): [] for env_index in range(env_count) for mode in MODES}
    if not requests:
        return responses

    sampling_n = args.samples_per_mode if args.use_vllm_n else 1
    outputs = llm_runner.generate_texts(requests, sampling_n=sampling_n)

    for texts, meta in zip(outputs, request_meta, strict=True):
        responses[(meta.env_index, meta.mode)].extend(texts)

    expected = args.samples_per_mode
    for key, values in responses.items():
        if not active[key[0]]:
            continue
        if len(values) != expected:
            raise RuntimeError(f"Expected {expected} samples for {key}, got {len(values)}")
    return responses


def json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_sample_rows(
    handle,
    args: argparse.Namespace,
    episode_seeds: Sequence[int],
    wave_index: int,
    step_index: int,
    env_count: int,
    group_n: int,
    active: Sequence[bool],
    pre_infos: Sequence[dict[str, Any]],
    responses: dict[tuple[int, str], list[str]],
    post_infos: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for env_index in range(env_count):
        if not active[env_index]:
            continue
        base = canonical_index(env_index, group_n)
        state_uid = f"seed:{episode_seeds[env_index]}:step:{step_index}"
        for mode_index, mode in enumerate(MODES):
            for sample_index in range(args.samples_per_mode):
                group_offset = mode_index * args.samples_per_mode + sample_index
                info = post_infos[base + group_offset]
                response = responses.get((env_index, mode), [""] * args.samples_per_mode)[sample_index]
                row = {
                    "wave_index": wave_index,
                    "episode_seed": episode_seeds[env_index],
                    "env_index": env_index,
                    "state_uid": state_uid,
                    "grounding_step": step_index,
                    "mode": mode,
                    "sample_index": sample_index,
                    "target_action": info.get("target_action"),
                    "target_action_pretty": info.get("target_action_pretty") or pre_infos[base].get("target_action_pretty"),
                    "predicted_action": info.get("predicted_action"),
                    "selected_cards": info.get("selected_cards"),
                    "selected_indices": info.get("selected_indices"),
                    "submit_kind": info.get("submit_kind"),
                    "target_action_match": float(info.get("target_action_match", 0.0)),
                    "click_valid_ratio": float(info.get("click_valid_ratio", 0.0)),
                    "submit_correct": float(info.get("submit_correct", 0.0)),
                    "projection_valid": float(info.get("projection_valid", 0.0)),
                    "reward": float(info.get("reward", 0.0)),
                    "tool_calling": float(info.get("tool_calling", 0.0)),
                    "plan": info.get("plan", ""),
                    "raw_tool_call_text": info.get("raw_tool_call_text", ""),
                    "raw_response": response,
                }
                handle.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")
                rows.append(row)
    return rows


def build_actions(
    args: argparse.Namespace,
    responses: dict[tuple[int, str], list[str]],
    active: Sequence[bool],
    env_count: int,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for env_index in range(env_count):
        for mode in MODES:
            mode_responses = responses.get((env_index, mode), [])
            for sample_index in range(args.samples_per_mode):
                if active[env_index] and sample_index < len(mode_responses):
                    actions.append(project_one(mode, mode_responses[sample_index], max_clicks=args.max_clicks))
                else:
                    actions.append(empty_action())
    return actions


def aggregate_state_rows(sample_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    state_base: dict[str, dict[str, Any]] = {}
    for row in sample_rows:
        state_uid = str(row["state_uid"])
        mode = str(row["mode"])
        grouped.setdefault((state_uid, mode), []).append(row)
        state_base.setdefault(
            state_uid,
            {
                "state_uid": state_uid,
                "episode_seed": row["episode_seed"],
                "grounding_step": row["grounding_step"],
                "target_action": row["target_action"],
                "target_action_pretty": row["target_action_pretty"],
            },
        )

    state_rows: list[dict[str, Any]] = []
    for state_uid, base in sorted(state_base.items(), key=lambda item: (int(item[1]["episode_seed"]), int(item[1]["grounding_step"]))):
        out = dict(base)
        for mode in MODES:
            rows = grouped.get((state_uid, mode), [])
            for metric in METRIC_KEYS:
                out[f"{mode}_{metric}"] = float(np.mean([float(row[metric]) for row in rows])) if rows else math.nan
        for metric in METRIC_KEYS:
            out[f"delta_no_plan_minus_plan_{metric}"] = out[f"no_plan_{metric}"] - out[f"plan_{metric}"]
        state_rows.append(out)
    return state_rows


def bootstrap_ci(values: np.ndarray, iters: int, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, math.nan
    if len(values) == 1 or iters <= 0:
        return float(values[0]), float(values[0])
    samples = rng.choice(values, size=(iters, len(values)), replace=True)
    boot = samples.mean(axis=1)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def summarize(state_rows: Sequence[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    rng = np.random.default_rng(args.seed)
    summary: list[dict[str, Any]] = []
    if args.bootstrap_unit == "episode":
        units = sorted({int(row["episode_seed"]) for row in state_rows})
    else:
        units = sorted({str(row["state_uid"]) for row in state_rows})

    for metric in METRIC_KEYS:
        plan_values = np.array([float(row[f"plan_{metric}"]) for row in state_rows], dtype=np.float64)
        no_plan_values = np.array([float(row[f"no_plan_{metric}"]) for row in state_rows], dtype=np.float64)
        state_deltas = no_plan_values - plan_values

        unit_deltas = []
        for unit in units:
            if args.bootstrap_unit == "episode":
                rows = [row for row in state_rows if int(row["episode_seed"]) == unit]
            else:
                rows = [row for row in state_rows if str(row["state_uid"]) == unit]
            deltas = [float(row[f"delta_no_plan_minus_plan_{metric}"]) for row in rows]
            unit_deltas.append(float(np.mean(deltas)))
        ci_low, ci_high = bootstrap_ci(np.array(unit_deltas, dtype=np.float64), args.bootstrap_iters, rng)

        summary.append(
            {
                "metric": metric,
                "num_states": len(state_rows),
                "num_bootstrap_units": len(units),
                "bootstrap_unit": args.bootstrap_unit,
                "plan_mean_by_state": float(np.nanmean(plan_values)) if len(plan_values) else math.nan,
                "no_plan_mean_by_state": float(np.nanmean(no_plan_values)) if len(no_plan_values) else math.nan,
                "delta_no_plan_minus_plan_by_state": float(np.nanmean(state_deltas)) if len(state_deltas) else math.nan,
                "delta_bootstrap_unit_mean": float(np.nanmean(unit_deltas)) if unit_deltas else math.nan,
                "delta_ci95_low": ci_low,
                "delta_ci95_high": ci_high,
            }
        )
    return summary


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_wave(
    llm_runner: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    wave_index: int,
    episode_seeds: Sequence[int],
    sample_handle,
) -> list[dict[str, Any]]:
    env_count = len(episode_seeds)
    group_n = 2 * args.samples_per_mode
    env = build_doudizhu_grounding_envs(
        seed=args.env_seed_start + wave_index * args.num_envs,
        env_num=env_count,
        group_n=group_n,
        is_train=False,
        env_config=make_env_config(args),
        resources_per_worker={"num_cpus": 0.1, "num_gpus": 0},
    )

    all_rows: list[dict[str, Any]] = []
    try:
        reset_kwargs = [{"seed": int(seed)} for seed in episode_seeds]
        images, infos = env.reset(kwargs=reset_kwargs)
        active = [True for _ in range(env_count)]

        for step_index in range(args.max_env_steps):
            if not any(active):
                break

            pre_infos = [dict(info) for info in infos]
            responses = generate_for_states(
                llm_runner=llm_runner,
                tokenizer=tokenizer,
                args=args,
                images=images,
                infos=infos,
                active=active,
                wave_index=wave_index,
                env_count=env_count,
                group_n=group_n,
            )
            actions = build_actions(args, responses, active=active, env_count=env_count)
            next_images, _rewards, dones, post_infos = env.step(actions)
            rows = write_sample_rows(
                sample_handle,
                args,
                episode_seeds=episode_seeds,
                wave_index=wave_index,
                step_index=step_index,
                env_count=env_count,
                group_n=group_n,
                active=active,
                pre_infos=pre_infos,
                responses=responses,
                post_infos=post_infos,
            )
            all_rows.extend(rows)

            active = [not bool(dones[canonical_index(env_index, group_n)]) for env_index in range(env_count)]
            images, infos = next_images, post_infos

            if args.flush_every > 0 and (step_index + 1) % args.flush_every == 0:
                sample_handle.flush()
    finally:
        env.close()
    return all_rows


def main() -> None:
    args = parse_args()
    if args.samples_per_mode <= 0:
        raise ValueError("--samples-per-mode must be positive")
    if args.num_envs <= 0 or args.num_episodes <= 0:
        raise ValueError("--num-envs and --num-episodes must be positive")
    if args.data_parallel_size <= 0 or args.tensor_model_parallel_size <= 0:
        raise ValueError("--data-parallel-size and --tensor-model-parallel-size must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )

    tokenizer = import_tokenizer(args.model_path, args.trust_remote_code)
    print(
        f"vLLM parallelism: data_parallel_size={args.data_parallel_size}, tensor_model_parallel_size={args.tensor_model_parallel_size}, total_gpu_slots={args.data_parallel_size * args.tensor_model_parallel_size}",
        flush=True,
    )
    llm_runner = build_llm_runner(args)

    sample_jsonl = output_dir / "samples.jsonl"
    all_sample_rows: list[dict[str, Any]] = []
    try:
        with sample_jsonl.open("w", encoding="utf-8") as sample_handle:
            remaining = args.num_episodes
            next_seed = args.env_seed_start
            wave_index = 0
            while remaining > 0:
                env_count = min(args.num_envs, remaining)
                episode_seeds = list(range(next_seed, next_seed + env_count))
                print(
                    f"[wave {wave_index}] envs={env_count} seeds={episode_seeds[0]}..{episode_seeds[-1]}",
                    flush=True,
                )
                wave_rows = run_wave(
                    llm_runner=llm_runner,
                    tokenizer=tokenizer,
                    args=args,
                    wave_index=wave_index,
                    episode_seeds=episode_seeds,
                    sample_handle=sample_handle,
                )
                all_sample_rows.extend(wave_rows)
                remaining -= env_count
                next_seed += env_count
                wave_index += 1
    finally:
        llm_runner.close()

    state_rows = aggregate_state_rows(all_sample_rows)
    summary_rows = summarize(state_rows, args)

    write_csv(output_dir / "state_metrics.csv", state_rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary_rows, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )

    primary = next(row for row in summary_rows if row["metric"] == "target_action_match")
    print(json.dumps(primary, indent=2, ensure_ascii=False), flush=True)
    print(f"Wrote samples to {sample_jsonl}", flush=True)
    print(f"Wrote summary to {output_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
