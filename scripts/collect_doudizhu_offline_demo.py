#!/usr/bin/env python3
"""Collect the best vLLM-generated Dou Dizhu trajectory for offline demo playback."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent_system.environments.env_package.doudizhu import build_doudizhu_envs, doudizhu_projection
from agent_system.environments.env_package.doudizhu.envs import DoudizhuSingleEnv
from agent_system.environments.prompts.doudizhu import DOUDIZHU_VISUAL_TEMPLATE, DOUDIZHU_VISUAL_TEMPLATE_ZH
from agent_system.multi_turn_rollout.utils import process_image

DEFAULT_MODEL_PATH = "checkpoints/verl_agent_doudizhu/grpo_qwen3_vl_4b/global_step_40"
INITIAL_MEMORY_EN = "Initial turn. Read the screenshot, identify your hand, and plan the first landlord play."
INITIAL_MEMORY_ZH = "初始回合。阅读截图，识别你的手牌，并规划地主首轮出牌。"
MEMORY_MULTIMODAL_TOKENS = (
    "<image>",
    "<video>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<|placeholder|>",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run N Dou Dizhu games with parallel environments and vLLM, then save the highest-scoring trajectory for offline demo playback.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", default="outputs/doudizhu_offline_demo")
    parser.add_argument("--num-games", type=int, default=16)
    parser.add_argument("--num-envs", type=int, default=8, help="Concurrent Dou Dizhu environments per wave.")
    parser.add_argument("--seed-start", type=int, default=100000)
    parser.add_argument("--max-env-steps", type=int, default=30)
    parser.add_argument("--language", default="en", choices=["en", "zh"])
    parser.add_argument("--chinese-mode", action="store_true", help="Shortcut for --language zh.")
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--max-clicks", type=int, default=20)
    parser.add_argument("--use-ray-envs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--env-worker-cpus", type=float, default=0.1)

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--max-response-length", "--max-new-tokens", dest="max_response_length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0, help="vLLM sampling seed.")
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
    return parser.parse_args()


def import_vllm():
    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError(
            "vLLM is not importable in this Python environment. Activate the vLLM environment, "
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
    return LLM(**supported_kwargs(LLM, engine_kwargs))


def build_sampling_params(args: argparse.Namespace):
    _LLM, SamplingParams = import_vllm()
    kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_response_length,
        "n": 1,
        "seed": args.seed,
        "stop_token_ids": getattr(args, "stop_token_ids", None),
    }
    return SamplingParams(**supported_kwargs(SamplingParams, kwargs))


def output_texts(outputs: Sequence[Any]) -> list[str]:
    texts = []
    for output in outputs:
        if output.outputs:
            texts.append(output.outputs[0].text.strip())
        else:
            texts.append("")
    return texts


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
        sampling_params = build_sampling_params(args)
        while True:
            item = input_queue.get()
            if item is None:
                break
            task_id, requests = item
            try:
                outputs = llm.generate(requests, sampling_params=sampling_params, use_tqdm=False)
                output_queue.put((task_id, output_texts(outputs), None))
            except Exception:
                output_queue.put((task_id, None, traceback.format_exc()))
    except Exception:
        output_queue.put(("__init__", None, traceback.format_exc()))


class LocalVLLMRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.llm = build_single_llm(args)
        self.sampling_params = build_sampling_params(args)

    def generate_texts(self, requests: list[dict[str, Any]]) -> list[str]:
        outputs = self.llm.generate(requests, sampling_params=self.sampling_params, use_tqdm=False)
        return output_texts(outputs)

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

    def generate_texts(self, requests: list[dict[str, Any]]) -> list[str]:
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
            self.input_queues[worker_index].put((task_id, chunk))

        results: list[str | None] = [None for _ in requests]
        while pending:
            task_id, texts, error = self.output_queue.get()
            if error:
                raise RuntimeError(f"vLLM worker task {task_id} failed:\n{error}")
            if task_id == "__init__":
                raise RuntimeError(f"vLLM worker failed during initialization:\n{error}")
            indices = pending.pop(task_id)
            if len(indices) != len(texts):
                raise RuntimeError(f"Worker task {task_id} returned {len(texts)} outputs for {len(indices)} requests.")
            for request_index, text in zip(indices, texts, strict=True):
                results[request_index] = text

        return [text if text is not None else "" for text in results]

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


def ui(en: str, zh: str, chinese_mode: bool) -> str:
    return zh if chinese_mode else en


def json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def make_env_config(args: argparse.Namespace, language: str, chinese_mode: bool) -> dict[str, Any]:
    return {
        "doudizhu": {
            "use_ray": bool(args.use_ray_envs),
            "language": language,
            "chinese_mode": chinese_mode,
            "image_width": args.image_width,
            "image_height": args.image_height,
            "max_clicks": args.max_clicks,
        }
    }


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


def build_request(tokenizer: Any, prompt: str, image: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    prompt_text = apply_chat_template(tokenizer, prompt, args.enable_thinking)
    return {
        "prompt": prompt_text,
        "multi_modal_data": {"image": [process_image(image)]},
    }


def sanitize_memory(memory: Any) -> str:
    if not isinstance(memory, str):
        return ""
    for token in MEMORY_MULTIMODAL_TOKENS:
        memory = memory.replace(token, "")
    return memory.strip()


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


def project_response(response: str, max_clicks: int) -> dict[str, Any]:
    actions, valids = doudizhu_projection([response], max_clicks=max_clicks)
    action = actions[0]
    action["projection_valid"] = int(valids[0])
    return action


def _norm_to_pixel(obs: np.ndarray, x: float, y: float):
    height, width = obs.shape[:2]
    px = int(round(float(x) / 1000.0 * (width - 1)))
    py = int(round(float(y) / 1000.0 * (height - 1)))
    return max(0, min(width - 1, px)), max(0, min(height - 1, py))


def _load_overlay_font(size: int):
    for font_name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def annotate_clicks(obs: np.ndarray, clicks):
    image = Image.fromarray(obs.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    font = _load_overlay_font(15)
    small_font = _load_overlay_font(13)
    colors = [
        (255, 64, 64, 235),
        (255, 190, 48, 235),
        (58, 202, 255, 235),
        (158, 255, 88, 235),
        (214, 122, 255, 235),
        (255, 118, 184, 235),
        (90, 255, 208, 235),
        (255, 255, 255, 235),
    ]
    previous = None
    for idx, click in enumerate(clicks if isinstance(clicks, list) else []):
        if not isinstance(click, (list, tuple)) or len(click) != 2:
            continue
        x, y = float(click[0]), float(click[1])
        px, py = _norm_to_pixel(obs, x, y)
        color = colors[idx % len(colors)]
        if previous is not None:
            draw.line((previous[0], previous[1], px, py), fill=color, width=3)
        previous = (px, py)
        radius = 14
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), outline=color, width=4)
        draw.line((px - 22, py, px + 22, py), fill=color, width=3)
        draw.line((px, py - 22, px, py + 22), fill=color, width=3)
        draw.ellipse((px - 9, py - 9, px + 9, py + 9), fill=(0, 0, 0, 130), outline=color, width=2)
        draw.text((px - 4, py - 8), str(idx + 1), fill=(255, 255, 255, 255), font=font)
        label = f"{idx + 1}: [{x:.0f}, {y:.0f}]"
        text_x = min(max(4, px + 18), image.width - 96)
        text_y = min(max(4, py - 24), image.height - 24)
        draw.rounded_rectangle((text_x - 3, text_y - 2, text_x + 88, text_y + 17), radius=4, fill=(0, 0, 0, 150))
        draw.text((text_x, text_y), label, fill=(255, 255, 255, 255), font=small_font)
    return np.array(image, dtype=np.uint8)


def format_demo_clicks(clicks, chinese_mode: bool) -> str:
    rows = []
    for idx, click in enumerate(clicks if isinstance(clicks, list) else []):
        if not isinstance(click, (list, tuple)) or len(click) != 2:
            continue
        rows.append(f"{idx + 1}. [{float(click[0]):.1f}, {float(click[1]):.1f}]")
    return "\n".join(rows) if rows else ui("(no valid projected clicks)", "（无有效解析点击）", chinese_mode)


def pretty_env_action(env: DoudizhuSingleEnv, action):
    if action is None or action == "":
        return "-"
    return env._pretty_action(action)


def format_step_outputs(
    env: DoudizhuSingleEnv,
    response: str,
    action: dict[str, Any],
    info: dict[str, Any],
    reward: float,
    done: bool,
    chinese_mode: bool,
):
    fallback = bool(info.get("fallback_used", False))
    projection_valid = bool(action.get("projection_valid", 0))
    result = ui("Fallback move was executed", "已执行兜底动作", chinese_mode) if fallback else ui("Valid move was executed", "已执行有效出牌", chinese_mode)
    game_state = ui("Game continues", "牌局继续", chinese_mode)
    if done:
        won = bool(info.get("won", 0))
        game_state = ui(
            "Game over: Player 0 won" if won else "Game over: Player 0 lost",
            "游戏结束：玩家 0 赢了" if won else "游戏结束：玩家 0 输了",
            chinese_mode,
        )

    summary = ui(
        f"{result}\n"
        f"Projection: {'valid' if projection_valid else 'invalid'}\n"
        f"Click valid ratio: {info.get('click_valid_ratio', 0.0):.2f}\n"
        f"Reward: {reward:.3f}\n"
        f"{game_state}",
        f"{result}\n"
        f"标签解析：{'有效' if projection_valid else '无效'}\n"
        f"点击有效比例：{info.get('click_valid_ratio', 0.0):.2f}\n"
        f"奖励：{reward:.3f}\n"
        f"{game_state}",
        chinese_mode,
    )

    action_parse_ok = bool(action.get("action_tag_parse_ok", False))
    action_parse_status = ui("success", "成功", chinese_mode)
    if not action_parse_ok:
        action_parse_status = ui(
            f"failed: {action.get('action_tag_parse_error', '-')}",
            f"失败：{action.get('action_tag_parse_error', '-')}",
            chinese_mode,
        )
    model_action = ui(
        f"Raw <action>:\n{action.get('raw_action_text') or '-'}\n\n"
        f"Parsed action:\n{action.get('normalized_action_text') or '-'}\n\n"
        f"Parse status: {action_parse_status}",
        f"原始 <action>：\n{action.get('raw_action_text') or '-'}\n\n"
        f"解析动作：\n{action.get('normalized_action_text') or '-'}\n\n"
        f"解析状态：{action_parse_status}",
        chinese_mode,
    )

    env_action = ui(
        f"Selected cards: {pretty_env_action(env, info.get('selected_cards'))}\n"
        f"Submit kind: {info.get('submit_kind') or '-'}\n"
        f"Executed action: {pretty_env_action(env, info.get('game_action'))}\n"
        f"Fallback used: {'yes' if fallback else 'no'}",
        f"选中的牌：{pretty_env_action(env, info.get('selected_cards'))}\n"
        f"提交类型：{info.get('submit_kind') or '-'}\n"
        f"实际执行动作：{pretty_env_action(env, info.get('game_action'))}\n"
        f"是否兜底：{'是' if fallback else '否'}",
        chinese_mode,
    )

    click_text = ui(
        f"Raw <tool_call>:\n{action.get('raw_tool_call_text') or '-'}\n\n"
        f"Parsed clicks:\n{format_demo_clicks(action.get('clicks', []), chinese_mode)}",
        f"原始 <tool_call>：\n{action.get('raw_tool_call_text') or '-'}\n\n"
        f"解析点击：\n{format_demo_clicks(action.get('clicks', []), chinese_mode)}",
        chinese_mode,
    )
    return summary, model_action, env_action, click_text, response or ""


def initial_memory(chinese_mode: bool) -> str:
    return INITIAL_MEMORY_ZH if chinese_mode else INITIAL_MEMORY_EN


def doudizhu_prompt(language: str, memory: str) -> str:
    template = DOUDIZHU_VISUAL_TEMPLATE_ZH if language == "zh" else DOUDIZHU_VISUAL_TEMPLATE
    no_memory = "没有上一轮记忆。" if language == "zh" else "No previous memory."
    return template.format(previous_memory=memory or no_memory)


def init_episode(seed: int, language: str, chinese_mode: bool) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "language": language,
        "chinese_mode": chinese_mode,
        "total_reward": 0.0,
        "episode_length": 0,
        "won": 0.0,
        "winner_id": None,
        "payoffs": None,
        "steps": [],
        "last_info": {},
    }


def collect_wave(
    llm_runner: Any,
    tokenizer: Any,
    format_env: DoudizhuSingleEnv,
    args: argparse.Namespace,
    wave_index: int,
    episode_seeds: Sequence[int],
    language: str,
    chinese_mode: bool,
) -> list[dict[str, Any]]:
    env_count = len(episode_seeds)
    env = build_doudizhu_envs(
        seed=args.seed_start + wave_index * args.num_envs,
        env_num=env_count,
        group_n=1,
        is_train=False,
        env_config=make_env_config(args, language, chinese_mode),
        resources_per_worker={"num_cpus": args.env_worker_cpus, "num_gpus": 0},
    )
    episodes = [init_episode(seed, language, chinese_mode) for seed in episode_seeds]
    memories = [initial_memory(chinese_mode) for _ in range(env_count)]
    active = [True for _ in range(env_count)]

    try:
        images, _infos = env.reset(kwargs=[{"seed": int(seed)} for seed in episode_seeds])
        for step_index in range(args.max_env_steps):
            if not any(active):
                break

            requests = []
            request_env_indices = []
            obs_before_by_env: dict[int, np.ndarray] = {}
            for env_index in range(env_count):
                if not active[env_index]:
                    continue
                obs_before_by_env[env_index] = images[env_index].copy()
                prompt = doudizhu_prompt(language, memories[env_index])
                requests.append(build_request(tokenizer, prompt, images[env_index], args))
                request_env_indices.append(env_index)

            response_texts = llm_runner.generate_texts(requests)
            actions = [empty_action() for _ in range(env_count)]
            raw_responses = ["" for _ in range(env_count)]
            overlays: dict[int, np.ndarray] = {}
            for env_index, response in zip(request_env_indices, response_texts, strict=True):
                action = project_response(response, args.max_clicks)
                actions[env_index] = action
                raw_responses[env_index] = response
                overlays[env_index] = annotate_clicks(obs_before_by_env[env_index], action.get("clicks", []))

            next_images, rewards, dones, step_infos = env.step(actions)
            for env_index in range(env_count):
                if not active[env_index]:
                    continue
                action = actions[env_index]
                info = dict(step_infos[env_index])
                reward = float(rewards[env_index])
                done = bool(dones[env_index])
                episode = episodes[env_index]
                episode["total_reward"] += reward
                episode["episode_length"] += 1
                episode["last_info"] = info
                episode["won"] = float(info.get("won", episode["won"]))
                episode["winner_id"] = info.get("winner_id")
                episode["payoffs"] = info.get("payoffs")

                memory = sanitize_memory(action.get("memory", ""))
                if memory:
                    memories[env_index] = memory[:512]

                summary, model_action, env_action, click_text, raw_response = format_step_outputs(
                    env=format_env,
                    response=raw_responses[env_index],
                    action=action,
                    info=info,
                    reward=reward,
                    done=done,
                    chinese_mode=chinese_mode,
                )
                episode["steps"].append(
                    {
                        "step_index": step_index,
                        "obs_after": next_images[env_index].copy(),
                        "click_overlay": overlays.get(env_index, obs_before_by_env[env_index]),
                        "summary": summary,
                        "model_action": model_action,
                        "env_action": env_action,
                        "clicks": click_text,
                        "raw_response": raw_response,
                        "reward": reward,
                        "done": done,
                        "action": action,
                        "info": info,
                    }
                )
            active = [active[idx] and not bool(dones[idx]) for idx in range(env_count)]
            images = next_images
    finally:
        env.close()

    for episode in episodes:
        last_info = dict(episode.get("last_info", {}))
        if not math.isfinite(float(episode["total_reward"])):
            episode["total_reward"] = float("-inf")
        episode["won"] = float(last_info.get("won", episode.get("won", 0.0)))
        episode["winner_id"] = last_info.get("winner_id", episode.get("winner_id"))
        episode["payoffs"] = last_info.get("payoffs", episode.get("payoffs"))
    return episodes


def save_best_episode(best: dict[str, Any], args: argparse.Namespace, elapsed_seconds: float) -> Path:
    output_dir = Path(args.output_dir).expanduser()
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    saved_steps = []
    for step in best["steps"]:
        step_index = int(step["step_index"])
        current_name = f"images/step_{step_index:03d}_current.png"
        overlay_name = f"images/step_{step_index:03d}_overlay.png"
        Image.fromarray(step["obs_after"].astype(np.uint8)).save(output_dir / current_name)
        Image.fromarray(step["click_overlay"].astype(np.uint8)).save(output_dir / overlay_name)
        saved_step = {key: value for key, value in step.items() if key not in {"obs_after", "click_overlay"}}
        saved_step.update(
            {
                "current_image": current_name,
                "overlay_image": overlay_name,
            }
        )
        saved_steps.append(saved_step)

    payload = {
        "schema_version": 2,
        "created_at_unix": time.time(),
        "elapsed_seconds": elapsed_seconds,
        "model_path": args.model_path,
        "num_games": int(args.num_games),
        "num_envs": int(args.num_envs),
        "data_parallel_size": int(args.data_parallel_size),
        "tensor_model_parallel_size": int(args.tensor_model_parallel_size),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "max_num_batched_tokens": int(args.max_num_batched_tokens),
        "max_num_seqs": int(args.max_num_seqs),
        "max_model_len": args.max_model_len,
        "dtype": args.dtype,
        "enforce_eager": bool(args.enforce_eager),
        "enable_chunked_prefill": bool(args.enable_chunked_prefill),
        "enable_prefix_caching": bool(args.enable_prefix_caching),
        "trust_remote_code": bool(args.trust_remote_code),
        "mm_processor_cache_gb": float(args.mm_processor_cache_gb),
        "seed_start": int(args.seed_start),
        "max_env_steps": int(args.max_env_steps),
        "language": best["language"],
        "chinese_mode": bool(best["chinese_mode"]),
        "selection_metric": "total_reward",
        "best_seed": int(best["seed"]),
        "best_total_reward": float(best["total_reward"]),
        "best_episode_length": int(best["episode_length"]),
        "best_won": float(best["won"]),
        "winner_id": best["winner_id"],
        "payoffs": best["payoffs"],
        "steps": saved_steps,
    }
    output_path = output_dir / "demo.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    return output_path


def validate_args(args: argparse.Namespace) -> None:
    if args.num_games <= 0:
        raise ValueError("--num-games must be positive.")
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive.")
    if args.max_env_steps <= 0:
        raise ValueError("--max-env-steps must be positive.")
    if args.max_response_length <= 0:
        raise ValueError("--max-response-length must be positive.")
    if args.data_parallel_size <= 0 or args.tensor_model_parallel_size <= 0:
        raise ValueError("--data-parallel-size and --tensor-model-parallel-size must be positive.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    language = "zh" if args.chinese_mode or args.language == "zh" else "en"
    chinese_mode = language == "zh"

    tokenizer = import_tokenizer(args.model_path, args.trust_remote_code)
    args.stop_token_ids = [int(tokenizer.eos_token_id)] if tokenizer.eos_token_id is not None else None
    print(
        f"vLLM parallelism: data_parallel_size={args.data_parallel_size}, "
        f"tensor_model_parallel_size={args.tensor_model_parallel_size}, "
        f"total_gpu_slots={args.data_parallel_size * args.tensor_model_parallel_size}",
        flush=True,
    )
    print(
        f"vLLM engine: gpu_memory_utilization={args.gpu_memory_utilization}, "
        f"max_num_batched_tokens={args.max_num_batched_tokens}, max_num_seqs={args.max_num_seqs}, "
        f"max_model_len={args.max_model_len}, dtype={args.dtype}",
        flush=True,
    )
    llm_runner = build_llm_runner(args)
    format_env = DoudizhuSingleEnv(seed=42, env_config=make_env_config(args, language, chinese_mode))
    best = None
    start_time = time.time()
    completed_games = 0

    try:
        while completed_games < args.num_games:
            env_count = min(args.num_envs, args.num_games - completed_games)
            episode_seeds = list(range(args.seed_start + completed_games, args.seed_start + completed_games + env_count))
            wave_index = completed_games // args.num_envs
            print(
                f"[wave {wave_index}] envs={env_count} seeds={episode_seeds[0]}..{episode_seeds[-1]}",
                flush=True,
            )
            episodes = collect_wave(
                llm_runner=llm_runner,
                tokenizer=tokenizer,
                format_env=format_env,
                args=args,
                wave_index=wave_index,
                episode_seeds=episode_seeds,
                language=language,
                chinese_mode=chinese_mode,
            )
            for episode in episodes:
                completed_games += 1
                score = float(episode["total_reward"])
                if best is None or score > float(best["total_reward"]):
                    best = episode
                print(
                    f"[{completed_games}/{args.num_games}] seed={episode['seed']} score={score:.3f} "
                    f"won={float(episode['won']):.0f} length={episode['episode_length']} "
                    f"best={float(best['total_reward']):.3f}",
                    flush=True,
                )

        if best is None:
            raise RuntimeError("No episode was collected.")
        output_path = save_best_episode(best, args, elapsed_seconds=time.time() - start_time)
        print(f"Saved best offline demo to: {output_path}", flush=True)
    finally:
        llm_runner.close()


if __name__ == "__main__":
    main()
