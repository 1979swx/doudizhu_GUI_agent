import argparse
import base64
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

# Usage
# -----
# This script starts a Gradio debugger for the Dou Dizhu GUI environment.
# It has four pages:
#   1. Human play: manually click the observation image, then submit the clicks
#      to the environment.
#   2. Spectator mode: a VL model reads the screenshot and chooses the full
#      card action and GUI clicks by itself.
#   3. Commander mode: you type a semantic card action such as "3 3",
#      "10 J Q K A", or "不要"; the model only converts that command into GUI
#      clicks.
#   4. Watch commander mode: a rule teacher chooses the semantic card action;
#      the model converts that command into GUI clicks, while the game advances
#      by the teacher action so you can watch grounding behavior.
#
# Basic local-model run:
#   conda activate verl-agent-bw
#   python scripts/human_play_doudizhu_web.py \
#     --model-backend local \
#     --model-path checkpoints/verl_agent_doudizhu/grpo_qwen3_vl_4b/global_step_40 \
#     --chinese-mode
#
# Local-model notes:
#   --model-path can be a merged HuggingFace model directory, an actor
#   directory, or a verl global_step directory. Unmerged FSDP actor checkpoints
#   are auto-merged by default. Use --no-auto-merge to disable this.
#   --device-map, --torch-dtype, --max-new-tokens, --temperature, --top-p, and
#   --enable-thinking control local Transformers generation.
#
# OpenAI-compatible API run:
#   export OPENAI_API_KEY=...
#   conda activate verl-agent-bw
#   python scripts/human_play_doudizhu_web.py \
#     --model-backend api \
#     --api-base-url https://api.openai.com/v1 \
#     --api-model <vision-model-name> \
#     --api-key-env OPENAI_API_KEY \
#     --chinese-mode
#
# Kimi/Moonshot example:
#   export MOONSHOT_API_KEY=...
#   conda activate verl-agent-bw
#   python scripts/human_play_doudizhu_web.py \
#     --model-backend api \
#     --api-base-url https://api.moonshot.cn/v1 \
#     --api-model kimi-k2.6 \
#     --api-key-env MOONSHOT_API_KEY \
#     --chinese-mode
#
# API backend notes:
#   The API backend sends the current screenshot as a PNG data URL in an
#   OpenAI-compatible chat/completions request. The API response must still
#   follow this script's XML prompt format, because the existing
#   doudizhu_projection and command parser are reused. There is intentionally
#   no format-repair or retry layer; invalid model output is surfaced directly
#   in the UI. The default temperature is 1.0.
#
# Web server:
#   --server-name defaults to 0.0.0.0 and --server-port defaults to 7860.
#
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import gradio as gr
except ImportError:
    print("Gradio is not installed. Please run: pip install gradio")
    sys.exit(1)

# Add project root to sys.path to ensure absolute imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from agent_system.environments.env_package.doudizhu.envs import DoudizhuSingleEnv
    from agent_system.environments.env_package.doudizhu_grounding.envs import DoudizhuGroundingSingleEnv
    from agent_system.environments.env_package.doudizhu.projection import doudizhu_projection, parse_left_click_tool_call
    from agent_system.environments.prompts.doudizhu import DOUDIZHU_VISUAL_TEMPLATE, DOUDIZHU_VISUAL_TEMPLATE_ZH
except ImportError as e:
    print(f"Failed to import DoudizhuSingleEnv: {e}")
    print("Make sure you run this script from the project root or the conda environment is active.")
    sys.exit(1)

DEFAULT_MODEL_PATH = "checkpoints/verl_agent_doudizhu/grpo_qwen3_vl_4b/global_step_40"
INITIAL_MEMORY_EN = "Initial turn. Read the screenshot, identify your hand, and plan the first landlord play."
INITIAL_MEMORY_ZH = "初始回合。阅读截图，识别你的手牌，并规划地主首轮出牌。"
COMMAND_PROMPT_TEMPLATE_EN = """<image>
You are controlling the Dou Dizhu GUI by normalized clicks.

Human commanded action: {command}

Your only job is to execute that commanded action on the screenshot. Do not choose a different card action.
- If the command is pass, click only the PASS button.
- Otherwise, click each matching card in your bottom hand, then click the PLAY button.
- Coordinates must be normalized numbers from 0 to 1000.

Return exactly two XML tags. In <plan>, briefly identify what you will click. In <tool_call>, output exactly one left_click(...) call:
<plan>Briefly identify which visible card(s) or button you will click.</plan><tool_call>left_click([x1,y1],[x2,y2])</tool_call>
"""
COMMAND_PROMPT_TEMPLATE_ZH = """
你正在通过鼠标点击来控制斗地主 GUI。

人类指挥动作：{command}

<image>你的任务是在截图中执行这个人类指挥动作。不要自行选择其它出牌。
- 你通过 [x,y] 坐标来进行点击动作，坐标必须是 0 到 1000 范围内的归一化数字，[0,0] 代表左上角，[1000,1000] 代表右下角。
- 游戏页面的底部有手牌，其上方有‘出牌’和‘不要’按钮，这是主要交互区域。
- 如果指挥动作是“不要”或 pass，只点击“不要”按钮。
- 如果指挥动作是出牌，则依次点击底部手牌中与指挥动作匹配的每张牌，然后点击“出牌”按钮。
- 也就是说，每一轮动作的最后必须以点击“出牌”或“不要”两个按钮之一结尾。

输出一个 left_click([x1,y1],[x2,y2],...,[xN,yN]) 调用，每个坐标对代表一次点击，N个坐标对代表N次点击。

在一个名为 <plan> </plan> 的 XML 标签包裹中规划你的行动，然后将确切行动输出为一个名为 <tool_call> </tool_call> 的 XML 标签包裹中。
例子：
指挥动作：不要
输出：<plan>指挥动作是不要，只点击“不要”按钮完成本轮。</plan><tool_call>left_click([566,764])</tool_call>

指挥动作：3 3
输出：<plan>指挥动作是出一对 3，依次点击两张 3，再点击“出牌”按钮。</plan><tool_call>left_click([55,850],[100,860],[430,755])</tool_call>


当前人类指挥动作：{command}
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Dou Dizhu human/debug and model spectator web UI.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Merged HF model dir, actor dir, or verl global_step dir.")
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument(
        "--device-map",
        default="cuda:0",
        help="Transformers device_map for model loading. Use cuda:0/gpu0/0 for single-GPU inference, or auto for automatic dispatch.",
    )
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--enable-thinking", action="store_true", help="Pass enable_thinking=True to tokenizer.apply_chat_template for model inference.")
    parser.add_argument("--no-auto-merge", action="store_true", help="Do not auto-merge verl FSDP actor checkpoints.")
    parser.add_argument("--model-backend", default="local", choices=["local", "api"], help="Inference backend for spectator/commander modes.")
    parser.add_argument("--api-base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"), help="OpenAI-compatible API base URL.")
    parser.add_argument("--api-model", default=os.environ.get("DOUDIZHU_API_MODEL", ""), help="Model name for API backend.")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Environment variable that stores the API key.")
    parser.add_argument("--api-timeout", type=float, default=60.0, help="API request timeout in seconds.")
    parser.add_argument("--api-thinking", default="default", choices=["default", "enabled", "disabled"], help="Optional Kimi-compatible thinking setting for API backend.")
    parser.add_argument("--language", default="en", choices=["en", "zh"], help="UI/prompt language for the Dou Dizhu environment.")
    parser.add_argument("--chinese-mode", action="store_true", help="Shortcut for --language zh.")
    return parser.parse_args()


ARGS = parse_args()
LANGUAGE = "zh" if ARGS.chinese_mode or ARGS.language == "zh" else "en"
CHINESE_MODE = LANGUAGE == "zh"
INITIAL_MEMORY = INITIAL_MEMORY_ZH if CHINESE_MODE else INITIAL_MEMORY_EN
SPECTATOR_PROMPT_TEMPLATE = DOUDIZHU_VISUAL_TEMPLATE_ZH if CHINESE_MODE else DOUDIZHU_VISUAL_TEMPLATE
COMMAND_PROMPT_TEMPLATE = COMMAND_PROMPT_TEMPLATE_ZH if CHINESE_MODE else COMMAND_PROMPT_TEMPLATE_EN


def ui(en: str, zh: str) -> str:
    return zh if CHINESE_MODE else en


MODE_HUMAN = ui("Human play", "人工游玩")
MODE_SPECTATOR = ui("Spectator mode", "旁观模式")
MODE_COMMANDER = ui("Commander mode", "指挥模式")
MODE_WATCH_COMMANDER = ui("Watch commander mode", "观看指挥模式")


def switch_mode(mode):
    return (
        gr.update(visible=mode == MODE_HUMAN),
        gr.update(visible=mode == MODE_SPECTATOR),
        gr.update(visible=mode == MODE_COMMANDER),
        gr.update(visible=mode == MODE_WATCH_COMMANDER),
    )


# Global state to track human interactions
current_clicks = []
current_obs = None
done = False
env = DoudizhuSingleEnv(
    seed=42,
    env_config={
        "doudizhu": {
            "language": LANGUAGE,
            "chinese_mode": CHINESE_MODE,
        }
    },
)
watch_commander_env = DoudizhuGroundingSingleEnv(
    seed=42,
    env_config={
        "doudizhu": {
            "language": LANGUAGE,
            "chinese_mode": CHINESE_MODE,
        },
        "doudizhu_grounding": {
            "teacher_policy": "rule_v1",
        },
    },
)
watch_commander_obs = None
watch_commander_done = False
spectator_agent = None
spectator_agent_key = None
spectator_memory = INITIAL_MEMORY
last_raw_response = ""


def _find_hf_weight_files(path: Path) -> bool:
    if not path.is_dir():
        return False
    direct_names = {
        "pytorch_model.bin",
        "model.safetensors",
        "pytorch_model.bin.index.json",
        "model.safetensors.index.json",
    }
    if any((path / name).exists() for name in direct_names):
        return True
    return any(path.glob("*.safetensors")) or any(path.glob("pytorch_model-*.bin"))


def _find_sharded_actor_dir(path: Path):
    candidates = [path, path / "actor"]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("model_world_size_*_rank_*.pt")):
            return candidate
    return None


def _resolve_model_dir(model_path: str, auto_merge: bool) -> str:
    path = Path(model_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()

    if _find_hf_weight_files(path):
        return str(path)

    actor_dir = path / "actor" if (path / "actor").is_dir() else path
    merged_dir = actor_dir / "hf_merged"
    if _find_hf_weight_files(merged_dir):
        return str(merged_dir)

    sharded_actor_dir = _find_sharded_actor_dir(path)
    if sharded_actor_dir is None:
        raise FileNotFoundError(
            f"Could not find HF model weights or verl actor shards under: {path}"
        )

    if not auto_merge:
        raise RuntimeError(
            "This looks like an unmerged verl FSDP actor checkpoint. Merge it first:\n"
            f"python scripts/model_merger.py merge --backend fsdp --local_dir {sharded_actor_dir} --target_dir {merged_dir}"
        )

    os.makedirs(merged_dir, exist_ok=True)
    from scripts.model_merger import FSDPModelMerger, ModelMergerConfig

    config = ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        local_dir=str(sharded_actor_dir),
        hf_model_config_path=str(sharded_actor_dir),
        target_dir=str(merged_dir),
    )
    FSDPModelMerger(config).merge_and_save()
    if not _find_hf_weight_files(merged_dir):
        raise RuntimeError(f"Checkpoint merge finished, but no HF weight files were found in {merged_dir}")
    return str(merged_dir)


def _torch_dtype(dtype_name: str):
    import torch

    if dtype_name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def _parse_device_map(device_map: str):
    if device_map is None:
        return None
    normalized = str(device_map).strip().lower()
    if not normalized or normalized in {"none", "null"}:
        return None
    if normalized in {"auto", "balanced", "balanced_low_0", "sequential"}:
        return normalized
    if normalized in {"cpu", "mps"}:
        return {"": normalized}
    if normalized.isdigit():
        return {"": f"cuda:{normalized}"}
    if normalized.startswith("gpu") and normalized[3:].isdigit():
        return {"": f"cuda:{normalized[3:]}"}
    if normalized.startswith("cuda"):
        return {"": normalized}
    return device_map


def _get_auto_model_class(model_dir: str):
    from transformers import AutoConfig, AutoModelForCausalLM

    try:
        from transformers import AutoModelForImageTextToText
    except ImportError:
        AutoModelForImageTextToText = None

    try:
        from transformers import AutoModelForVision2Seq
    except ImportError:
        AutoModelForVision2Seq = None

    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    if (
        AutoModelForImageTextToText is not None
        and type(model_config) in AutoModelForImageTextToText._model_mapping.keys()
    ):
        return AutoModelForImageTextToText
    if AutoModelForVision2Seq is not None and type(model_config) in AutoModelForVision2Seq._model_mapping.keys():
        return AutoModelForVision2Seq
    return AutoModelForCausalLM


def _extract_xml_tag(text: str, tag: str):
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.IGNORECASE | re.DOTALL)
    if match is None:
        return None
    return match.group(1).strip()


def parse_command_tool_call_response(response: str, command: str, max_clicks: int):
    tool_call_text = _extract_xml_tag(response, "tool_call") if isinstance(response, str) else None
    clicks, normalized_tool_calls, valid = parse_left_click_tool_call(tool_call_text or "", max_clicks=int(max_clicks))

    return {
        "clicks": clicks,
        "projection_valid": int(valid),
        "semantic_action": command.strip() if isinstance(command, str) else "",
        "chat": "",
        "memory": "",
        "raw_action_text": command.strip() if isinstance(command, str) else "",
        "raw_tool_call_text": tool_call_text or "",
        "raw_response": response if isinstance(response, str) else "",
        "tool_calls": normalized_tool_calls,
        "tool_calling": len(normalized_tool_calls),
    }


class DoudizhuSpectatorAgent:
    def __init__(self, model_path: str, auto_merge: bool, device_map: str, torch_dtype: str):
        import torch
        from transformers import AutoProcessor, AutoTokenizer

        self.model_dir = _resolve_model_dir(model_path, auto_merge=auto_merge)
        self.processor = AutoProcessor.from_pretrained(self.model_dir, trust_remote_code=True)
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
        auto_model_class = _get_auto_model_class(self.model_dir)
        self.model = auto_model_class.from_pretrained(
            self.model_dir,
            torch_dtype=_torch_dtype(torch_dtype),
            device_map=_parse_device_map(device_map),
            trust_remote_code=True,
        )
        self.model.eval()
        self.torch = torch

    def _generate_response(
        self,
        obs: np.ndarray,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        enable_thinking: bool,
    ):
        chat = [{"role": "user", "content": prompt}]
        prompt_text = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=bool(enable_thinking),
        )
        prompt_text = prompt_text.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
        if "<|image_pad|>" not in prompt_text:
            prompt_text = "<|vision_start|><|image_pad|><|vision_end|>\n" + prompt_text

        image = Image.fromarray(obs.astype(np.uint8)).convert("RGB")
        inputs = self.processor(text=[prompt_text], images=[image], return_tensors="pt")
        device = getattr(self.model, "device", None)
        if device is not None:
            inputs = inputs.to(device)

        generate_kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": bool(temperature and temperature > 0),
            "top_p": float(top_p),
        }
        if generate_kwargs["do_sample"]:
            generate_kwargs["temperature"] = float(temperature)

        with self.torch.no_grad():
            output_ids = self.model.generate(**inputs, **generate_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, prompt_len:]
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return response

    def generate_action(
        self,
        obs: np.ndarray,
        memory: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        enable_thinking: bool,
    ):
        no_memory = "没有上一轮记忆。" if CHINESE_MODE else "No previous memory."
        prompt = SPECTATOR_PROMPT_TEMPLATE.format(previous_memory=memory or no_memory)
        response = self._generate_response(
            obs,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            enable_thinking=enable_thinking,
        )
        actions, _valids = doudizhu_projection([response], max_clicks=env.max_clicks)
        return response, actions[0]

    def generate_commanded_action(
        self,
        obs: np.ndarray,
        command: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        enable_thinking: bool,
    ):
        prompt = COMMAND_PROMPT_TEMPLATE.format(command=(command or "").strip())
        response = self._generate_response(
            obs,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            enable_thinking=enable_thinking,
        )
        return response, parse_command_tool_call_response(response, command or "", max_clicks=env.max_clicks)


def _api_chat_url(base_url: str) -> str:
    base_url = (base_url or "").strip().rstrip("/")
    if not base_url:
        raise ValueError("API base URL is empty.")
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def _image_data_url(obs: np.ndarray) -> str:
    image = Image.fromarray(obs.astype(np.uint8)).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _message_content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return "" if content is None else str(content)


class DoudizhuApiAgent:
    def __init__(self, base_url: str, model: str, api_key_env: str, timeout: float, thinking: str):
        self.base_url = (base_url or "").strip()
        self.model = (model or "").strip()
        self.api_key_env = (api_key_env or "").strip()
        self.timeout = float(timeout)
        self.thinking = (thinking or "default").strip()
        if not self.model:
            raise ValueError("API model is empty. Set --api-model or fill the API model field.")
        if not self.api_key_env:
            raise ValueError("API key env var name is empty.")
        if not os.environ.get(self.api_key_env):
            raise ValueError(f"API key env var {self.api_key_env!r} is not set.")
        if self.thinking not in {"default", "enabled", "disabled"}:
            raise ValueError("API thinking must be one of: default, enabled, disabled.")

    def _generate_response(
        self,
        obs: np.ndarray,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        enable_thinking: bool,
    ):
        del enable_thinking
        prompt_text = (prompt or "").replace("<image>", "The current screenshot is attached.")
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": _image_data_url(obs)}},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ],
            "max_tokens": int(max_new_tokens),
        }
        if self.thinking != "default":
            payload["thinking"] = {"type": self.thinking}
        if self.thinking != "disabled":
            payload["temperature"] = float(temperature)
            payload["top_p"] = float(top_p)
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {os.environ[self.api_key_env]}",
            "Content-Type": "application/json",
        }
        req = urllib_request.Request(_api_chat_url(self.base_url), data=data, headers=headers, method="POST")
        try:
            with urllib_request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"API request failed with HTTP {exc.code}: {error_body}") from exc
        except URLError as exc:
            raise RuntimeError(f"API request failed: {exc}") from exc

        parsed = json.loads(body)
        choices = parsed.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f"API response has no choices: {body}")
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                return _message_content_to_text(message.get("content")).strip()
            return _message_content_to_text(first.get("text")).strip()
        raise RuntimeError(f"API response choice is invalid: {body}")

    def generate_action(
        self,
        obs: np.ndarray,
        memory: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        enable_thinking: bool,
    ):
        no_memory = "没有上一轮记忆。" if CHINESE_MODE else "No previous memory."
        prompt = SPECTATOR_PROMPT_TEMPLATE.format(previous_memory=memory or no_memory)
        response = self._generate_response(
            obs,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            enable_thinking=enable_thinking,
        )
        actions, _valids = doudizhu_projection([response], max_clicks=env.max_clicks)
        return response, actions[0]

    def generate_commanded_action(
        self,
        obs: np.ndarray,
        command: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        enable_thinking: bool,
    ):
        prompt = COMMAND_PROMPT_TEMPLATE.format(command=(command or "").strip())
        response = self._generate_response(
            obs,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            enable_thinking=enable_thinking,
        )
        return response, parse_command_tool_call_response(response, command or "", max_clicks=env.max_clicks)


def _load_spectator_agent(
    model_backend: str,
    model_path: str,
    auto_merge: bool,
    device_map: str,
    torch_dtype: str,
    api_base_url: str,
    api_model: str,
    api_key_env: str,
    api_timeout: float,
    api_thinking: str,
):
    global spectator_agent, spectator_agent_key
    key = (
        model_backend,
        model_path,
        bool(auto_merge),
        device_map,
        torch_dtype,
        api_base_url,
        api_model,
        api_key_env,
        float(api_timeout),
        api_thinking,
    )
    if spectator_agent is not None and spectator_agent_key == key:
        return spectator_agent
    if model_backend == "api":
        spectator_agent = DoudizhuApiAgent(
            base_url=api_base_url,
            model=api_model,
            api_key_env=api_key_env,
            timeout=api_timeout,
            thinking=api_thinking,
        )
    else:
        spectator_agent = DoudizhuSpectatorAgent(
            model_path=model_path,
            auto_merge=auto_merge,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )
    spectator_agent_key = key
    return spectator_agent


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
    normalized_clicks = clicks if isinstance(clicks, list) else []
    for idx, click in enumerate(normalized_clicks):
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


def _format_clicks(clicks, obs: np.ndarray):
    rows = []
    for idx, click in enumerate(clicks if isinstance(clicks, list) else []):
        if not isinstance(click, (list, tuple)) or len(click) != 2:
            continue
        px, py = _norm_to_pixel(obs, float(click[0]), float(click[1]))
        rows.append(f"{idx + 1}. norm=({float(click[0]):.1f}, {float(click[1]):.1f}), pixel=({px}, {py})")
    return "\n".join(rows) if rows else "(no valid projected clicks)"


def reset_env():
    global current_clicks, current_obs, done
    current_clicks = []
    done = False

    # Use a random seed for variety
    seed = np.random.randint(0, 100000)
    current_obs, info = env.reset(seed=seed)

    msg = ui(
        "Game started! You are Player 0 (Landlord).\n"
        "Instruction: Click on the image to select cards (or PASS/PLAY buttons). "
        "Your clicks will be recorded as normalized coordinates (1-1000). "
        "When ready, click 'Submit Clicks to Env'.",
        "游戏开始！你是玩家 0（地主）。\n"
        "操作说明：点击图像选择手牌，或点击“出牌/不要”按钮。"
        "点击会记录为 1-1000 的归一化坐标。"
        "准备好后点击“提交点击到环境”。",
    )

    return current_obs, msg, "[]", ""

def handle_click(evt: gr.SelectData):
    """Translates the Gradio image click into normalized coordinates (1-1000)"""
    global current_clicks, current_obs, done
    if done:
        return ui("Game is over. Please click 'Reset Game'.", "游戏已结束，请点击“重置游戏”。"), json.dumps(current_clicks)

    width = current_obs.shape[1]
    height = current_obs.shape[0]
    px, py = evt.index
    
    # Inverse of norm_to_pixel: px = x / 1000.0 * (width - 1)
    norm_x = (px / max(1, width - 1)) * 1000.0
    norm_y = (py / max(1, height - 1)) * 1000.0
    
    # Clamp to [1.0, 1000.0] as expected by the environment
    norm_x = round(max(1.0, min(1000.0, norm_x)), 2)
    norm_y = round(max(1.0, min(1000.0, norm_y)), 2)
    
    current_clicks.append([norm_x, norm_y])
    
    msg = ui(
        f"Added click at ({norm_x}, {norm_y}). Total clicks pending: {len(current_clicks)}",
        f"已添加点击 ({norm_x}, {norm_y})。待提交点击数：{len(current_clicks)}",
    )
    return msg, json.dumps(current_clicks)

def step_env(chat, memory):
    """Simulates the RL Agent taking a step with the accumulated clicks"""
    global current_clicks, current_obs, done
    if done:
        return current_obs, ui("Game over. Please reset.", "游戏已结束，请重置。"), json.dumps([]), memory
    
    # Construct the identical JSON structure an LLM would output
    action = {
        "clicks": current_clicks,
        "projection_valid": 1.0,  # Simulate perfect XML projection valid
        "chat": chat,
        "memory": memory,
    }
    
    current_obs, reward, done, info = env.step(action)
    
    fallback = info.get("fallback_used", False)
    msg = ui(
        f"Reward: {reward:.3f} | Valid Clicks Ratio: {info.get('click_valid_ratio', 0.0):.2f}\n"
        f"Game Action Parsed: {info.get('game_action')} ",
        f"奖励：{reward:.3f} | 有效点击比例：{info.get('click_valid_ratio', 0.0):.2f}\n"
        f"解析出的游戏动作：{info.get('game_action')} ",
    )
    
    if fallback:
        msg += ui("(FALLBACK TRIGGERED! Invalid Move)\n", "（触发兜底动作！无效出牌）\n")
    else:
        msg += ui("(Valid Move!)\n", "（有效出牌）\n")
    
    if done:
        won = bool(info.get("won", 0))
        msg += ui(
            f"\nGame Over! You {'WON' if won else 'LOST'}",
            f"\n游戏结束！你{'赢了' if won else '输了'}",
        )
        
    current_clicks = []
    return current_obs, msg, "[]", info.get("memory", memory)

def clear_clicks():
    global current_clicks
    current_clicks = []
    return ui("Clicks cleared.", "已清空点击。"), "[]"


def reset_spectator(seed_value):
    global current_clicks, current_obs, done, spectator_memory, last_raw_response
    current_clicks = []
    done = False
    spectator_memory = INITIAL_MEMORY
    last_raw_response = ""
    if seed_value is None or seed_value == "":
        seed = int(np.random.randint(0, 100000))
    else:
        seed = int(seed_value)
    current_obs, info = env.reset(seed=seed)
    status = ui(
        f"Spectator game reset with seed={seed}.\n"
        "Load the model if needed, then click Agent Step or Auto-play.",
        f"旁观游戏已重置，seed={seed}。\n"
        "如有需要请先加载模型，然后点击“Agent 单步”或“自动运行”。",
    )
    return current_obs, current_obs, status, "{}", spectator_memory


def load_spectator_model(model_backend, model_path, auto_merge, device_map, torch_dtype, api_base_url, api_model, api_key_env, api_timeout, api_thinking):
    try:
        agent = _load_spectator_agent(
            model_backend,
            model_path,
            bool(auto_merge),
            device_map,
            torch_dtype,
            api_base_url,
            api_model,
            api_key_env,
            api_timeout,
            api_thinking,
        )
        if model_backend == "api":
            return ui(
                f"API backend configured:\n{agent.model} @ {_api_chat_url(agent.base_url)}\nAPI key env: {agent.api_key_env}\nthinking: {agent.thinking}",
                f"API 后端已配置：\n{agent.model} @ {_api_chat_url(agent.base_url)}\nAPI key 环境变量：{agent.api_key_env}\nthinking：{agent.thinking}",
            )
        return ui(f"Model loaded from:\n{agent.model_dir}", f"模型已加载：\n{agent.model_dir}")
    except Exception as exc:
        return ui(
            f"Model load failed:\n{type(exc).__name__}: {exc}",
            f"模型加载失败：\n{type(exc).__name__}: {exc}",
        )


def _spectator_step_core(
    model_backend,
    model_path,
    auto_merge,
    device_map,
    torch_dtype,
    api_base_url,
    api_model,
    api_key_env,
    api_timeout,
    api_thinking,
    max_new_tokens,
    temperature,
    top_p,
    enable_thinking,
):
    global current_obs, done, spectator_memory, last_raw_response
    if current_obs is None:
        current_obs, _info = env.reset(seed=int(np.random.randint(0, 100000)))
        done = False
    if done:
        return (
            current_obs,
            current_obs,
            ui("Game over. Reset spectator game to continue.", "游戏已结束。请重置旁观游戏后继续。"),
            "{}",
            spectator_memory,
        )

    obs_before = current_obs.copy()
    agent = _load_spectator_agent(
        model_backend,
        model_path,
        bool(auto_merge),
        device_map,
        torch_dtype,
        api_base_url,
        api_model,
        api_key_env,
        api_timeout,
        api_thinking,
    )
    response, action = agent.generate_action(
        obs_before,
        spectator_memory,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        enable_thinking=bool(enable_thinking),
    )
    last_raw_response = response
    overlay = annotate_clicks(obs_before, action.get("clicks", []))
    current_obs, reward, done, info = env.step(action)
    if isinstance(action.get("memory"), str) and action["memory"].strip():
        spectator_memory = action["memory"].strip()
    else:
        spectator_memory = info.get("memory", spectator_memory)

    fallback = bool(info.get("fallback_used", False))
    result = ui("FALLBACK: invalid game move", "兜底动作：无效出牌") if fallback else ui("Valid game move", "有效出牌")
    status = ui(
        f"Reward: {reward:.3f} | Projection valid: {action.get('projection_valid', 0)} | "
        f"Click valid ratio: {info.get('click_valid_ratio', 0.0):.2f}\n"
        f"Parsed game action: {info.get('game_action')} | {result}\n"
        f"Submit kind: {info.get('submit_kind')} | Selected cards: {info.get('selected_cards') or '-'}\n\n"
        f"Click positions:\n{_format_clicks(action.get('clicks', []), obs_before)}\n\n"
        f"Chat: {action.get('chat', '')}\n"
        f"Memory: {spectator_memory}",
        f"奖励：{reward:.3f} | 标签解析有效：{action.get('projection_valid', 0)} | "
        f"有效点击比例：{info.get('click_valid_ratio', 0.0):.2f}\n"
        f"解析出的游戏动作：{info.get('game_action')} | {result}\n"
        f"提交类型：{info.get('submit_kind')} | 选中的牌：{info.get('selected_cards') or '-'}\n\n"
        f"点击位置：\n{_format_clicks(action.get('clicks', []), obs_before)}\n\n"
        f"聊天：{action.get('chat', '')}\n"
        f"记忆：{spectator_memory}",
    )
    if done:
        won = bool(info.get("won", 0))
        status += ui(
            f"\n\nGame over. {'Player 0 won.' if won else 'Player 0 lost.'}",
            f"\n\n游戏结束。玩家 0 {'赢了。' if won else '输了。'}",
        )

    action_json = json.dumps(
        {
            "clicks": action.get("clicks", []),
            "projection_valid": action.get("projection_valid", 0),
            "raw_action_text": action.get("raw_action_text", ""),
            "raw_response": response,
        },
        ensure_ascii=False,
        indent=2,
    )
    return current_obs, overlay, status, action_json, spectator_memory


def spectator_step(model_backend, model_path, auto_merge, device_map, torch_dtype, api_base_url, api_model, api_key_env, api_timeout, api_thinking, max_new_tokens, temperature, top_p, enable_thinking):
    try:
        return _spectator_step_core(
            model_backend,
            model_path,
            auto_merge,
            device_map,
            torch_dtype,
            api_base_url,
            api_model,
            api_key_env,
            api_timeout,
            api_thinking,
            max_new_tokens,
            temperature,
            top_p,
            enable_thinking,
        )
    except Exception as exc:
        status = ui(
            f"Agent step failed:\n{type(exc).__name__}: {exc}",
            f"Agent 单步失败：\n{type(exc).__name__}: {exc}",
        )
        fallback_obs = current_obs if current_obs is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        return fallback_obs, fallback_obs, status, "{}", spectator_memory


def spectator_auto_play(
    model_backend,
    model_path,
    auto_merge,
    device_map,
    torch_dtype,
    api_base_url,
    api_model,
    api_key_env,
    api_timeout,
    api_thinking,
    max_new_tokens,
    temperature,
    top_p,
    enable_thinking,
    steps,
    delay_seconds,
):
    total_steps = max(1, int(steps))
    delay = max(0.0, float(delay_seconds))
    for _idx in range(total_steps):
        outputs = spectator_step(
            model_backend,
            model_path,
            auto_merge,
            device_map,
            torch_dtype,
            api_base_url,
            api_model,
            api_key_env,
            api_timeout,
            api_thinking,
            max_new_tokens,
            temperature,
            top_p,
            enable_thinking,
        )
        yield outputs
        if done:
            break
        if delay > 0:
            time.sleep(delay)


def reset_commander(seed_value):
    global current_clicks, current_obs, done, last_raw_response
    current_clicks = []
    done = False
    last_raw_response = ""
    if seed_value is None or seed_value == "":
        seed = int(np.random.randint(0, 100000))
    else:
        seed = int(seed_value)
    current_obs, _info = env.reset(seed=seed)
    status = ui(
        f"Commander game reset with seed={seed}.\n"
        "Enter a semantic card command, then click Execute Command.",
        f"指挥游戏已重置，seed={seed}。\n"
        "输入语义出牌指令，然后点击“执行指令”。",
    )
    return current_obs, current_obs, status, "{}"


def initialize_pages():
    global spectator_memory, last_raw_response, watch_commander_obs, watch_commander_done
    human_obs, human_status, clicks_json, human_memory = reset_env()
    spectator_memory = INITIAL_MEMORY
    last_raw_response = ""
    watch_commander_done = False
    watch_commander_obs, watch_info = watch_commander_env.reset(seed=int(np.random.randint(0, 100000)))
    watch_command = watch_info.get("target_action_pretty") or watch_info.get("target_action") or ""
    spectator_status = ui(
        "Game initialized. Load the model if needed, then click Agent Step or Auto-play.",
        "游戏已初始化。如有需要请先加载模型，然后点击“Agent 单步”或“自动运行”。",
    )
    commander_status = ui(
        "Game initialized. Enter a semantic card command, then click Execute Command.",
        "游戏已初始化。输入语义出牌指令，然后点击“执行指令”。",
    )
    watch_commander_status = ui(
        f"Game initialized. Rule command: {watch_command}\n"
        "Load the model if needed, then click Model Step or Auto-watch.",
        f"游戏已初始化。Rule 指挥动作：{watch_command}\n"
        "如有需要请先加载模型，然后点击“模型单步”或“自动观看”。",
    )
    return (
        human_obs,
        human_status,
        clicks_json,
        human_memory,
        human_obs,
        human_obs,
        spectator_status,
        "{}",
        spectator_memory,
        human_obs,
        human_obs,
        commander_status,
        "{}",
        watch_commander_obs,
        watch_commander_obs,
        watch_commander_status,
        "{}",
    )


def _commander_step_core(
    model_backend,
    model_path,
    auto_merge,
    device_map,
    torch_dtype,
    api_base_url,
    api_model,
    api_key_env,
    api_timeout,
    api_thinking,
    max_new_tokens,
    temperature,
    top_p,
    enable_thinking,
    command,
):
    global current_obs, done, last_raw_response
    command = (command or "").strip()
    if not command:
        fallback_obs = current_obs if current_obs is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        return (
            fallback_obs,
            fallback_obs,
            ui("Enter a commanded action first.", "请先输入指挥动作。"),
            "{}",
        )
    if current_obs is None:
        current_obs, _info = env.reset(seed=int(np.random.randint(0, 100000)))
        done = False
    if done:
        return (
            current_obs,
            current_obs,
            ui("Game over. Reset commander game to continue.", "游戏已结束。请重置指挥游戏后继续。"),
            "{}",
        )

    obs_before = current_obs.copy()
    agent = _load_spectator_agent(
        model_backend,
        model_path,
        bool(auto_merge),
        device_map,
        torch_dtype,
        api_base_url,
        api_model,
        api_key_env,
        api_timeout,
        api_thinking,
    )
    response, action = agent.generate_commanded_action(
        obs_before,
        command,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        enable_thinking=bool(enable_thinking),
    )
    last_raw_response = response
    overlay = annotate_clicks(obs_before, action.get("clicks", []))
    current_obs, reward, done, info = env.step(action)

    fallback = bool(info.get("fallback_used", False))
    result = ui("FALLBACK: invalid game move", "兜底动作：无效出牌") if fallback else ui("Valid game move", "有效出牌")
    status = ui(
        f"Commanded action: {command}\n"
        f"Reward: {reward:.3f} | Projection valid: {action.get('projection_valid', 0)} | "
        f"Click valid ratio: {info.get('click_valid_ratio', 0.0):.2f}\n"
        f"Parsed game action: {info.get('game_action')} | {result}\n"
        f"Submit kind: {info.get('submit_kind')} | Selected cards: {info.get('selected_cards') or '-'}\n\n"
        f"Click positions:\n{_format_clicks(action.get('clicks', []), obs_before)}",
        f"指挥动作：{command}\n"
        f"奖励：{reward:.3f} | 标签解析有效：{action.get('projection_valid', 0)} | "
        f"有效点击比例：{info.get('click_valid_ratio', 0.0):.2f}\n"
        f"解析出的游戏动作：{info.get('game_action')} | {result}\n"
        f"提交类型：{info.get('submit_kind')} | 选中的牌：{info.get('selected_cards') or '-'}\n\n"
        f"点击位置：\n{_format_clicks(action.get('clicks', []), obs_before)}",
    )
    if done:
        won = bool(info.get("won", 0))
        status += ui(
            f"\n\nGame over. {'Player 0 won.' if won else 'Player 0 lost.'}",
            f"\n\n游戏结束。玩家 0 {'赢了。' if won else '输了。'}",
        )

    action_json = json.dumps(
        {
            "commanded_action": command,
            "clicks": action.get("clicks", []),
            "projection_valid": action.get("projection_valid", 0),
            "raw_tool_call_text": action.get("raw_tool_call_text", ""),
            "raw_response": response,
        },
        ensure_ascii=False,
        indent=2,
    )
    return current_obs, overlay, status, action_json


def commander_step(model_backend, model_path, auto_merge, device_map, torch_dtype, api_base_url, api_model, api_key_env, api_timeout, api_thinking, max_new_tokens, temperature, top_p, enable_thinking, command):
    try:
        return _commander_step_core(
            model_backend,
            model_path,
            auto_merge,
            device_map,
            torch_dtype,
            api_base_url,
            api_model,
            api_key_env,
            api_timeout,
            api_thinking,
            max_new_tokens,
            temperature,
            top_p,
            enable_thinking,
            command,
        )
    except Exception as exc:
        status = ui(
            f"Commander step failed:\n{type(exc).__name__}: {exc}",
            f"指挥单步失败：\n{type(exc).__name__}: {exc}",
        )
        fallback_obs = current_obs if current_obs is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        return fallback_obs, fallback_obs, status, "{}"


def reset_watch_commander(seed_value):
    global watch_commander_obs, watch_commander_done, last_raw_response
    watch_commander_done = False
    last_raw_response = ""
    if seed_value is None or seed_value == "":
        seed = int(np.random.randint(0, 100000))
    else:
        seed = int(seed_value)
    watch_commander_obs, info = watch_commander_env.reset(seed=seed)
    command = info.get("target_action_pretty") or info.get("target_action") or ""
    status = ui(
        f"Watch commander game reset with seed={seed}.\n"
        f"Rule command: {command}\n"
        "Click Model Step to let the model execute the rule command.",
        f"观看指挥游戏已重置，seed={seed}。\n"
        f"Rule 指挥动作：{command}\n"
        "点击“模型单步”让模型执行 rule 指挥。",
    )
    return watch_commander_obs, watch_commander_obs, status, "{}"


def _watch_commander_step_core(
    model_backend,
    model_path,
    auto_merge,
    device_map,
    torch_dtype,
    api_base_url,
    api_model,
    api_key_env,
    api_timeout,
    api_thinking,
    max_new_tokens,
    temperature,
    top_p,
    enable_thinking,
):
    global watch_commander_obs, watch_commander_done, last_raw_response
    if watch_commander_obs is None:
        watch_commander_obs, _info = watch_commander_env.reset(seed=int(np.random.randint(0, 100000)))
        watch_commander_done = False
    if watch_commander_done or watch_commander_env.done:
        watch_commander_done = True
        return (
            watch_commander_obs,
            watch_commander_obs,
            ui("Game over. Reset watch commander game to continue.", "游戏已结束。请重置观看指挥游戏后继续。"),
            "{}",
        )

    obs_before = watch_commander_obs.copy()
    target_action = watch_commander_env.target_action
    command = watch_commander_env._pretty_action(target_action)
    agent = _load_spectator_agent(
        model_backend,
        model_path,
        bool(auto_merge),
        device_map,
        torch_dtype,
        api_base_url,
        api_model,
        api_key_env,
        api_timeout,
        api_thinking,
    )
    response, action = agent.generate_commanded_action(
        obs_before,
        command,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        enable_thinking=bool(enable_thinking),
    )
    last_raw_response = response
    overlay = annotate_clicks(obs_before, action.get("clicks", []))

    rewards, _dones, infos = watch_commander_env.score_group([action])
    reward = float(rewards[0])
    info = infos[0]
    next_obs, next_info = watch_commander_env.advance_teacher()
    watch_commander_obs = next_obs
    watch_commander_done = bool(watch_commander_env.done)
    next_command = next_info.get("target_action_pretty") or next_info.get("target_action") or ""

    matched = bool(info.get("target_action_match", 0.0))
    result = ui("Matched rule command", "已匹配 rule 指挥") if matched else ui("Did not match rule command", "未匹配 rule 指挥")
    status = ui(
        f"Rule command: {command}\n"
        f"Model parsed action: {watch_commander_env._pretty_action(info.get('predicted_action'))}\n"
        f"Reward: {reward:.3f} | Projection valid: {action.get('projection_valid', 0)} | "
        f"Click valid ratio: {info.get('click_valid_ratio', 0.0):.2f} | "
        f"Submit correct: {info.get('submit_correct', 0.0):.0f} | {result}\n"
        f"Selected cards: {watch_commander_env._pretty_action(info.get('selected_cards')) or '-'} | Submit kind: {info.get('submit_kind')}\n"
        f"Teacher advanced the game with: {command}\n"
        f"Next rule command: {next_command or '-'}\n\n"
        f"Click positions:\n{_format_clicks(action.get('clicks', []), obs_before)}",
        f"Rule 指挥动作：{command}\n"
        f"模型解析动作：{watch_commander_env._pretty_action(info.get('predicted_action'))}\n"
        f"奖励：{reward:.3f} | 标签解析有效：{action.get('projection_valid', 0)} | "
        f"有效点击比例：{info.get('click_valid_ratio', 0.0):.2f} | "
        f"提交正确：{info.get('submit_correct', 0.0):.0f} | {result}\n"
        f"选中的牌：{watch_commander_env._pretty_action(info.get('selected_cards')) or '-'} | 提交类型：{info.get('submit_kind')}\n"
        f"底层牌局已按 teacher 动作推进：{command}\n"
        f"下一条 rule 指挥：{next_command or '-'}\n\n"
        f"点击位置：\n{_format_clicks(action.get('clicks', []), obs_before)}",
    )
    if watch_commander_done:
        won = bool(next_info.get("won", 0))
        status += ui(
            f"\n\nGame over after teacher advance. {'Player 0 won.' if won else 'Player 0 lost.'}",
            f"\n\nTeacher 推进后游戏结束。玩家 0 {'赢了。' if won else '输了。'}",
        )

    action_json = json.dumps(
        {
            "rule_command": command,
            "model_predicted_action": watch_commander_env._pretty_action(info.get("predicted_action")),
            "target_action_match": info.get("target_action_match", 0.0),
            "click_valid_ratio": info.get("click_valid_ratio", 0.0),
            "submit_correct": info.get("submit_correct", 0.0),
            "clicks": action.get("clicks", []),
            "projection_valid": action.get("projection_valid", 0),
            "raw_tool_call_text": action.get("raw_tool_call_text", ""),
            "raw_response": response,
        },
        ensure_ascii=False,
        indent=2,
    )
    return watch_commander_obs, overlay, status, action_json


def watch_commander_step(model_backend, model_path, auto_merge, device_map, torch_dtype, api_base_url, api_model, api_key_env, api_timeout, api_thinking, max_new_tokens, temperature, top_p, enable_thinking):
    try:
        return _watch_commander_step_core(
            model_backend,
            model_path,
            auto_merge,
            device_map,
            torch_dtype,
            api_base_url,
            api_model,
            api_key_env,
            api_timeout,
            api_thinking,
            max_new_tokens,
            temperature,
            top_p,
            enable_thinking,
        )
    except Exception as exc:
        status = ui(
            f"Watch commander step failed:\n{type(exc).__name__}: {exc}",
            f"观看指挥单步失败：\n{type(exc).__name__}: {exc}",
        )
        fallback_obs = watch_commander_obs if watch_commander_obs is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        return fallback_obs, fallback_obs, status, "{}"


def watch_commander_auto_play(
    model_backend,
    model_path,
    auto_merge,
    device_map,
    torch_dtype,
    api_base_url,
    api_model,
    api_key_env,
    api_timeout,
    api_thinking,
    max_new_tokens,
    temperature,
    top_p,
    enable_thinking,
    steps,
    delay_seconds,
):
    total_steps = max(1, int(steps))
    delay = max(0.0, float(delay_seconds))
    for _idx in range(total_steps):
        outputs = watch_commander_step(
            model_backend,
            model_path,
            auto_merge,
            device_map,
            torch_dtype,
            api_base_url,
            api_model,
            api_key_env,
            api_timeout,
            api_thinking,
            max_new_tokens,
            temperature,
            top_p,
            enable_thinking,
        )
        yield outputs
        if watch_commander_done:
            break
        if delay > 0:
            time.sleep(delay)


with gr.Blocks(title=ui("Doudizhu Human, Commander, Watch, and Spectator Debugger", "斗地主人工、指挥、观看与旁观调试器"), theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 斗地主 (Dou Dizhu) Agentic Environment")
    mode_selector = gr.Radio(
        [MODE_HUMAN, MODE_SPECTATOR, MODE_COMMANDER, MODE_WATCH_COMMANDER],
        value=MODE_HUMAN,
        label=ui("Page", "页面"),
    )

    with gr.Group(visible=True) as human_page:
        gr.Markdown(
            ui(
                "Click on the game UI to simulate the LLM's coordinate outputs, then submit them to the environment.",
                "点击游戏界面来模拟 LLM 输出的坐标，然后提交到环境。",
            )
        )
        with gr.Row():
            with gr.Column(scale=2):
                img = gr.Image(interactive=False, label=ui("Environment Observation (Click to select coords)", "环境观察（点击选择坐标）"))

                with gr.Row():
                    clear_btn = gr.Button(ui("Clear Clicks", "清空点击"))
                    step_btn = gr.Button(ui("Submit Clicks to Env", "提交点击到环境"), variant="primary")
                    reset_btn = gr.Button(ui("Reset Game", "重置游戏"))

            with gr.Column(scale=1):
                status_out = gr.Textbox(label=ui("Status / Step Result", "状态 / 单步结果"), lines=6)
                clicks_out = gr.Textbox(label=ui("Current <action> JSON", "当前 <action> JSON"), interactive=False)
                chat_in = gr.Textbox(label="<chat>", placeholder=ui("Say something to the peasants...", "给农民玩家说一句话..."))
                memory_in = gr.Textbox(label="<memory>", placeholder=ui("Write a note to yourself for the next turn...", "给下一轮写一条记忆..."))

    with gr.Group(visible=False) as spectator_page:
        gr.Markdown(
            ui(
                "Watch a trained VL agent play. The right image marks the exact normalized click coordinates on the pre-action game screen.",
                "观看训练好的 VL agent 出牌。右侧图像会标出动作执行前屏幕上的归一化点击坐标。",
            )
        )
        with gr.Row():
            with gr.Column(scale=2):
                spectator_img = gr.Image(interactive=False, label=ui("Current Observation After Agent Move", "Agent 动作后的当前观察"))
                spectator_overlay = gr.Image(interactive=False, label=ui("Last Agent Click Overlay", "上一轮 Agent 点击标注"))
            with gr.Column(scale=1):
                model_backend_in = gr.Dropdown(["local", "api"], label=ui("Model Backend", "模型后端"), value=ARGS.model_backend)
                model_path_in = gr.Textbox(label=ui("Model / Checkpoint Path", "模型 / checkpoint 路径"), value=ARGS.model_path)
                auto_merge_in = gr.Checkbox(label=ui("Auto-merge verl FSDP checkpoint if needed", "需要时自动合并 verl FSDP checkpoint"), value=not ARGS.no_auto_merge)
                with gr.Row():
                    device_map_in = gr.Textbox(label="device_map", value=ARGS.device_map)
                    dtype_in = gr.Dropdown(["auto", "float16", "bfloat16", "float32"], label="torch_dtype", value=ARGS.torch_dtype)
                api_base_url_in = gr.Textbox(label=ui("API Base URL (OpenAI-compatible)", "API Base URL（OpenAI 兼容）"), value=ARGS.api_base_url)
                api_model_in = gr.Textbox(label=ui("API Model", "API 模型"), value=ARGS.api_model)
                with gr.Row():
                    api_key_env_in = gr.Textbox(label=ui("API Key Env", "API Key 环境变量"), value=ARGS.api_key_env)
                    api_timeout_in = gr.Number(label=ui("API Timeout", "API 超时"), value=ARGS.api_timeout)
                    api_thinking_in = gr.Dropdown(["default", "enabled", "disabled"], label="API thinking", value=ARGS.api_thinking)
                with gr.Row():
                    max_tokens_in = gr.Number(label="max_new_tokens", value=ARGS.max_new_tokens, precision=0)
                    temp_in = gr.Number(label="temperature", value=ARGS.temperature)
                    top_p_in = gr.Number(label="top_p", value=ARGS.top_p)
                enable_thinking_in = gr.Checkbox(
                    label="enable_thinking",
                    value=bool(ARGS.enable_thinking),
                )
                seed_in = gr.Number(label=ui("Reset Seed (blank = random)", "重置 Seed（留空为随机）"), value=None, precision=0)
                with gr.Row():
                    load_model_btn = gr.Button(ui("Load Model", "加载模型"))
                    spectator_reset_btn = gr.Button(ui("Reset Spectator Game", "重置旁观游戏"))
                with gr.Row():
                    spectator_step_btn = gr.Button(ui("Agent Step", "Agent 单步"), variant="primary")
                    auto_steps_in = gr.Number(label=ui("Auto steps", "自动步数"), value=10, precision=0)
                    delay_in = gr.Number(label=ui("Delay seconds", "延迟秒数"), value=0.8)
                auto_play_btn = gr.Button(ui("Auto-play", "自动运行"))
                spectator_status = gr.Textbox(label=ui("Spectator Status", "旁观状态"), lines=12)
                spectator_action_json = gr.Textbox(label=ui("Projected Action / Raw Response", "投影动作 / 原始响应"), lines=12, interactive=False)
                spectator_memory_out = gr.Textbox(label=ui("Model Memory", "模型记忆"), lines=3, interactive=False)

    with gr.Group(visible=False) as commander_page:
        gr.Markdown(
            ui(
                "Give the semantic card action yourself; the model only converts that command into left_click(...) GUI clicks.",
                "由人输入语义出牌动作；模型只负责把该指令转换成 left_click(...) GUI 点击。",
            )
        )
        with gr.Row():
            with gr.Column(scale=2):
                commander_img = gr.Image(interactive=False, label=ui("Current Observation After Command", "指令执行后的当前观察"))
                commander_overlay = gr.Image(interactive=False, label=ui("Last Command Click Overlay", "上一条指令点击标注"))
            with gr.Column(scale=1):
                commander_model_backend_in = gr.Dropdown(["local", "api"], label=ui("Model Backend", "模型后端"), value=ARGS.model_backend)
                commander_model_path_in = gr.Textbox(label=ui("Model / Checkpoint Path", "模型 / checkpoint 路径"), value=ARGS.model_path)
                commander_auto_merge_in = gr.Checkbox(label=ui("Auto-merge verl FSDP checkpoint if needed", "需要时自动合并 verl FSDP checkpoint"), value=not ARGS.no_auto_merge)
                with gr.Row():
                    commander_device_map_in = gr.Textbox(label="device_map", value=ARGS.device_map)
                    commander_dtype_in = gr.Dropdown(["auto", "float16", "bfloat16", "float32"], label="torch_dtype", value=ARGS.torch_dtype)
                commander_api_base_url_in = gr.Textbox(label=ui("API Base URL (OpenAI-compatible)", "API Base URL（OpenAI 兼容）"), value=ARGS.api_base_url)
                commander_api_model_in = gr.Textbox(label=ui("API Model", "API 模型"), value=ARGS.api_model)
                with gr.Row():
                    commander_api_key_env_in = gr.Textbox(label=ui("API Key Env", "API Key 环境变量"), value=ARGS.api_key_env)
                    commander_api_timeout_in = gr.Number(label=ui("API Timeout", "API 超时"), value=ARGS.api_timeout)
                    commander_api_thinking_in = gr.Dropdown(["default", "enabled", "disabled"], label="API thinking", value=ARGS.api_thinking)
                with gr.Row():
                    commander_max_tokens_in = gr.Number(label="max_new_tokens", value=min(512, ARGS.max_new_tokens), precision=0)
                    commander_temp_in = gr.Number(label="temperature", value=ARGS.temperature)
                    commander_top_p_in = gr.Number(label="top_p", value=ARGS.top_p)
                commander_enable_thinking_in = gr.Checkbox(
                    label="enable_thinking",
                    value=bool(ARGS.enable_thinking),
                )
                commander_seed_in = gr.Number(label=ui("Reset Seed (blank = random)", "重置 Seed（留空为随机）"), value=None, precision=0)
                commander_command_in = gr.Textbox(
                    label=ui("Commanded Action", "指挥动作"),
                    placeholder=ui("Examples: 3 3, 10 J Q K A, pass", "例如：3 3、10 J Q K A、不要"),
                )
                with gr.Row():
                    commander_load_model_btn = gr.Button(ui("Load Model", "加载模型"))
                    commander_reset_btn = gr.Button(ui("Reset Commander Game", "重置指挥游戏"))
                commander_step_btn = gr.Button(ui("Execute Command", "执行指令"), variant="primary")
                commander_status = gr.Textbox(label=ui("Commander Status", "指挥状态"), lines=12)
                commander_action_json = gr.Textbox(label=ui("Projected tool_call / Raw Response", "投影 tool_call / 原始响应"), lines=12, interactive=False)

    with gr.Group(visible=False) as watch_commander_page:
        gr.Markdown(
            ui(
                "Watch the grounding task: a rule teacher commands the card action, the model only clicks, and the game advances by the teacher action.",
                "观看 grounding 任务：rule teacher 负责指挥出牌，模型只负责点击，底层牌局始终按 teacher 动作推进。",
            )
        )
        with gr.Row():
            with gr.Column(scale=2):
                watch_commander_img = gr.Image(interactive=False, label=ui("Current Observation After Teacher Advance", "Teacher 推进后的当前观察"))
                watch_commander_overlay = gr.Image(interactive=False, label=ui("Last Model Click Overlay", "上一轮模型点击标注"))
            with gr.Column(scale=1):
                watch_model_backend_in = gr.Dropdown(["local", "api"], label=ui("Model Backend", "模型后端"), value=ARGS.model_backend)
                watch_model_path_in = gr.Textbox(label=ui("Model / Checkpoint Path", "模型 / checkpoint 路径"), value=ARGS.model_path)
                watch_auto_merge_in = gr.Checkbox(label=ui("Auto-merge verl FSDP checkpoint if needed", "需要时自动合并 verl FSDP checkpoint"), value=not ARGS.no_auto_merge)
                with gr.Row():
                    watch_device_map_in = gr.Textbox(label="device_map", value=ARGS.device_map)
                    watch_dtype_in = gr.Dropdown(["auto", "float16", "bfloat16", "float32"], label="torch_dtype", value=ARGS.torch_dtype)
                watch_api_base_url_in = gr.Textbox(label=ui("API Base URL (OpenAI-compatible)", "API Base URL（OpenAI 兼容）"), value=ARGS.api_base_url)
                watch_api_model_in = gr.Textbox(label=ui("API Model", "API 模型"), value=ARGS.api_model)
                with gr.Row():
                    watch_api_key_env_in = gr.Textbox(label=ui("API Key Env", "API Key 环境变量"), value=ARGS.api_key_env)
                    watch_api_timeout_in = gr.Number(label=ui("API Timeout", "API 超时"), value=ARGS.api_timeout)
                    watch_api_thinking_in = gr.Dropdown(["default", "enabled", "disabled"], label="API thinking", value=ARGS.api_thinking)
                with gr.Row():
                    watch_max_tokens_in = gr.Number(label="max_new_tokens", value=min(512, ARGS.max_new_tokens), precision=0)
                    watch_temp_in = gr.Number(label="temperature", value=ARGS.temperature)
                    watch_top_p_in = gr.Number(label="top_p", value=ARGS.top_p)
                watch_enable_thinking_in = gr.Checkbox(
                    label="enable_thinking",
                    value=bool(ARGS.enable_thinking),
                )
                watch_seed_in = gr.Number(label=ui("Reset Seed (blank = random)", "重置 Seed（留空为随机）"), value=None, precision=0)
                with gr.Row():
                    watch_load_model_btn = gr.Button(ui("Load Model", "加载模型"))
                    watch_reset_btn = gr.Button(ui("Reset Watch Game", "重置观看游戏"))
                with gr.Row():
                    watch_step_btn = gr.Button(ui("Model Step", "模型单步"), variant="primary")
                    watch_auto_steps_in = gr.Number(label=ui("Auto steps", "自动步数"), value=10, precision=0)
                    watch_delay_in = gr.Number(label=ui("Delay seconds", "延迟秒数"), value=0.8)
                watch_auto_btn = gr.Button(ui("Auto-watch", "自动观看"))
                watch_status = gr.Textbox(label=ui("Watch Commander Status", "观看指挥状态"), lines=14)
                watch_action_json = gr.Textbox(label=ui("Grounding Action / Raw Response", "Grounding 动作 / 原始响应"), lines=12, interactive=False)

    mode_selector.change(
        switch_mode,
        inputs=[mode_selector],
        outputs=[human_page, spectator_page, commander_page, watch_commander_page],
    )

    img.select(handle_click, outputs=[status_out, clicks_out])
    clear_btn.click(clear_clicks, outputs=[status_out, clicks_out])
    reset_btn.click(reset_env, outputs=[img, status_out, clicks_out, memory_in])
    step_btn.click(step_env, inputs=[chat_in, memory_in], outputs=[img, status_out, clicks_out, memory_in])
    demo.load(
        initialize_pages,
        outputs=[
            img,
            status_out,
            clicks_out,
            memory_in,
            spectator_img,
            spectator_overlay,
            spectator_status,
            spectator_action_json,
            spectator_memory_out,
            commander_img,
            commander_overlay,
            commander_status,
            commander_action_json,
            watch_commander_img,
            watch_commander_overlay,
            watch_status,
            watch_action_json,
        ],
    )

    spectator_reset_btn.click(
        reset_spectator,
        inputs=[seed_in],
        outputs=[spectator_img, spectator_overlay, spectator_status, spectator_action_json, spectator_memory_out],
    )
    load_model_btn.click(
        load_spectator_model,
        inputs=[model_backend_in, model_path_in, auto_merge_in, device_map_in, dtype_in, api_base_url_in, api_model_in, api_key_env_in, api_timeout_in, api_thinking_in],
        outputs=[spectator_status],
    )
    spectator_step_btn.click(
        spectator_step,
        inputs=[model_backend_in, model_path_in, auto_merge_in, device_map_in, dtype_in, api_base_url_in, api_model_in, api_key_env_in, api_timeout_in, api_thinking_in, max_tokens_in, temp_in, top_p_in, enable_thinking_in],
        outputs=[spectator_img, spectator_overlay, spectator_status, spectator_action_json, spectator_memory_out],
    )
    auto_play_btn.click(
        spectator_auto_play,
        inputs=[
            model_backend_in,
            model_path_in,
            auto_merge_in,
            device_map_in,
            dtype_in,
            api_base_url_in,
            api_model_in,
            api_key_env_in,
            api_timeout_in,
            api_thinking_in,
            max_tokens_in,
            temp_in,
            top_p_in,
            enable_thinking_in,
            auto_steps_in,
            delay_in,
        ],
        outputs=[spectator_img, spectator_overlay, spectator_status, spectator_action_json, spectator_memory_out],
    )

    commander_reset_btn.click(
        reset_commander,
        inputs=[commander_seed_in],
        outputs=[commander_img, commander_overlay, commander_status, commander_action_json],
    )
    commander_load_model_btn.click(
        load_spectator_model,
        inputs=[
            commander_model_backend_in,
            commander_model_path_in,
            commander_auto_merge_in,
            commander_device_map_in,
            commander_dtype_in,
            commander_api_base_url_in,
            commander_api_model_in,
            commander_api_key_env_in,
            commander_api_timeout_in,
            commander_api_thinking_in,
        ],
        outputs=[commander_status],
    )
    commander_step_btn.click(
        commander_step,
        inputs=[
            commander_model_backend_in,
            commander_model_path_in,
            commander_auto_merge_in,
            commander_device_map_in,
            commander_dtype_in,
            commander_api_base_url_in,
            commander_api_model_in,
            commander_api_key_env_in,
            commander_api_timeout_in,
            commander_api_thinking_in,
            commander_max_tokens_in,
            commander_temp_in,
            commander_top_p_in,
            commander_enable_thinking_in,
            commander_command_in,
        ],
        outputs=[commander_img, commander_overlay, commander_status, commander_action_json],
    )

    watch_reset_btn.click(
        reset_watch_commander,
        inputs=[watch_seed_in],
        outputs=[watch_commander_img, watch_commander_overlay, watch_status, watch_action_json],
    )
    watch_load_model_btn.click(
        load_spectator_model,
        inputs=[
            watch_model_backend_in,
            watch_model_path_in,
            watch_auto_merge_in,
            watch_device_map_in,
            watch_dtype_in,
            watch_api_base_url_in,
            watch_api_model_in,
            watch_api_key_env_in,
            watch_api_timeout_in,
            watch_api_thinking_in,
        ],
        outputs=[watch_status],
    )
    watch_step_btn.click(
        watch_commander_step,
        inputs=[
            watch_model_backend_in,
            watch_model_path_in,
            watch_auto_merge_in,
            watch_device_map_in,
            watch_dtype_in,
            watch_api_base_url_in,
            watch_api_model_in,
            watch_api_key_env_in,
            watch_api_timeout_in,
            watch_api_thinking_in,
            watch_max_tokens_in,
            watch_temp_in,
            watch_top_p_in,
            watch_enable_thinking_in,
        ],
        outputs=[watch_commander_img, watch_commander_overlay, watch_status, watch_action_json],
    )
    watch_auto_btn.click(
        watch_commander_auto_play,
        inputs=[
            watch_model_backend_in,
            watch_model_path_in,
            watch_auto_merge_in,
            watch_device_map_in,
            watch_dtype_in,
            watch_api_base_url_in,
            watch_api_model_in,
            watch_api_key_env_in,
            watch_api_timeout_in,
            watch_api_thinking_in,
            watch_max_tokens_in,
            watch_temp_in,
            watch_top_p_in,
            watch_enable_thinking_in,
            watch_auto_steps_in,
            watch_delay_in,
        ],
        outputs=[watch_commander_img, watch_commander_overlay, watch_status, watch_action_json],
    )

if __name__ == "__main__":
    print("Starting Gradio server...")
    # Bind to 0.0.0.0 so it can be accessed remotely
    demo.launch(server_name=ARGS.server_name, server_port=ARGS.server_port, share=False)
