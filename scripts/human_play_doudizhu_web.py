import os
import sys
import json
import time
import argparse
from pathlib import Path
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
    from agent_system.environments.env_package.doudizhu.projection import doudizhu_projection
    from agent_system.environments.prompts.doudizhu import DOUDIZHU_VISUAL_TEMPLATE
except ImportError as e:
    print(f"Failed to import DoudizhuSingleEnv: {e}")
    print("Make sure you run this script from the project root or the conda environment is active.")
    sys.exit(1)

DEFAULT_MODEL_PATH = "checkpoints/verl_agent_doudizhu/grpo_qwen3_vl_4b/global_step_40"
INITIAL_MEMORY = "Initial turn. Read the screenshot, identify your hand, and plan the first landlord play."


def parse_args():
    parser = argparse.ArgumentParser(description="Dou Dizhu human/debug and model spectator web UI.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Merged HF model dir, actor dir, or verl global_step dir.")
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--device-map", default="auto", help="Transformers device_map for model loading.")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--no-auto-merge", action="store_true", help="Do not auto-merge verl FSDP actor checkpoints.")
    return parser.parse_args()


ARGS = parse_args()

# Global state to track human interactions
current_clicks = []
current_obs = None
done = False
env = DoudizhuSingleEnv(seed=42)
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


class DoudizhuSpectatorAgent:
    def __init__(self, model_path: str, auto_merge: bool, device_map: str, torch_dtype: str):
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer

        self.model_dir = _resolve_model_dir(model_path, auto_merge=auto_merge)
        self.processor = AutoProcessor.from_pretrained(self.model_dir, trust_remote_code=True)
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_dir,
            torch_dtype=_torch_dtype(torch_dtype),
            device_map=device_map if device_map else None,
            trust_remote_code=True,
        )
        self.model.eval()
        self.torch = torch

    def generate_action(self, obs: np.ndarray, memory: str, max_new_tokens: int, temperature: float, top_p: float):
        prompt = DOUDIZHU_VISUAL_TEMPLATE.format(previous_memory=memory or "No previous memory.")
        chat = [{"role": "user", "content": prompt}]
        prompt_text = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
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
        actions, _valids = doudizhu_projection([response], max_clicks=env.max_clicks)
        return response, actions[0]


def _load_spectator_agent(model_path: str, auto_merge: bool, device_map: str, torch_dtype: str):
    global spectator_agent, spectator_agent_key
    key = (model_path, bool(auto_merge), device_map, torch_dtype)
    if spectator_agent is not None and spectator_agent_key == key:
        return spectator_agent
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
    small_font = _load_overlay_font(12)
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
    
    msg = "🎮 Game started! You are Player 0 (Landlord).\n"
    msg += "👉 Instruction: Click on the image to select cards (or PASS/PLAY buttons). "
    msg += "Your clicks will be recorded as normalized coordinates (1-1000). "
    msg += "When ready, click 'Submit Clicks to Env'."
    
    return current_obs, msg, "[]", ""

def handle_click(evt: gr.SelectData):
    """Translates the Gradio image click into normalized coordinates (1-1000)"""
    global current_clicks, current_obs, done
    if done:
        return "Game is over. Please click 'Reset Game'.", json.dumps(current_clicks)
    
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
    
    msg = f"📍 Added click at ({norm_x}, {norm_y}). Total clicks pending: {len(current_clicks)}"
    return msg, json.dumps(current_clicks)

def step_env(chat, memory):
    """Simulates the RL Agent taking a step with the accumulated clicks"""
    global current_clicks, current_obs, done
    if done:
        return current_obs, "Game over. Please reset.", json.dumps([]), memory
    
    # Construct the identical JSON structure an LLM would output
    action = {
        "clicks": current_clicks,
        "projection_valid": 1.0,  # Simulate perfect XML projection valid
        "chat": chat,
        "memory": memory
    }
    
    current_obs, reward, done, info = env.step(action)
    
    fallback = info.get("fallback_used", False)
    msg = f"💰 Reward: {reward:.3f} | 🎯 Valid Clicks Ratio: {info.get('click_valid_ratio', 0.0):.2f}\n"
    msg += f"🃏 Game Action Parsed: {info.get('game_action')} "
    
    if fallback:
         msg += "❌ (FALLBACK TRIGGERED! Invalid Move)\n"
    else:
         msg += "✅ (Valid Move!)\n"
    
    if done:
        won = bool(info.get('won', 0))
        msg += f"\n🏁 Game Over! You {'WON 🏆' if won else 'LOST 💀'}"
        
    current_clicks = [] # Reset clicks after submission
    return current_obs, msg, "[]", info.get('memory', memory)

def clear_clicks():
    global current_clicks
    current_clicks = []
    return "🧹 Clicks cleared.", "[]"


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
    status = (
        f"Spectator game reset with seed={seed}.\n"
        "Load the model if needed, then click Agent Step or Auto-play."
    )
    return current_obs, current_obs, status, "{}", spectator_memory


def load_spectator_model(model_path, auto_merge, device_map, torch_dtype):
    try:
        agent = _load_spectator_agent(model_path, bool(auto_merge), device_map, torch_dtype)
        return f"Model loaded from:\n{agent.model_dir}"
    except Exception as exc:
        return f"Model load failed:\n{type(exc).__name__}: {exc}"


def _spectator_step_core(model_path, auto_merge, device_map, torch_dtype, max_new_tokens, temperature, top_p):
    global current_obs, done, spectator_memory, last_raw_response
    if current_obs is None:
        current_obs, _info = env.reset(seed=int(np.random.randint(0, 100000)))
        done = False
    if done:
        return current_obs, current_obs, "Game over. Reset spectator game to continue.", "{}", spectator_memory

    obs_before = current_obs.copy()
    agent = _load_spectator_agent(model_path, bool(auto_merge), device_map, torch_dtype)
    response, action = agent.generate_action(
        obs_before,
        spectator_memory,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
    )
    last_raw_response = response
    overlay = annotate_clicks(obs_before, action.get("clicks", []))
    current_obs, reward, done, info = env.step(action)
    if isinstance(action.get("memory"), str) and action["memory"].strip():
        spectator_memory = action["memory"].strip()
    else:
        spectator_memory = info.get("memory", spectator_memory)

    fallback = bool(info.get("fallback_used", False))
    result = "FALLBACK: invalid game move" if fallback else "Valid game move"
    status = (
        f"Reward: {reward:.3f} | Projection valid: {action.get('projection_valid', 0)} | "
        f"Click valid ratio: {info.get('click_valid_ratio', 0.0):.2f}\n"
        f"Parsed game action: {info.get('game_action')} | {result}\n"
        f"Submit kind: {info.get('submit_kind')} | Selected cards: {info.get('selected_cards') or '-'}\n\n"
        f"Click positions:\n{_format_clicks(action.get('clicks', []), obs_before)}\n\n"
        f"Chat: {action.get('chat', '')}\n"
        f"Memory: {spectator_memory}"
    )
    if done:
        won = bool(info.get("won", 0))
        status += f"\n\nGame over. {'Player 0 won.' if won else 'Player 0 lost.'}"

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


def spectator_step(model_path, auto_merge, device_map, torch_dtype, max_new_tokens, temperature, top_p):
    try:
        return _spectator_step_core(model_path, auto_merge, device_map, torch_dtype, max_new_tokens, temperature, top_p)
    except Exception as exc:
        status = f"Agent step failed:\n{type(exc).__name__}: {exc}"
        fallback_obs = current_obs if current_obs is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        return fallback_obs, fallback_obs, status, "{}", spectator_memory


def spectator_auto_play(model_path, auto_merge, device_map, torch_dtype, max_new_tokens, temperature, top_p, steps, delay_seconds):
    total_steps = max(1, int(steps))
    delay = max(0.0, float(delay_seconds))
    for _idx in range(total_steps):
        outputs = spectator_step(model_path, auto_merge, device_map, torch_dtype, max_new_tokens, temperature, top_p)
        yield outputs
        if done:
            break
        if delay > 0:
            time.sleep(delay)


with gr.Blocks(title="Doudizhu Human and Spectator Debugger", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 斗地主 (Dou Dizhu) Agentic Environment")

    with gr.Tab("Human play"):
        gr.Markdown("Click on the game UI to simulate the LLM's coordinate outputs, then submit them to the environment.")
        with gr.Row():
            with gr.Column(scale=2):
                img = gr.Image(interactive=False, label="Environment Observation (Click to select coords)")

                with gr.Row():
                    clear_btn = gr.Button("Clear Clicks")
                    step_btn = gr.Button("Submit Clicks to Env", variant="primary")
                    reset_btn = gr.Button("Reset Game")

            with gr.Column(scale=1):
                status_out = gr.Textbox(label="Status / Step Result", lines=6)
                clicks_out = gr.Textbox(label="Current <action> JSON", interactive=False)
                chat_in = gr.Textbox(label="<chat> Input (Optional)", placeholder="Say something to the peasants...")
                memory_in = gr.Textbox(label="<memory> Input/Output", placeholder="Write a note to yourself for the next turn...")

        img.select(handle_click, outputs=[status_out, clicks_out])
        clear_btn.click(clear_clicks, outputs=[status_out, clicks_out])
        reset_btn.click(reset_env, outputs=[img, status_out, clicks_out, memory_in])
        step_btn.click(step_env, inputs=[chat_in, memory_in], outputs=[img, status_out, clicks_out, memory_in])
        demo.load(reset_env, outputs=[img, status_out, clicks_out, memory_in])

    with gr.Tab("Spectator mode"):
        gr.Markdown("Watch a trained VL agent play. The right image marks the exact normalized click coordinates on the pre-action game screen.")
        with gr.Row():
            with gr.Column(scale=2):
                spectator_img = gr.Image(interactive=False, label="Current Observation After Agent Move")
                spectator_overlay = gr.Image(interactive=False, label="Last Agent Click Overlay")
            with gr.Column(scale=1):
                model_path_in = gr.Textbox(label="Model / Checkpoint Path", value=ARGS.model_path)
                auto_merge_in = gr.Checkbox(label="Auto-merge verl FSDP checkpoint if needed", value=not ARGS.no_auto_merge)
                with gr.Row():
                    device_map_in = gr.Textbox(label="device_map", value=ARGS.device_map)
                    dtype_in = gr.Dropdown(["auto", "float16", "bfloat16", "float32"], label="torch_dtype", value=ARGS.torch_dtype)
                with gr.Row():
                    max_tokens_in = gr.Number(label="max_new_tokens", value=ARGS.max_new_tokens, precision=0)
                    temp_in = gr.Number(label="temperature", value=ARGS.temperature)
                    top_p_in = gr.Number(label="top_p", value=ARGS.top_p)
                seed_in = gr.Number(label="Reset Seed (blank = random)", value=None, precision=0)
                with gr.Row():
                    load_model_btn = gr.Button("Load Model")
                    spectator_reset_btn = gr.Button("Reset Spectator Game")
                with gr.Row():
                    spectator_step_btn = gr.Button("Agent Step", variant="primary")
                    auto_steps_in = gr.Number(label="Auto steps", value=10, precision=0)
                    delay_in = gr.Number(label="Delay seconds", value=0.8)
                auto_play_btn = gr.Button("Auto-play")
                spectator_status = gr.Textbox(label="Spectator Status", lines=12)
                spectator_action_json = gr.Textbox(label="Projected Action / Raw Response", lines=12, interactive=False)
                spectator_memory_out = gr.Textbox(label="Model Memory", lines=3, interactive=False)

        spectator_reset_btn.click(
            reset_spectator,
            inputs=[seed_in],
            outputs=[spectator_img, spectator_overlay, spectator_status, spectator_action_json, spectator_memory_out],
        )
        load_model_btn.click(
            load_spectator_model,
            inputs=[model_path_in, auto_merge_in, device_map_in, dtype_in],
            outputs=[spectator_status],
        )
        spectator_step_btn.click(
            spectator_step,
            inputs=[model_path_in, auto_merge_in, device_map_in, dtype_in, max_tokens_in, temp_in, top_p_in],
            outputs=[spectator_img, spectator_overlay, spectator_status, spectator_action_json, spectator_memory_out],
        )
        auto_play_btn.click(
            spectator_auto_play,
            inputs=[model_path_in, auto_merge_in, device_map_in, dtype_in, max_tokens_in, temp_in, top_p_in, auto_steps_in, delay_in],
            outputs=[spectator_img, spectator_overlay, spectator_status, spectator_action_json, spectator_memory_out],
        )
        demo.load(reset_spectator, inputs=[seed_in], outputs=[spectator_img, spectator_overlay, spectator_status, spectator_action_json, spectator_memory_out])

if __name__ == "__main__":
    print("Starting Gradio server...")
    # Bind to 0.0.0.0 so it can be accessed remotely
    demo.launch(server_name=ARGS.server_name, server_port=ARGS.server_port, share=False)
