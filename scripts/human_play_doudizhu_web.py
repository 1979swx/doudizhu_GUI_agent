import os
import sys
import json
import numpy as np

try:
    import gradio as gr
except ImportError:
    print("Gradio is not installed. Please run: pip install gradio")
    sys.exit(1)

# Add project root to sys.path to ensure absolute imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from agent_system.environments.env_package.doudizhu.envs import DoudizhuSingleEnv
except ImportError as e:
    print(f"Failed to import DoudizhuSingleEnv: {e}")
    print("Make sure you run this script from the project root or the conda environment is active.")
    sys.exit(1)

# Global state to track human interactions
current_clicks = []
current_obs = None
done = False
env = DoudizhuSingleEnv(seed=42)

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

with gr.Blocks(title="Doudizhu Human Debugger", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 斗地主 (Dou Dizhu) Agentic Environment 🎮")
    gr.Markdown("Click on the game UI to simulate the LLM's coordinate outputs, then submit them to the environment.")
    
    with gr.Row():
        with gr.Column(scale=2):
            img = gr.Image(interactive=False, label="Environment Observation (Click to select coords)")
            
            with gr.Row():
                clear_btn = gr.Button("🧹 Clear Clicks")
                step_btn = gr.Button("🚀 Submit Clicks to Env", variant="primary")
                reset_btn = gr.Button("🔄 Reset Game")
                
        with gr.Column(scale=1):
            status_out = gr.Textbox(label="Status / Step Result", lines=6)
            clicks_out = gr.Textbox(label="Current <action> JSON", interactive=False)
            chat_in = gr.Textbox(label="💬 <chat> Input (Optional)", placeholder="Say something to the peasants...")
            memory_in = gr.Textbox(label="🧠 <memory> Input/Output", placeholder="Write a note to yourself for the next turn...")
            
    # Wire up the events
    img.select(handle_click, outputs=[status_out, clicks_out])
    clear_btn.click(clear_clicks, outputs=[status_out, clicks_out])
    reset_btn.click(reset_env, outputs=[img, status_out, clicks_out, memory_in])
    step_btn.click(step_env, inputs=[chat_in, memory_in], outputs=[img, status_out, clicks_out, memory_in])
    
    # Initialize the game on load
    demo.load(reset_env, outputs=[img, status_out, clicks_out, memory_in])

if __name__ == "__main__":
    print("Starting Gradio server...")
    # Bind to 0.0.0.0 so it can be accessed remotely
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
