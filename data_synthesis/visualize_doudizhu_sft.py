#!/usr/bin/env python3
"""Visualize synthesized Doudizhu grounding SFT data.

This script reads a parquet file, extracts images, and draws the gold click
coordinates to verify data quality.
"""

import argparse
import io
import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Doudizhu SFT parquet data.")
    parser.add_argument("--input", type=Path, default=Path("doudizhu_grounding_sft/train.parquet"), help="Path to the parquet file.")
    parser.add_argument("--output-dir", type=Path, default=Path("visualize_samples"), help="Directory to save visualized images.")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to visualize.")
    parser.add_argument("--show-prompt", action="store_true", help="Overlay the full prompt text on the image.")
    return parser.parse_args()


def main():
    args = parse_args()
    
    if not args.input.exists():
        print(f"Error: Input file {args.input} not found.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {args.input}...")
    df = pd.read_parquet(args.input)
    
    # Limit number of samples and select randomly
    num_to_show = min(len(df), args.num_samples)
    samples = df.sample(n=num_to_show)

    for i, (_, row) in enumerate(samples.iterrows()):
        # 1. Extract Image
        img_data = row['images'][0]['bytes']
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        # 2. Parse Info
        # Parquet might store dicts as strings depending on the engine/version
        info = row['extra_info']
        if isinstance(info, str):
            info = json.loads(info)
            
        target_action = info.get('target_action_pretty', info.get('target_action', 'Unknown'))
        clicks = info.get('gold_clicks', [])
        
        # 3. Draw Clicks
        w, h = img.size
        for idx, (cx, cy) in enumerate(clicks):
            # Convert normalized [0, 1000] to pixel coordinates
            px = cx * w / 1000.0
            py = cy * h / 1000.0
            
            # Draw circle
            r = 8
            color = "red" if idx < len(clicks) - 1 else "green"  # Final click (Submit) is green
            draw.ellipse([px - r, py - r, px + r, py + r], fill=color, outline="white", width=2)
            
            # Draw index
            draw.text((px + r + 2, py - r), str(idx + 1), fill="white")

        # 4. Add Text Label
        label = f"Sample {i} | Action: {target_action}"
        if args.show_prompt:
            label = f"{label}\n{row['question']}"
        try:
            # Try to use a larger font if available, else fallback
            font = ImageFont.load_default()
            draw.text((10, 10), label, fill="yellow", font=font)
        except Exception:
            draw.text((10, 10), label, fill="yellow")

        # 5. Save
        output_path = args.output_dir / f"sample_{i:03d}.png"
        img.save(output_path)
        print(f"Saved: {output_path} (Action: {target_action})")

    print(f"\nFinished! Visualized {num_to_show} samples in '{args.output_dir}/'")


if __name__ == "__main__":
    main()
