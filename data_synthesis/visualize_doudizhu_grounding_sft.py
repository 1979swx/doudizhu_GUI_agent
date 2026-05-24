#!/usr/bin/env python3
"""Render HTML samples from synthesized Dou Dizhu grounding SFT parquet files."""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Dou Dizhu grounding SFT parquet data as an HTML review page.")
    parser.add_argument("--input", type=Path, required=True, help="Path to train/val/test parquet.")
    parser.add_argument("--output", type=Path, default=Path("visualize_doudizhu_grounding.html"))
    parser.add_argument("--num-samples", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260523)
    return parser.parse_args()


def normalize_obj(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    if isinstance(value, np.ndarray):
        return [normalize_obj(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): normalize_obj(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_obj(item) for item in value]
    return value


def png_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def load_image(image_entry: Any) -> Image.Image:
    if isinstance(image_entry, dict):
        data = image_entry.get("bytes")
    else:
        data = image_entry["bytes"]
    return Image.open(io.BytesIO(data)).convert("RGB")


def normalize_clicks(value: Any) -> list[list[int]]:
    normalized = normalize_obj(value)
    if not isinstance(normalized, list):
        return []
    clicks: list[list[int]] = []
    for item in normalized:
        if isinstance(item, list) and len(item) == 2:
            try:
                x = int(round(float(item[0])))
                y = int(round(float(item[1])))
            except (TypeError, ValueError):
                continue
            clicks.append([max(0, min(1000, x)), max(0, min(1000, y))])
    return clicks


def annotate_image(image: Image.Image, clicks: list[list[int]]) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    width, height = annotated.size
    points = [(x * width / 1000.0, y * height / 1000.0) for x, y in clicks]
    if len(points) >= 2:
        draw.line(points, fill=(255, 202, 40), width=3)
    for index, (px, py) in enumerate(points, start=1):
        radius = 10
        is_submit = index == len(points)
        fill = (29, 142, 64) if is_submit else (220, 38, 38)
        draw.ellipse([px - radius, py - radius, px + radius, py + radius], fill=fill, outline=(255, 255, 255), width=3)
        label = str(index)
        bbox = draw.textbbox((0, 0), label)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        draw.rectangle([px + radius + 4, py - radius, px + radius + text_width + 12, py - radius + text_height + 8], fill=(20, 24, 31))
        draw.text((px + radius + 8, py - radius + 4), label, fill=(255, 255, 255))
    return annotated


def compact_prompt(question: Any) -> str:
    text = str(question)
    return text.strip()


def sample_summary(extra_infos: list[dict[str, Any]]) -> dict[str, Any]:
    categories = Counter(str(info.get("action_category", "unknown")) for info in extra_infos)
    submit_kinds = Counter(str(info.get("verifier", {}).get("submit_kind", "unknown")) for info in extra_infos)
    return {
        "sampled_action_categories": dict(sorted(categories.items())),
        "sampled_submit_kinds": dict(sorted(submit_kinds.items())),
    }


def render_sample(index: int, row: pd.Series) -> tuple[str, dict[str, Any]]:
    images = normalize_obj(row["images"])
    extra_info = normalize_obj(row["extra_info"])
    if not isinstance(extra_info, dict):
        extra_info = {"raw_extra_info": extra_info}

    image = load_image(images[0])
    clicks = normalize_clicks(extra_info.get("gold_clicks", []))
    annotated_url = png_data_url(annotate_image(image, clicks))

    target_action = html.escape(str(extra_info.get("target_action_pretty", extra_info.get("target_action", "unknown"))))
    action_category = html.escape(str(extra_info.get("action_category", "unknown")))
    sample_id = html.escape(str(extra_info.get("sample_id", "")))
    answer = html.escape(str(row["answer"]))
    question = html.escape(compact_prompt(row["question"]))
    clicks_text = html.escape(json.dumps(clicks, ensure_ascii=False))
    verifier = extra_info.get("verifier", {})
    pretty_verifier = html.escape(json.dumps(verifier, ensure_ascii=False, indent=2, default=str))
    pretty_extra = html.escape(json.dumps(extra_info, ensure_ascii=False, indent=2, default=str))

    block = f"""
    <section class="sample">
      <div class="sample-head">
        <h2>#{index} {target_action}</h2>
        <div class="badges">
          <span>{action_category}</span>
          <span>{len(clicks)} clicks</span>
        </div>
      </div>
      <p class="sample-id">{sample_id}</p>
      <img src="{annotated_url}" />
      <div class="grid">
        <div>
          <h3>Gold Clicks</h3>
          <pre>{clicks_text}</pre>
        </div>
        <div>
          <h3>Answer</h3>
          <pre>{answer}</pre>
        </div>
      </div>
      <h3>Question</h3>
      <pre>{question}</pre>
      <h3>Verifier</h3>
      <pre>{pretty_verifier}</pre>
      <h3>Extra Info</h3>
      <pre>{pretty_extra}</pre>
    </section>
    """
    return block, extra_info


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.input)
    if len(df) == 0:
        raise ValueError(f"{args.input} contains no rows.")

    sample_count = min(args.num_samples, len(df))
    samples = df.sample(n=sample_count, random_state=args.seed)
    blocks: list[str] = []
    extra_infos: list[dict[str, Any]] = []
    for index, row in samples.iterrows():
        block, extra_info = render_sample(index, row)
        blocks.append(block)
        extra_infos.append(extra_info)

    summary = html.escape(json.dumps(sample_summary(extra_infos), ensure_ascii=False, indent=2))
    html_text = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <title>Dou Dizhu Grounding SFT Samples</title>
      <style>
        body {{ font-family: sans-serif; margin: 24px; background: #f6f7f9; color: #20242a; }}
        .sample {{ background: white; border: 1px solid #d8dee8; border-radius: 8px; padding: 16px; margin-bottom: 20px; }}
        .sample-head {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; }}
        .badges {{ display: flex; gap: 8px; flex-wrap: wrap; }}
        .badges span {{ background: #e9eef6; border: 1px solid #d3dbe8; border-radius: 999px; padding: 4px 10px; font-size: 13px; }}
        .sample-id {{ color: #5d6675; margin-top: 0; }}
        img {{ width: 640px; max-width: 100%; display: block; border: 1px solid #ccd3dd; }}
        pre {{ white-space: pre-wrap; word-break: break-word; background: #f0f2f5; padding: 12px; border-radius: 6px; }}
        h1, h2 {{ margin-top: 0; }}
        h3 {{ margin-bottom: 8px; }}
        .grid {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 12px; }}
        @media (max-width: 860px) {{ .grid {{ grid-template-columns: 1fr; }} }}
      </style>
    </head>
    <body>
      <h1>Dou Dizhu Grounding SFT Samples</h1>
      <p>Source: {html.escape(str(args.input))}</p>
      <p>Rows: {len(df)}; sampled: {sample_count}; seed: {args.seed}</p>
      <h2>Sample Summary</h2>
      <pre>{summary}</pre>
      {''.join(blocks)}
    </body>
    </html>
    """
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_text, encoding="utf-8")
    print(f"Wrote {sample_count} samples to {args.output}")


if __name__ == "__main__":
    main()
