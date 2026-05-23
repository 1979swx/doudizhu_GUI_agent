#!/usr/bin/env python3
"""Render HTML samples from synthesized Dou Dizhu QA SFT parquet files."""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Dou Dizhu QA SFT parquet data as an HTML review page.")
    parser.add_argument("--input", type=Path, required=True, help="Path to train/val/test parquet.")
    parser.add_argument("--output", type=Path, default=Path("visualize_doudizhu_qa.html"))
    parser.add_argument("--num-samples", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260523)
    return parser.parse_args()


def normalize_obj(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def image_data_url(image_entry: Any) -> str:
    if isinstance(image_entry, dict):
        data = image_entry.get("bytes")
    else:
        data = image_entry["bytes"]
    image = Image.open(io.BytesIO(data)).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.input)
    if len(df) == 0:
        raise ValueError(f"{args.input} contains no rows.")

    sample_count = min(args.num_samples, len(df))
    samples = df.sample(n=sample_count, random_state=args.seed)
    blocks: list[str] = []
    for idx, row in samples.iterrows():
        images = normalize_obj(row["images"])
        extra_info = normalize_obj(row["extra_info"])
        image_url = image_data_url(images[0])
        question = html.escape(str(row["question"]))
        answer = html.escape(str(row["answer"]))
        pretty_extra = html.escape(json.dumps(extra_info, ensure_ascii=False, indent=2, default=str))
        task = html.escape(str(extra_info.get("task_id", "unknown") if isinstance(extra_info, dict) else "unknown"))
        blocks.append(
            f"""
            <section class="sample">
              <h2>#{idx} {task}</h2>
              <img src="{image_url}" />
              <h3>Question</h3>
              <pre>{question}</pre>
              <h3>Answer</h3>
              <pre>{answer}</pre>
              <h3>Extra Info</h3>
              <pre>{pretty_extra}</pre>
            </section>
            """
        )

    html_text = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <title>Dou Dizhu QA SFT Samples</title>
      <style>
        body {{ font-family: sans-serif; margin: 24px; background: #f6f7f9; color: #20242a; }}
        .sample {{ background: white; border: 1px solid #d8dee8; border-radius: 8px; padding: 16px; margin-bottom: 20px; }}
        img {{ width: 640px; max-width: 100%; display: block; border: 1px solid #ccd3dd; }}
        pre {{ white-space: pre-wrap; word-break: break-word; background: #f0f2f5; padding: 12px; border-radius: 6px; }}
        h2 {{ margin-top: 0; }}
      </style>
    </head>
    <body>
      <h1>Dou Dizhu QA SFT Samples</h1>
      <p>Source: {html.escape(str(args.input))}</p>
      {''.join(blocks)}
    </body>
    </html>
    """
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_text, encoding="utf-8")
    print(f"Wrote {sample_count} samples to {args.output}")


if __name__ == "__main__":
    main()
