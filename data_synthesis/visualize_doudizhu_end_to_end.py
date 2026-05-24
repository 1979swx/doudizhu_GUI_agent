#!/usr/bin/env python3
"""Render HTML samples from synthesized Dou Dizhu end-to-end SFT parquet files."""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

TAG_NAMES = ("plan", "action", "tool_call", "chat", "memory")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Dou Dizhu end-to-end SFT parquet data as an HTML review page.")
    parser.add_argument("--input", type=Path, required=True, help="Path to train/val/test parquet.")
    parser.add_argument("--output", type=Path, default=Path("visualize_doudizhu_end_to_end.html"))
    parser.add_argument("--num-samples", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument(
        "--order",
        choices=("random", "episode"),
        default="random",
        help="random samples rows; episode shows the first rows sorted by episode/step.",
    )
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


def load_image(image_entry: Any) -> Image.Image:
    image_entry = normalize_obj(image_entry)
    if isinstance(image_entry, dict):
        data = image_entry.get("bytes")
    else:
        data = image_entry["bytes"]
    return Image.open(io.BytesIO(data)).convert("RGB")


def png_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def normalize_clicks(value: Any) -> list[list[int]]:
    value = normalize_obj(value)
    if not isinstance(value, list):
        return []
    clicks: list[list[int]] = []
    for item in value:
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


def extract_tags(answer: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for tag in TAG_NAMES:
        match = re.search(rf"<{tag}>(.*?)</{tag}>", answer or "", flags=re.IGNORECASE | re.DOTALL)
        result[tag] = match.group(1).strip() if match else ""
    return result


def compact_json(value: Any) -> str:
    return json.dumps(normalize_obj(value), ensure_ascii=False, indent=2, default=str)


def extra_info_from_row(row: pd.Series) -> dict[str, Any]:
    extra_info = normalize_obj(row["extra_info"])
    if isinstance(extra_info, dict):
        return extra_info
    return {"raw_extra_info": extra_info}


def row_sort_key(row: pd.Series) -> tuple[int, int, int]:
    extra_info = extra_info_from_row(row)
    return (
        int(extra_info.get("episode_seed", 0)),
        int(extra_info.get("episode_index", 0)),
        int(extra_info.get("step_index", 0)),
    )


def choose_samples(df: pd.DataFrame, num_samples: int, seed: int, order: str) -> pd.DataFrame:
    sample_count = min(max(0, num_samples), len(df))
    if sample_count == 0:
        return df.iloc[0:0]
    if order == "episode":
        keyed = [(row_sort_key(row), idx) for idx, row in df.iterrows()]
        indices = [idx for _key, idx in sorted(keyed)[:sample_count]]
        return df.loc[indices]
    return df.sample(n=sample_count, random_state=seed)


def full_summary(df: pd.DataFrame) -> dict[str, Any]:
    categories: Counter[str] = Counter()
    submit_kinds: Counter[str] = Counter()
    models: Counter[str] = Counter()
    backends: Counter[str] = Counter()
    verifier_rates: Counter[str] = Counter()
    token_values: list[int] = []
    for _idx, row in df.iterrows():
        extra = extra_info_from_row(row)
        categories[str(extra.get("action_category", "unknown"))] += 1
        submit_kinds[str(extra.get("submit_kind", "unknown"))] += 1
        models[str(extra.get("source_model", "unknown"))] += 1
        backends[str(extra.get("source_backend", "unknown"))] += 1
        verifier = normalize_obj(extra.get("verifier", {}))
        if isinstance(verifier, dict):
            for key in ("projection_valid", "click_valid_ratio", "rule_action_valid", "action_match", "fallback_used", "selected_before_pass"):
                verifier_rates[f"{key}:{verifier.get(key)}"] += 1
        tokens = normalize_obj(extra.get("tokens", {}))
        if isinstance(tokens, dict):
            try:
                token_values.append(int(tokens.get("response_tokens_target_model", 0)))
            except (TypeError, ValueError):
                pass
    return {
        "rows": len(df),
        "action_category": dict(sorted(categories.items())),
        "submit_kind": dict(sorted(submit_kinds.items())),
        "source_model": dict(sorted(models.items())),
        "source_backend": dict(sorted(backends.items())),
        "verifier_values": dict(sorted(verifier_rates.items())),
        "response_tokens": {
            "min": min(token_values) if token_values else 0,
            "max": max(token_values) if token_values else 0,
            "mean": round(sum(token_values) / len(token_values), 2) if token_values else 0,
        },
    }


def render_tag_table(tags: dict[str, str]) -> str:
    rows = []
    for tag in TAG_NAMES:
        rows.append(
            f"""
            <tr>
              <th>{html.escape(tag)}</th>
              <td><pre>{html.escape(tags.get(tag, ""))}</pre></td>
            </tr>
            """
        )
    return f'<table class="tags">{"".join(rows)}</table>'


def render_sample(display_index: int, row_index: int, row: pd.Series) -> tuple[str, dict[str, Any]]:
    images = normalize_obj(row["images"])
    extra_info = extra_info_from_row(row)
    image = load_image(images[0])
    clicks = normalize_clicks(extra_info.get("tool_clicks", []))
    annotated_url = png_data_url(annotate_image(image, clicks))
    tags = extract_tags(str(row["answer"]))

    sample_id = html.escape(str(extra_info.get("sample_id", "")))
    split = html.escape(str(extra_info.get("split", "")))
    action = html.escape(str(extra_info.get("game_action", "")))
    category = html.escape(str(extra_info.get("action_category", "unknown")))
    seed = html.escape(str(extra_info.get("episode_seed", "")))
    step = html.escape(str(extra_info.get("step_index", "")))
    click_text = html.escape(json.dumps(clicks, ensure_ascii=False))
    answer = html.escape(str(row["answer"]))
    raw_response = str(extra_info.get("raw_response", ""))
    raw_response_block = ""
    if raw_response and raw_response != str(row["answer"]):
        raw_response_block = f"<h3>Raw Response</h3><pre>{html.escape(raw_response)}</pre>"
    question = html.escape(str(row["question"]).strip())
    verifier = html.escape(compact_json(extra_info.get("verifier", {})))
    episode = html.escape(compact_json(extra_info.get("episode", {})))
    api = html.escape(compact_json(extra_info.get("api", {})))
    distribution = html.escape(compact_json(extra_info.get("distribution", {})))
    pretty_extra = html.escape(compact_json(extra_info))

    block = f"""
    <section class="sample">
      <div class="sample-head">
        <h2>#{display_index} row={row_index} action={action}</h2>
        <div class="badges">
          <span>{split}</span>
          <span>{category}</span>
          <span>seed {seed}</span>
          <span>step {step}</span>
          <span>{len(clicks)} clicks</span>
        </div>
      </div>
      <p class="sample-id">{sample_id}</p>
      <img src="{annotated_url}" />
      <div class="grid">
        <div>
          <h3>Clicks</h3>
          <pre>{click_text}</pre>
          <h3>Answer Tags</h3>
          {render_tag_table(tags)}
        </div>
        <div>
          <h3>Verifier</h3>
          <pre>{verifier}</pre>
          <h3>Episode</h3>
          <pre>{episode}</pre>
          <h3>API</h3>
          <pre>{api}</pre>
        </div>
      </div>
      <details>
        <summary>Full Answer</summary>
        <pre>{answer}</pre>
      </details>
      {raw_response_block}
      <details>
        <summary>Prompt</summary>
        <pre>{question}</pre>
      </details>
      <details>
        <summary>Distribution</summary>
        <pre>{distribution}</pre>
      </details>
      <details>
        <summary>Extra Info</summary>
        <pre>{pretty_extra}</pre>
      </details>
    </section>
    """
    return block, extra_info


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.input)

    samples = choose_samples(df, args.num_samples, args.seed, args.order)
    blocks: list[str] = []
    sampled_extra: list[dict[str, Any]] = []
    for display_index, (row_index, row) in enumerate(samples.iterrows(), start=1):
        block, extra_info = render_sample(display_index, int(row_index), row)
        blocks.append(block)
        sampled_extra.append(extra_info)

    summary = html.escape(compact_json(full_summary(df)))
    sampled_summary = html.escape(
        compact_json(
            {
                "sampled": len(sampled_extra),
                "action_category": dict(Counter(str(info.get("action_category", "unknown")) for info in sampled_extra)),
                "episode_seed": dict(Counter(str(info.get("episode_seed", "unknown")) for info in sampled_extra)),
            }
        )
    )
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
        .sample-head h2 {{ font-size: 20px; margin: 0; }}
        .badges {{ display: flex; gap: 8px; flex-wrap: wrap; }}
        .badges span {{ background: #e9eef6; border: 1px solid #d3dbe8; border-radius: 999px; padding: 4px 10px; font-size: 13px; }}
        .sample-id {{ color: #5d6675; margin-top: 8px; }}
        img {{ width: 640px; max-width: 100%; display: block; border: 1px solid #ccd3dd; margin: 12px 0; }}
        pre {{ white-space: pre-wrap; word-break: break-word; background: #f0f2f5; padding: 12px; border-radius: 6px; margin-top: 6px; }}
        h1, h2 {{ margin-top: 0; }}
        h3 {{ margin-bottom: 8px; }}
        summary {{ cursor: pointer; font-weight: 700; margin: 12px 0 6px; }}
        .grid {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 12px; }}
        .tags {{ width: 100%; border-collapse: collapse; background: #f8fafc; border: 1px solid #d8dee8; }}
        .tags th {{ width: 110px; vertical-align: top; text-align: left; padding: 8px; border-bottom: 1px solid #d8dee8; }}
        .tags td {{ padding: 8px; border-bottom: 1px solid #d8dee8; }}
        .tags pre {{ margin: 0; background: transparent; padding: 0; }}
        @media (max-width: 860px) {{ .grid {{ grid-template-columns: 1fr; }} }}
      </style>
    </head>
    <body>
      <h1>Dou Dizhu End-to-End SFT Samples</h1>
      <p>Source: {html.escape(str(args.input))}</p>
      <p>Rows: {len(df)}; sampled: {len(samples)}; seed: {args.seed}; order: {html.escape(args.order)}</p>
      <h2>Full Dataset Summary</h2>
      <pre>{summary}</pre>
      <h2>Sample Summary</h2>
      <pre>{sampled_summary}</pre>
      {"".join(blocks)}
    </body>
    </html>
    """
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_text, encoding="utf-8")
    print(f"Wrote {len(samples)} samples to {args.output}")


if __name__ == "__main__":
    main()
