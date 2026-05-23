#!/usr/bin/env python3
"""Standalone FSDP SFT trainer for Qwen3.5-style VLM and text datasets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
from io import BytesIO

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.distributed.fsdp import CPUOffload, FullStateDictConfig, MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

try:
    from transformers import AutoModelForImageTextToText
except ImportError:  # pragma: no cover
    AutoModelForImageTextToText = None

try:
    from transformers import AutoModelForVision2Seq
except ImportError:  # pragma: no cover
    AutoModelForVision2Seq = None

from transformers import AutoModelForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verl.models.transformers.monkey_patch import apply_monkey_patch  # noqa: E402
from verl.utils.dataset.vision_utils import process_image  # noqa: E402
from verl.utils.fs import copy_to_local  # noqa: E402
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, get_init_weight_context_manager, init_fn  # noqa: E402
from verl.utils.torch_functional import get_cosine_schedule_with_warmup  # noqa: E402


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FSDP SFT for Qwen3.5 VLM/text synthetic data.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-file", nargs="+", required=True)
    parser.add_argument("--val-file", nargs="+", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--project-name", default="verl_agent_sft")
    parser.add_argument("--experiment-name", default="qwen3_5_sft")
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--response-key", default="answer")
    parser.add_argument("--image-key", default="images")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--truncation", choices=("error", "right"), default="error")
    parser.add_argument("--train-batch-size", type=int, default=32, help="Global train batch size.")
    parser.add_argument("--micro-batch-size-per-gpu", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--betas", default="0.9,0.95")
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-freq", type=int, default=500)
    parser.add_argument("--val-freq", type=int, default=500)
    parser.add_argument("--save-final", type=str_to_bool, default=True)
    parser.add_argument("--fsdp-sharding", choices=("zero2", "zero3", "no_shard", "hybrid"), default="zero3")
    parser.add_argument("--fsdp-min-num-params", type=int, default=0)
    parser.add_argument("--fsdp-transformer-layer-cls-to-wrap", default=None)
    parser.add_argument("--cpu-offload", type=str_to_bool, default=False)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--model-dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--trust-remote-code", type=str_to_bool, default=False)
    parser.add_argument("--gradient-checkpointing", type=str_to_bool, default=True)
    parser.add_argument("--enable-thinking", type=str_to_bool, default=False)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--logger", choices=("console", "wandb", "none"), default="console")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, int, torch.device]:
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % max(torch.cuda.device_count(), 1)))
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size, torch.device("cuda", local_rank)


def series_to_item(value: Any) -> Any:
    while isinstance(value, (pd.Series, np.ndarray)) and len(value) == 1:
        value = value[0]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def load_parquet(files: list[str]) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in files]
    if not frames:
        raise ValueError("No parquet files provided.")
    return pd.concat(frames, ignore_index=True)


def safe_apply_chat_template(processor: AutoProcessor, messages: list[dict[str, Any]], *, add_generation_prompt: bool, enable_thinking: bool) -> str:
    kwargs = {
        "add_generation_prompt": add_generation_prompt,
        "tokenize": False,
        "enable_thinking": enable_thinking,
    }
    try:
        return processor.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return processor.apply_chat_template(messages, **kwargs)


def load_processor_or_tokenizer(model_path: str, trust_remote_code: bool):
    try:
        return AutoProcessor.from_pretrained(model_path, trust_remote_code=trust_remote_code, use_fast=True)
    except Exception as processor_error:  # noqa: BLE001
        try:
            return AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code, use_fast=True)
        except Exception:
            raise processor_error


def encode_text_and_images(processor, text: str, images: list[Any] | None) -> dict[str, torch.Tensor]:
    if images:
        return processor(text=[text], images=images, return_tensors="pt")
    return processor(text=[text], return_tensors="pt")


def decode_image(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, dict):
        if "bytes" in image:
            return Image.open(BytesIO(image["bytes"])).convert("RGB")
        if isinstance(image.get("image"), bytes):
            return Image.open(BytesIO(image["image"])).convert("RGB")
        if isinstance(image.get("image"), BytesIO):
            return Image.open(image["image"]).convert("RGB")
    return process_image(dict(image) if isinstance(image, dict) else image)


def normalize_messages(prompt_value: Any, has_images: bool) -> list[dict[str, Any]]:
    prompt_value = series_to_item(prompt_value)
    if isinstance(prompt_value, str):
        messages = [{"role": "user", "content": prompt_value}]
    elif isinstance(prompt_value, dict):
        messages = [dict(prompt_value)]
    elif isinstance(prompt_value, list):
        messages = [dict(message) for message in prompt_value]
    else:
        raise TypeError(f"Unsupported prompt type: {type(prompt_value)!r}")

    if not has_images:
        return messages

    image_placeholders = 0
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            parts: list[dict[str, Any]] = []
            segments = content.split("<image>")
            for idx, segment in enumerate(segments):
                if segment:
                    parts.append({"type": "text", "text": segment})
                if idx < len(segments) - 1:
                    parts.append({"type": "image"})
                    image_placeholders += 1
            if parts:
                message["content"] = parts
            else:
                message["content"] = [{"type": "image"}]
                image_placeholders += 1
        elif isinstance(content, list):
            image_placeholders += sum(1 for item in content if isinstance(item, dict) and item.get("type") == "image")
    if has_images and image_placeholders == 0 and messages:
        content = messages[0].get("content", "")
        if isinstance(content, list):
            messages[0]["content"] = [{"type": "image"}] + content
        else:
            messages[0]["content"] = [{"type": "image"}, {"type": "text", "text": str(content)}]
    return messages


class SyntheticSFTDataset(Dataset):
    def __init__(
        self,
        parquet_files: list[str],
        processor: AutoProcessor,
        prompt_key: str,
        response_key: str,
        image_key: str,
        max_length: int,
        truncation: str,
        enable_thinking: bool,
    ):
        self.dataframe = load_parquet(parquet_files)
        self.processor = processor
        self.prompt_key = prompt_key
        self.response_key = response_key
        self.image_key = image_key
        self.max_length = int(max_length)
        self.truncation = truncation
        self.enable_thinking = enable_thinking
        self.has_images = bool(image_key in self.dataframe.columns and self.dataframe[image_key].notna().any())

    def __len__(self) -> int:
        return len(self.dataframe)

    def _images(self, row: pd.Series) -> list[Any] | None:
        if self.image_key not in row or row[self.image_key] is None:
            return None
        raw_images = series_to_item(row[self.image_key])
        if raw_images is None:
            return None
        if not isinstance(raw_images, list):
            raw_images = [raw_images]
        if not raw_images:
            return None
        return [decode_image(dict(image) if isinstance(image, dict) else image) for image in raw_images]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.dataframe.iloc[index]
        images = self._images(row)
        has_images = bool(images)
        prompt_messages = normalize_messages(row[self.prompt_key], has_images=has_images)
        answer = str(series_to_item(row[self.response_key]))
        full_messages = prompt_messages + [{"role": "assistant", "content": answer}]

        prompt_text = safe_apply_chat_template(
            self.processor,
            prompt_messages,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        full_text = safe_apply_chat_template(
            self.processor,
            full_messages,
            add_generation_prompt=False,
            enable_thinking=self.enable_thinking,
        )

        prompt_inputs = encode_text_and_images(self.processor, prompt_text, images)
        full_inputs = encode_text_and_images(self.processor, full_text, images)

        input_ids = full_inputs["input_ids"][0]
        attention_mask = full_inputs["attention_mask"][0]
        prompt_length = int(prompt_inputs["input_ids"].shape[-1])

        if input_ids.shape[-1] > self.max_length:
            if self.truncation == "error":
                raise ValueError(f"Sample {index} length {input_ids.shape[-1]} exceeds max_length={self.max_length}")
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]

        labels = input_ids.clone()
        labels[: min(prompt_length, labels.shape[-1])] = -100
        labels[attention_mask == 0] = -100

        item: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        for key, value in full_inputs.items():
            if key in item or key in {"input_ids", "attention_mask"}:
                continue
            if isinstance(value, torch.Tensor):
                item[key] = value
        return item


class SFTCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = int(pad_token_id)

    def __call__(self, features: list[dict[str, torch.Tensor]], include_raw_features: bool = True) -> dict[str, torch.Tensor]:
        max_len = max(feature["input_ids"].shape[-1] for feature in features)
        batch: dict[str, list[torch.Tensor] | torch.Tensor] = {"input_ids": [], "attention_mask": [], "labels": []}

        for feature in features:
            length = feature["input_ids"].shape[-1]
            pad_len = max_len - length
            batch["input_ids"].append(F.pad(feature["input_ids"], (0, pad_len), value=self.pad_token_id))
            batch["attention_mask"].append(F.pad(feature["attention_mask"], (0, pad_len), value=0))
            batch["labels"].append(F.pad(feature["labels"], (0, pad_len), value=-100))

        output = {key: torch.stack(value, dim=0) for key, value in batch.items()}
        extra_keys = sorted(set().union(*(set(feature.keys()) for feature in features)) - set(output.keys()))
        for key in extra_keys:
            values = [feature[key] for feature in features if key in feature]
            if not values:
                continue
            if all(isinstance(value, torch.Tensor) for value in values):
                if key.endswith("token_type_ids"):
                    padded_values = []
                    for value in values:
                        if value.dim() == 1:
                            padded_values.append(F.pad(value, (0, max_len - value.shape[-1]), value=0))
                        elif value.dim() == 2 and value.shape[0] == 1:
                            padded_values.append(F.pad(value, (0, max_len - value.shape[-1]), value=0))
                        else:
                            raise ValueError(f"Unsupported token type tensor shape for {key}: {tuple(value.shape)}")
                    output[key] = torch.stack(padded_values, dim=0) if padded_values[0].dim() == 1 else torch.cat(padded_values, dim=0)
                else:
                    output[key] = torch.cat(values, dim=0)
        if include_raw_features:
            output["_raw_features"] = features
        return output


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
        if not key.startswith("_")
    }


def sharding_strategy(name: str) -> ShardingStrategy:
    return {
        "zero3": ShardingStrategy.FULL_SHARD,
        "zero2": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
        "hybrid": ShardingStrategy.HYBRID_SHARD,
    }[name]


def model_class_for_config(config: AutoConfig):
    if AutoModelForImageTextToText is not None and type(config) in AutoModelForImageTextToText._model_mapping.keys():
        return AutoModelForImageTextToText
    if AutoModelForVision2Seq is not None and type(config) in AutoModelForVision2Seq._model_mapping.keys():
        return AutoModelForVision2Seq
    return AutoModelForCausalLM


def build_model(args: argparse.Namespace, local_model_path: str, rank: int, device: torch.device):
    config = AutoConfig.from_pretrained(
        local_model_path,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    )
    model_dtype = torch.float32 if args.model_dtype == "fp32" else torch.bfloat16
    init_context = get_init_weight_context_manager(use_meta_tensor=not getattr(config, "tie_word_embeddings", False), mesh=None)
    with init_context():
        model_cls = model_class_for_config(config)
        model = model_cls.from_pretrained(
            local_model_path,
            config=config,
            torch_dtype=model_dtype,
            trust_remote_code=args.trust_remote_code,
        )
        apply_monkey_patch(model=model, ulysses_sp_size=1, use_remove_padding=False)
        model.to(model_dtype)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False

    wrap_policy_config: dict[str, Any] = {"min_num_params": args.fsdp_min_num_params}
    if args.fsdp_transformer_layer_cls_to_wrap:
        wrap_policy_config["transformer_layer_cls_to_wrap"] = [
            item.strip() for item in args.fsdp_transformer_layer_cls_to_wrap.split(",") if item.strip()
        ]
    auto_wrap_policy = get_fsdp_wrap_policy(model, config=wrap_policy_config)
    cpu_offload = CPUOffload(offload_params=True) if args.cpu_offload else None
    mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)

    fsdp_model = FSDP(
        model,
        cpu_offload=cpu_offload,
        param_init_fn=init_fn,
        use_orig_params=False,
        auto_wrap_policy=auto_wrap_policy,
        device_id=device,
        sharding_strategy=sharding_strategy(args.fsdp_sharding),
        mixed_precision=mixed_precision,
        sync_module_states=True,
        forward_prefetch=False,
    )
    if rank == 0:
        print(f"Loaded {model_cls.__name__} with FSDP sharding={args.fsdp_sharding}, model_dtype={args.model_dtype}")
    return model, fsdp_model


def compute_loss(fsdp_model: FSDP, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    labels = batch.pop("labels")
    outputs = fsdp_model(**batch, use_cache=False)
    logits = outputs.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="sum",
    )
    valid_tokens = (shift_labels != -100).sum()
    return loss / valid_tokens.clamp_min(1), valid_tokens


def validate(fsdp_model: FSDP, dataloader: DataLoader, device: torch.device) -> float:
    fsdp_model.eval()
    total_loss = torch.zeros((), device=device)
    total_tokens = torch.zeros((), device=device)
    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            labels = batch.pop("labels")
            outputs = fsdp_model(**batch, use_cache=False)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
            total_loss += loss
            total_tokens += (shift_labels != -100).sum()
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
    fsdp_model.train()
    return (total_loss / total_tokens.clamp_min(1)).item()


def save_checkpoint(
    args: argparse.Namespace,
    step: int,
    model,
    fsdp_model: FSDP,
    processor: AutoProcessor,
    rank: int,
) -> None:
    path = Path(args.output_dir) / f"global_step_{step}"
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, cfg):
        state_dict = fsdp_model.state_dict()
    if rank == 0:
        path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(path, state_dict=state_dict)
        processor.save_pretrained(path)
        with (path / "sft_args.json").open("w", encoding="utf-8") as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=2)
        print(f"Saved checkpoint to {path}")
    dist.barrier()


def maybe_init_wandb(args: argparse.Namespace, rank: int):
    if args.logger != "wandb" or rank != 0:
        return None
    import wandb

    return wandb.init(project=args.project_name, name=args.experiment_name, config=vars(args))


def main() -> None:
    args = parse_args()
    rank, _local_rank, world_size, device = setup_distributed()
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    if args.train_batch_size % world_size != 0:
        raise ValueError(f"--train-batch-size {args.train_batch_size} must be divisible by world_size {world_size}")
    per_rank_batch_size = args.train_batch_size // world_size
    if per_rank_batch_size % args.micro_batch_size_per_gpu != 0:
        raise ValueError(
            f"Per-rank batch size {per_rank_batch_size} must be divisible by "
            f"--micro-batch-size-per-gpu {args.micro_batch_size_per_gpu}"
        )

    local_model_path = copy_to_local(args.model_path, verbose=(rank == 0))
    processor = load_processor_or_tokenizer(local_model_path, trust_remote_code=args.trust_remote_code)
    tokenizer = getattr(processor, "tokenizer", processor)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = SyntheticSFTDataset(
        args.train_file,
        processor,
        args.prompt_key,
        args.response_key,
        args.image_key,
        args.max_length,
        args.truncation,
        args.enable_thinking,
    )
    val_dataset = (
        SyntheticSFTDataset(
            args.val_file,
            processor,
            args.prompt_key,
            args.response_key,
            args.image_key,
            args.max_length,
            args.truncation,
            args.enable_thinking,
        )
        if args.val_file
        else None
    )

    collator = SFTCollator(pad_token_id=tokenizer.pad_token_id)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=per_rank_batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collator,
        drop_last=True,
    )
    val_loader = None
    if val_dataset is not None:
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=max(1, args.micro_batch_size_per_gpu),
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collator,
            drop_last=False,
        )

    model, fsdp_model = build_model(args, local_model_path, rank, device)
    betas = tuple(float(item) for item in args.betas.split(","))
    optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=args.lr, betas=betas, weight_decay=args.weight_decay)

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    wandb_run = maybe_init_wandb(args, rank)

    if rank == 0:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        print(f"Train examples={len(train_dataset)}, val examples={0 if val_dataset is None else len(val_dataset)}")
        print(f"World size={world_size}, per-rank batch={per_rank_batch_size}, total_steps={total_steps}")

    global_step = 0
    fsdp_model.train()
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        iterator = tqdm(train_loader, disable=rank != 0, desc=f"epoch {epoch + 1}/{args.epochs}")
        for batch in iterator:
            global_step += 1
            micro_batches = []
            raw_features = batch.get("_raw_features")
            if raw_features is not None:
                for start in range(0, len(raw_features), args.micro_batch_size_per_gpu):
                    micro_batch = collator(
                        raw_features[start : start + args.micro_batch_size_per_gpu],
                        include_raw_features=False,
                    )
                    micro_batches.append(move_batch_to_device(micro_batch, device))
            else:
                batch = move_batch_to_device(batch, device)
                batch_size = batch["input_ids"].shape[0]
                for start in range(0, batch_size, args.micro_batch_size_per_gpu):
                    micro_batches.append({key: value[start : start + args.micro_batch_size_per_gpu] for key, value in batch.items()})

            optimizer.zero_grad(set_to_none=True)
            step_loss = torch.zeros((), device=device)
            step_tokens = torch.zeros((), device=device)
            for micro_batch in micro_batches:
                loss, valid_tokens = compute_loss(fsdp_model, micro_batch)
                (loss / len(micro_batches)).backward()
                step_loss += loss.detach() * valid_tokens
                step_tokens += valid_tokens

            grad_norm = fsdp_model.clip_grad_norm_(args.grad_clip)
            if torch.isfinite(grad_norm):
                optimizer.step()
                scheduler.step()
            else:
                optimizer.zero_grad(set_to_none=True)

            dist.all_reduce(step_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(step_tokens, op=dist.ReduceOp.SUM)
            train_loss = (step_loss / step_tokens.clamp_min(1)).item()
            lr = scheduler.get_last_lr()[0]

            if rank == 0:
                metrics = {"train/loss": train_loss, "train/lr": lr, "train/grad_norm": float(grad_norm)}
                iterator.set_postfix(loss=f"{train_loss:.4f}", lr=f"{lr:.2e}")
                if wandb_run is not None:
                    wandb_run.log(metrics, step=global_step)
                elif args.logger == "console":
                    print(f"step={global_step} loss={train_loss:.6f} lr={lr:.3e} grad_norm={float(grad_norm):.4f}")

            if val_loader is not None and args.val_freq > 0 and global_step % args.val_freq == 0:
                val_loss = validate(fsdp_model, val_loader, device)
                if rank == 0:
                    print(f"step={global_step} val_loss={val_loss:.6f}")
                    if wandb_run is not None:
                        wandb_run.log({"val/loss": val_loss}, step=global_step)

            if args.save_freq > 0 and global_step % args.save_freq == 0:
                save_checkpoint(args, global_step, model, fsdp_model, processor, rank)

            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break

    if val_loader is not None:
        val_loss = validate(fsdp_model, val_loader, device)
        if rank == 0:
            print(f"final val_loss={val_loss:.6f}")
            if wandb_run is not None:
                wandb_run.log({"val/loss": val_loss}, step=global_step)
    if args.save_final:
        save_checkpoint(args, global_step, model, fsdp_model, processor, rank)
    if wandb_run is not None:
        wandb_run.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
