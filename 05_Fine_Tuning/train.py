"""
train.py — Fine-tune Qwen3.5 on Quipu ChatML data using Unsloth + LoRA.

Architecture notes:
  - Qwen3.5 uses a hybrid Gated DeltaNet + Attention architecture.
  - Unsloth does NOT recommend QLoRA (4-bit) for Qwen3.5. We use bfloat16.
  - Thinking mode is disabled by default for Qwen3.5 "Small" models (≤4B).
    Training on plain JSON-only assistant responses reinforces non-thinking output.

RunPod setup:
  pip install -r requirements_training.txt

Usage:
  # Train and save only LoRA adapters (default):
  python train.py \\
      --model Qwen/Qwen3.5-0.8B \\
      --data data/train_chatml.jsonl \\
      --output outputs/qwen35-0.8b-quipu \\
      --epochs 2

  # Merge LoRA into base model weights (required for vLLM serving):
  python train.py \\
      --model Qwen/Qwen3.5-0.8B \\
      --data data/train_chatml.jsonl \\
      --output outputs/qwen35-0.8b-quipu-merged \\
      --epochs 2 \\
      --merge-16bit

For a quick smoke test (validates pipeline, no real learning):
  python train.py --data data/train_chatml.jsonl --max-steps 50
"""

import unsloth
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LoRA target modules for Qwen3.5 (attention + MLP projections)
# DeltaNet linear-attention layers are excluded — only standard attention
# and FFN layers are targeted for stability with the hybrid architecture.
# ---------------------------------------------------------------------------
LORA_TARGET_MODULES: list[str] = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3.5-0.8B with Unsloth LoRA on Quipu data."
    )
    # Data & model
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-0.8B",
                        help="HuggingFace model ID.")
    parser.add_argument("--data", type=str, default="data/train_chatml.jsonl",
                        help="Path to ChatML JSONL dataset.")
    parser.add_argument("--output", type=str, default="outputs/qwen35-0.8b-quipu",
                        help="Directory to save the LoRA adapter and tokenizer.")

    # Training
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs (1–3 recommended).")
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Override epochs with a fixed step count (useful for smoke tests).")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-device training batch size.")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps. Effective batch = batch_size × grad_accum.")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Peak learning rate.")
    parser.add_argument("--max-seq-len", type=int, default=1024,
                        help="Maximum token sequence length.")
    parser.add_argument("--eval-split", type=float, default=0.1,
                        help="Fraction of data to use for validation (0.0 = disable eval).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")

    # LoRA
    parser.add_argument("--lora-r", type=int, default=16,
                        help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="LoRA alpha. Recommended: 2 × lora_r.")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                        help="LoRA dropout.")

    # Export
    parser.add_argument("--merge-16bit", action="store_true", default=False,
                        help="Merge LoRA adapters into base model weights and save as a "
                             "full bfloat16 HuggingFace checkpoint. Required for vLLM serving. "
                             "Omit to save LoRA adapters only.")

    return parser.parse_args()


def load_dataset(data_path: str, eval_split: float, seed: int) -> tuple[Dataset, Dataset | None]:
    """Load and split the ChatML JSONL dataset.

    Args:
        data_path: Path to the train_chatml.jsonl file.
        eval_split: Fraction of data to reserve for evaluation (0.0 = all train).
        seed: Random seed for reproducible splitting.

    Returns:
        A tuple (train_dataset, eval_dataset). eval_dataset is None if eval_split == 0.
    """
    logger.info("Loading dataset from %s ...", data_path)
    records: list[dict] = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    logger.info("Loaded %d records.", len(records))
    dataset = Dataset.from_list(records)

    if eval_split > 0.0:
        split = dataset.train_test_split(test_size=eval_split, seed=seed)
        logger.info(
            "Split → train: %d | eval: %d",
            len(split["train"]), len(split["test"]),
        )
        return split["train"], split["test"]

    return dataset, None


def format_chat(example: dict, tokenizer) -> dict:
    """Apply the model's chat template to a single ChatML example.

    Args:
        example: A dict with a 'messages' key (list of role/content dicts).
        tokenizer: The model tokenizer with a chat template.

    Returns:
        A dict with a 'text' key containing the formatted string.
    """
    text: str = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def main() -> None:
    """Main training entrypoint."""
    args = parse_args()

    # --- Validate inputs ---
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error("Dataset not found: %s", data_path)
        sys.exit(1)

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Quipu SLM Fine-Tuning")
    logger.info("  Model:       %s", args.model)
    logger.info("  Data:        %s", data_path)
    logger.info("  Output:      %s", output_path)
    logger.info("  Epochs:      %s", args.epochs if args.max_steps < 0 else f"overridden by --max-steps={args.max_steps}")
    logger.info("  Batch size:  %d (× %d accum = %d effective)", args.batch_size, args.grad_accum, args.batch_size * args.grad_accum)
    logger.info("  LoRA:        r=%d  alpha=%d  dropout=%.2f", args.lora_r, args.lora_alpha, args.lora_dropout)
    logger.info("=" * 60)

    # --- Load model + tokenizer (bfloat16, no quantization) ---
    logger.info("Loading model %s in bfloat16 (no quantization) ...", args.model)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        load_in_4bit=False,       # Unsloth doesn't recommend 4-bit for Qwen3.5
        load_in_8bit=False,       # Full bfloat16 for stability
        dtype=torch.bfloat16,
    )

    # Apply Qwen3.5 chat template (handles thinking mode suppression)
    tokenizer = get_chat_template(tokenizer, chat_template="chatml")

    # --- Add LoRA adapters ---
    logger.info("Applying LoRA (r=%d) to target modules: %s", args.lora_r, LORA_TARGET_MODULES)
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Reduces VRAM significantly
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Trainable params: %s / %s (%.2f%%)",
        f"{trainable_params:,}", f"{total_params:,}",
        100 * trainable_params / total_params,
    )

    # --- Load and format dataset ---
    train_dataset, eval_dataset = load_dataset(args.data, args.eval_split, args.seed)

    train_dataset = train_dataset.map(
        lambda ex: format_chat(ex, tokenizer),
        desc="Formatting train set",
    )
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(
            lambda ex: format_chat(ex, tokenizer),
            desc="Formatting eval set",
        )

    # --- Configure SFTTrainer ---
    use_eval = eval_dataset is not None
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)] if use_eval else []

    training_args = SFTConfig(
        output_dir=str(output_path / "checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,            # -1 = use epochs
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="epoch" if use_eval else "no",
        eval_strategy="epoch" if use_eval else "no",
        load_best_model_at_end=use_eval,
        metric_for_best_model="eval_loss" if use_eval else None,
        greater_is_better=False,
        seed=args.seed,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        packing=True,                        # Efficient batching for short sequences
        report_to="wandb",                    # Disable W&B / MLflow by default
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        callbacks=callbacks,
    )

    # --- Train ---
    logger.info("Starting training ...")
    start_time = time.time()
    train_result = trainer.train()
    elapsed = time.time() - start_time

    # --- Save model ---
    if args.merge_16bit:
        # Merge LoRA weights into the base model for direct vLLM loading.
        # Produces a standard HuggingFace checkpoint (config.json + .safetensors),
        # not a PEFT adapter. Requires more disk space but no runtime PEFT dependency.
        logger.info(
            "Merging LoRA into base model (bfloat16) and saving to %s ...", output_path
        )
        model.save_pretrained_merged(
            str(output_path), tokenizer, save_method="merged_16bit"
        )
        tokenizer.save_pretrained(str(output_path))
        save_mode = "merged_16bit"
    else:
        # Save only the LoRA adapter weights (lightweight, ~tens of MB).
        logger.info("Saving LoRA adapter to %s ...", output_path)
        model.save_pretrained(str(output_path))
        tokenizer.save_pretrained(str(output_path))
        save_mode = "lora_adapter"

    # --- Save training summary JSON ---
    summary = {
        "model": args.model,
        "data": str(data_path),
        "output": str(output_path),
        "save_mode": save_mode,
        "merge_16bit": args.merge_16bit,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "lr": args.lr,
        "max_seq_len": args.max_seq_len,
        "eval_split": args.eval_split,
        "seed": args.seed,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset) if eval_dataset else 0,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_pct": round(100 * trainable_params / total_params, 4),
        "train_loss": round(train_result.training_loss, 6),
        "elapsed_seconds": round(elapsed, 1),
    }
    summary_path = output_path / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info("Training complete in %.1f min", elapsed / 60)
    logger.info("  Train loss:  %.4f", train_result.training_loss)
    logger.info("  Save mode:   %s", save_mode)
    logger.info("  Output:      %s", output_path)
    logger.info("  Summary:     %s", summary_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
