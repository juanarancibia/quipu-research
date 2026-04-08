"""
bakeoff_pipeline.py — Automated Model Bake-off for Quipu SLM variants.

For each model in MODELS_TO_TEST, this pipeline runs two sequential phases:

  Phase A — Fine-Tuning:  Train the model using train.py with --merge-16bit.
  Phase B — Uploading:    Automatically upload the merged model to Hugging Face Hub.

After all models are processed a comparative summary is written to:
  bakeoff_summary_report.md   (human-readable Markdown table)
  bakeoff_summary_report.json (machine-readable)

Usage:
  python bakeoff_pipeline.py
  python bakeoff_pipeline.py --dry-run          # Skip actual subprocesses
  python bakeoff_pipeline.py --epochs 1         # Override training epochs
  python bakeoff_pipeline.py --hf-user USER     # Hugging Face username or org for upload
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import TypeAlias

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bakeoff")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODELS_TO_TEST: list[str] = [
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B"
]

_HERE = Path(__file__).parent.resolve()
TRAIN_SCRIPT: Path = _HERE / "train.py"
OUTPUT_BASE: Path = _HERE / "outputs"
REPORT_DIR: Path = _HERE


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
ModelResult: TypeAlias = dict[str, Any]

RUN_TIMESTAMP: str = datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _safe_model_dir_name(model_id: str) -> str:
    """Convert a HuggingFace model ID to a filesystem-safe directory name."""
    short_name = model_id.split("/")[-1]
    return f"{short_name}-quipu-merged_{RUN_TIMESTAMP}"


def _find_latest_model_output_dir(model_id: str) -> Path | None:
    """Find the newest existing output directory for the given model ID."""
    short_name = model_id.split("/")[-1]
    pattern = f"{short_name}-quipu-merged_*"
    candidates = [
        p for p in OUTPUT_BASE.glob(pattern)
        if p.is_dir()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


# ---------------------------------------------------------------------------
# Phase implementations
# ---------------------------------------------------------------------------
def phase_a_finetune(
    model_id: str,
    output_dir: Path,
    epochs: int,
    data_path: Path,
    dry_run: bool,
) -> bool:
    """Phase A: Fine-tune the model and merge LoRA into 16-bit weights."""
    cmd: list[str] = [
        sys.executable, str(TRAIN_SCRIPT),
        "--model", model_id,
        "--data", str(data_path),
        "--output", str(output_dir),
        "--epochs", str(epochs),
        "--merge-16bit",
    ]
    logger.info("[Phase A] Fine-tuning %s → %s", model_id, output_dir)
    logger.info("[Phase A] Command: %s", " ".join(cmd))

    if dry_run:
        logger.info("[Phase A] DRY-RUN — skipping subprocess.")
        return True

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.error("[Phase A] train.py failed with exit code %d.", result.returncode)
        return False

    logger.info("[Phase A] Fine-tuning complete.")
    return True


def phase_b_upload(
    output_dir: Path,
    repo_name: str,
    dry_run: bool,
) -> bool:
    """Phase B: Upload the merged model to Hugging Face Hub."""
    cmd: list[str] = [
        "huggingface-cli", "upload",
        repo_name,
        str(output_dir),
        "--repo-type", "model",
        "--private"
    ]
    logger.info("[Phase B] Uploading %s to Hugging Face repo %s ...", output_dir, repo_name)
    logger.info("[Phase B] Command: %s", " ".join(cmd))

    if dry_run:
        logger.info("[Phase B] DRY-RUN — skipping Hugging Face upload.")
        return True

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.error("[Phase B] Hugging Face upload failed with exit code %d.", result.returncode)
        return False

    logger.info("[Phase B] Upload complete.")
    return True


# ---------------------------------------------------------------------------
# Per-model orchestration
# ---------------------------------------------------------------------------
def run_model_bakeoff(
    model_id: str,
    epochs: int,
    data_path: Path,
    dry_run: bool,
    skip_train: bool,
    hf_user: str | None,
) -> ModelResult:
    """Run both phases for a single model."""
    output_dir: Path | None = None

    logger.info("=" * 60)
    logger.info("BAKEOFF | Model: %s", model_id)
    logger.info("=" * 60)

    try:
        if skip_train:
            found_dir = _find_latest_model_output_dir(model_id)
            if not found_dir:
                msg = f"--skip-train requested but no existing weights found for {model_id} in {OUTPUT_BASE}"
                logger.error("[Phase A] %s", msg)
                return {"model": model_id, "error": msg}

            output_dir = found_dir
            logger.info("[Phase A] --skip-train provided. Skipping training.")
            logger.info("          ↳ Using existing weights: %s", output_dir)
        else:
            output_dir = OUTPUT_BASE / _safe_model_dir_name(model_id)
            # Phase A — Fine-Tuning
            success = phase_a_finetune(model_id, output_dir, epochs, data_path, dry_run)
            if not success:
                return {"model": model_id, "output_dir": str(output_dir), "error": "Phase A failed"}

        # Phase B — Upload
        hf_repo_name = f"{hf_user}/{output_dir.name}" if hf_user else output_dir.name
        success = phase_b_upload(output_dir, hf_repo_name, dry_run)
        if not success:
            return {"model": model_id, "output_dir": str(output_dir), "error": "Phase B failed"}

        return {
            "model": model_id,
            "output_dir": str(output_dir),
            "hf_repo": hf_repo_name,
        }

    except Exception as exc:
        logger.exception("[Bakeoff] Unexpected error for model %s: %s", model_id, exc)
        return {"model": model_id, "output_dir": str(output_dir), "error": str(exc)}


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------
def generate_summary_report(model_results: list[ModelResult]) -> None:
    """Compile results from all models and write Markdown + JSON reports."""
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    # --- JSON report ---
    json_report: dict[str, Any] = {"generated_at": timestamp, "results": model_results}
    json_path = REPORT_DIR / f"bakeoff_summary_report_{RUN_TIMESTAMP}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2, ensure_ascii=False)
    logger.info("JSON report written to %s", json_path)

    # --- Markdown report ---
    md_path = REPORT_DIR / f"bakeoff_summary_report_{RUN_TIMESTAMP}.md"
    header = (
        "# Quipu SLM Bake-off — Summary Report\n\n"
        f"_Generated: {timestamp}_\n\n"
    )
    columns = ["model", "status", "hf_repo", "output_dir", "error"]
    col_headers = " | ".join(f"**{c}**" for c in columns)
    separator = " | ".join("---" for _ in columns)

    table_rows: list[str] = []
    for row in model_results:
        st = "❌ Failed" if "error" in row else "✅ Success"
        err = row.get("error", "")
        cells = [
            str(row.get("model", "N/A")),
            st,
            str(row.get("hf_repo", "N/A")),
            str(row.get("output_dir", "N/A")),
            err
        ]
        table_rows.append(" | ".join(cells))

    md_content = (
        header
        + "## Results\n\n"
        + f"| {col_headers} |\n"
        + f"| {separator} |\n"
        + "\n".join(f"| {r} |" for r in table_rows)
        + "\n"
    )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info("Markdown report written to %s", md_path)


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Automated Quipu Model Bake-off Pipeline."
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="List of model IDs to test (e.g. 0.8B 2B). If not provided, tests all defaults.",
    )
    parser.add_argument(
        "--skip-train", action="store_true", default=False,
        help="Skip Phase A (training) and try to use most recent existing weights from outputs/.",
    )
    parser.add_argument(
        "--epochs", type=int, default=2,
        help="Training epochs for each model (default: 2).",
    )
    parser.add_argument(
        "--hf-user", type=str, default=None,
        help="Hugging Face username or org for upload (e.g., 'Jarancibia').",
    )
    parser.add_argument(
        "--data", type=str, default=str(_HERE / "data" / "train_chatml.jsonl"),
        help="Path to the ChatML JSONL training dataset.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Log all commands but skip actual subprocess execution.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the bakeoff pipeline."""
    args = parse_args()

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | bakeoff | %(message)s",
        datefmt="%H:%M:%S",
    )

    data_path = Path(args.data)

    # Mapeador de shorthands a IDs de HuggingFace
    MODEL_MAP = {
        "0.8B": "Qwen/Qwen3.5-0.8B",
        "2B":   "Qwen/Qwen3.5-2B",
        "4B":   "Qwen/Qwen3.5-4B",
        "9B":   "Qwen/Qwen3.5-9B"
    }

    if args.models:
        models_to_run = [MODEL_MAP.get(m, m) for m in args.models]
    else:
        models_to_run = MODELS_TO_TEST

    if not args.dry_run and not data_path.exists():
        logger.error("Training dataset not found: %s", data_path)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Quipu Model Bake-off Pipeline")
    logger.info("  Models:     %s", models_to_run)
    logger.info("  Epochs:     %d", args.epochs)
    logger.info("  HF User:    %s", args.hf_user if args.hf_user else "default authenticated user")
    logger.info("  Dry-run:    %s", args.dry_run)
    logger.info("  Skip-Train: %s", args.skip_train)
    logger.info("=" * 60)

    all_results: list[ModelResult] = []

    for model_id in models_to_run:
        result = run_model_bakeoff(
            model_id=model_id,
            epochs=args.epochs,
            data_path=data_path,
            dry_run=args.dry_run,
            skip_train=args.skip_train,
            hf_user=args.hf_user,
        )
        all_results.append(result)
        status = "✓" if "error" not in result else f"✗ {result['error']}"
        logger.info("Completed %s → %s", model_id, status)

    logger.info("=" * 60)
    logger.info("All models processed. Generating summary report ...")
    generate_summary_report(all_results)
    logger.info("Bakeoff pipeline finished.")


if __name__ == "__main__":
    main()
