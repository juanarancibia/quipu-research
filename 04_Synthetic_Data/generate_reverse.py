"""
Strategy 1: Output → Input Reverse Generation.

Takes existing validated outputs from golden_dataset.jsonl, asks an LLM to
generate diverse natural-language inputs that would produce the same outputs.
Appends validated results directly to the golden dataset.

Usage:
    python generate_reverse.py                         # Auto-detect gaps, generate
    python generate_reverse.py --model gpt-4o-mini     # Specific model
    python generate_reverse.py --dry-run               # Show plan without generating
    python generate_reverse.py --limit 10              # Generate for 10 seeds max
    python generate_reverse.py --non-tx 15             # Also generate 15 non-transactional
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import litellm

from balance_analyzer import BalanceReport, analyze_dataset, select_entries_for_generation
from config import GenerationConfig
from prompts.reverse_generation import (
    MULTI_TX_DESCRIPTION,
    MULTI_TX_INSTRUCTION,
    NON_TRANSACTIONAL_SYSTEM_PROMPT,
    NON_TRANSACTIONAL_USER_PROMPT,
    REVERSE_SYSTEM_PROMPT,
    REVERSE_USER_PROMPT,
    SINGLE_TX_DESCRIPTION,
)
from validators.quality_checks import validate_entry_schema, validate_generated_input

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_input_variants(
    targets: list[dict[str, Any]],
    n_variants: int,
    config: GenerationConfig,
) -> list[str]:
    """Call the LLM to generate diverse input variants for given transaction targets.

    Args:
        targets: The structured transaction targets to generate inputs for.
        n_variants: Number of input variants to generate.
        config: Generation configuration.

    Returns:
        A list of generated input strings.
    """
    # Prepare the transaction JSON for the prompt
    # Remove fields irrelevant to input generation
    clean_targets = []
    for t in targets:
        clean_t = {
            "type": t["type"],
            "amount": t["amount"],
            "currency": t["currency"],
            "category": t["category"],
            "description": t["description"],
            "date_delta_days": t.get("date_delta_days"),
            "date_raw_expression": t.get("date_raw_expression"),
        }
        clean_targets.append(clean_t)

    transaction_json = json.dumps(clean_targets, ensure_ascii=False, indent=2)

    # Build multi-transaction-aware prompt
    n_tx = len(clean_targets)
    if n_tx > 1:
        tx_description = MULTI_TX_DESCRIPTION.format(n=n_tx)
        multi_tx_instruction = MULTI_TX_INSTRUCTION.format(n=n_tx)
    else:
        tx_description = SINGLE_TX_DESCRIPTION
        multi_tx_instruction = ""

    system_prompt = REVERSE_SYSTEM_PROMPT.format(n_variants=n_variants)
    user_prompt = REVERSE_USER_PROMPT.format(
        n_variants=n_variants,
        tx_description=tx_description,
        transaction_json=transaction_json,
        multi_tx_instruction=multi_tx_instruction,
    )

    try:
        response = litellm.completion(
            model=config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=config.temperature,
        )
        raw_output = response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return []

    # Parse the response
    return _parse_string_array(raw_output)


def generate_non_transactional(
    n_variants: int,
    config: GenerationConfig,
) -> list[str]:
    """Generate non-transactional messages (negative examples).

    Args:
        n_variants: Number of messages to generate.
        config: Generation configuration.

    Returns:
        A list of non-transactional input strings.
    """
    system_prompt = NON_TRANSACTIONAL_SYSTEM_PROMPT.format(n_variants=n_variants)
    user_prompt = NON_TRANSACTIONAL_USER_PROMPT.format(n_variants=n_variants)

    try:
        response = litellm.completion(
            model=config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=config.temperature,
        )
        raw_output = response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"LLM call for non-transactional failed: {e}")
        return []

    return _parse_string_array(raw_output)


def _parse_string_array(raw_output: str) -> list[str]:
    """Parse an LLM response that should be a JSON array of strings.

    Args:
        raw_output: Raw text from the LLM.

    Returns:
        A list of strings, or empty list if parsing fails.
    """
    cleaned = raw_output.strip()
    # Strip markdown code fences if present
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM output as JSON: {e}")
        logger.debug(f"Raw output: {raw_output[:300]}")
        return []

    if not isinstance(parsed, list):
        logger.error(f"Expected JSON array, got {type(parsed)}")
        return []

    return [str(item) for item in parsed if isinstance(item, str) and item.strip()]


def build_entry(
    generated_input: str,
    targets: list[dict[str, Any]],
    conversation_date: str,
    source_tag: str,
) -> dict[str, Any]:
    """Build a golden dataset entry from a generated input and existing targets.

    Args:
        generated_input: The generated natural-language input.
        targets: The validated transaction targets.
        conversation_date: ISO date string for the entry.
        source_tag: Metadata source identifier.

    Returns:
        A dictionary conforming to the golden dataset JSONL schema.
    """
    return {
        "input": generated_input,
        "conversation_date": conversation_date,
        "targets": targets,
        "metadata": {
            "source": source_tag,
            "num_transactions": len(targets),
        },
    }


def run_reverse_generation(
    config: GenerationConfig,
    limit: int | None = None,
    non_tx_count: int = 0,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Main entry point for reverse generation.

    Analyzes the dataset for gaps, generates input variants for
    underrepresented categories, validates them, and returns the results.

    Args:
        config: Generation configuration.
        limit: Max number of seed entries to process.
        non_tx_count: Number of non-transactional entries to generate.
        dry_run: If True, show the plan without generating.

    Returns:
        A list of generated entries ready to append to the golden dataset.
    """
    # Step 1: Analyze current dataset balance
    logger.info(f"📊 Analyzing dataset: {config.golden_dataset_path}")
    report = analyze_dataset(config.golden_dataset_path)
    print(report.summary())

    # Load all entries for seed selection
    with open(config.golden_dataset_path, "r", encoding="utf-8") as f:
        all_entries = [json.loads(line) for line in f if line.strip()]

    # Step 2: Build generation plan
    plan = select_entries_for_generation(report, all_entries)
    if limit:
        plan = plan[:limit]

    total_variants = sum(n for _, n in plan)
    print(f"\n🎯 Generation plan: {len(plan)} seed entries → {total_variants} new inputs")
    if non_tx_count > 0:
        print(f"   + {non_tx_count} non-transactional messages")

    if dry_run:
        print("\n📋 Planned generations:")
        for entry, n_variants in plan:
            targets = entry.get("targets", [])
            cat = targets[0]["category"] if targets else "non-tx"
            typ = targets[0]["type"] if targets else "N/A"
            print(f"   [{typ}:{cat}] '{entry['input'][:60]}...' → {n_variants} variants")
        return []

    # Step 3: Generate
    generated_entries: list[dict[str, Any]] = []
    total_warnings = 0
    total_rejected = 0

    print(f"\n🔄 Generating with model: {config.model}")
    print(f"   Temperature: {config.temperature}")
    print()

    for i, (seed_entry, n_variants) in enumerate(plan):
        targets = seed_entry.get("targets", [])
        cat = targets[0]["category"] if targets else "?"
        typ = targets[0]["type"] if targets else "?"
        conversation_date = seed_entry.get("conversation_date", "2026-03-01")

        logger.info(
            f"[{i + 1}/{len(plan)}] Generating {n_variants} variants for "
            f"[{typ}:{cat}] '{seed_entry['input'][:50]}...'"
        )

        variants = generate_input_variants(targets, n_variants, config)

        for variant in variants:
            # Validate
            is_valid, warnings = validate_generated_input(variant, targets)
            if not is_valid:
                logger.warning(f"  ❌ Rejected: '{variant[:60]}...' — {warnings}")
                total_rejected += 1
                continue

            if warnings:
                total_warnings += len(warnings)
                for w in warnings:
                    logger.info(f"  ⚠️  Warning: {w}")

            entry = build_entry(variant, targets, conversation_date, config.source_tag)

            # Schema validation
            schema_valid, schema_errors = validate_entry_schema(entry)
            if not schema_valid:
                logger.warning(f"  ❌ Schema invalid: {schema_errors}")
                total_rejected += 1
                continue

            generated_entries.append(entry)
            logger.info(f"  ✅ '{variant[:60]}...'")

    # Step 4: Generate non-transactional entries
    if non_tx_count > 0:
        logger.info(f"\n🚫 Generating {non_tx_count} non-transactional messages...")
        non_tx_messages = generate_non_transactional(non_tx_count, config)
        for msg in non_tx_messages:
            entry = build_entry(
                generated_input=msg,
                targets=[],
                conversation_date="2026-03-01",
                source_tag=config.source_tag,
            )
            generated_entries.append(entry)
            logger.info(f"  ✅ (non-tx) '{msg[:60]}...'")

    # Summary
    print(f"\n📊 Generation Summary:")
    print(f"   Generated: {len(generated_entries)} entries")
    print(f"   Rejected: {total_rejected}")
    print(f"   Warnings: {total_warnings}")

    return generated_entries


def append_to_dataset(
    entries: list[dict[str, Any]],
    dataset_path: Path,
) -> None:
    """Append generated entries to the golden dataset JSONL file.

    Args:
        entries: List of entries to append.
        dataset_path: Path to golden_dataset.jsonl.
    """
    with open(dataset_path, "a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"✅ Appended {len(entries)} entries to {dataset_path}")


def save_to_file(
    entries: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Save generated entries to a separate JSONL file (for review before appending).

    Args:
        entries: List of entries to save.
        output_path: Path to the output file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"💾 Saved {len(entries)} entries to {output_path}")


def main() -> None:
    """CLI entry point for reverse generation."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic inputs for existing golden dataset outputs."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="LiteLLM model string (auto-detected if not specified)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of seed entries to process",
    )
    parser.add_argument(
        "--non-tx",
        type=int,
        default=0,
        help="Number of non-transactional entries to generate",
    )
    parser.add_argument(
        "--variants",
        type=int,
        default=3,
        help="Default number of input variants per seed (default: 3)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="LLM temperature for generation (default: 0.8)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the generation plan without actually generating",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append generated entries directly to golden_dataset.jsonl",
    )
    parser.add_argument(
        "--source-tag",
        type=str,
        default="synthetic_reverse",
        help="Source tag for metadata (default: synthetic_reverse)",
    )

    args = parser.parse_args()

    config = GenerationConfig(
        model=args.model,
        variants_per_entry=args.variants,
        temperature=args.temperature,
        source_tag=args.source_tag,
    )

    print(f"🤖 Model: {config.model}")
    print(f"📁 Dataset: {config.golden_dataset_path}")
    print()

    generated = run_reverse_generation(
        config=config,
        limit=args.limit,
        non_tx_count=args.non_tx,
        dry_run=args.dry_run,
    )

    if not generated:
        if not args.dry_run:
            print("No entries generated.")
        return

    if args.append:
        append_to_dataset(generated, config.golden_dataset_path)
    else:
        save_to_file(generated, config.output_path)
        print(f"\n💡 To append to golden dataset, re-run with --append flag")
        print(f"   Or manually review {config.output_path} first")


if __name__ == "__main__":
    main()
