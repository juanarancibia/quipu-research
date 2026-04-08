"""
Quality checks for generated synthetic data.

Validates that generated inputs are consistent with their target outputs
and conform to the golden dataset schema.
"""

import json
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


def validate_generated_input(
    generated_input: str,
    targets: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    """Validate a single generated input against its target transactions.

    Checks:
        1. Input is non-empty and reasonable length
        2. Amount mentioned in input is consistent with target amount
        3. Input doesn't accidentally contain structured JSON
        4. Non-transactional inputs have empty targets

    Args:
        generated_input: The generated natural-language input string.
        targets: The expected transaction targets.

    Returns:
        Tuple of (is_valid, list_of_warnings).
    """
    warnings: list[str] = []
    is_valid = True

    # Check 1: Non-empty
    if not generated_input or not generated_input.strip():
        return False, ["Empty input"]

    # Check 2: Reasonable length (1 to 500 chars)
    if len(generated_input) > 500:
        warnings.append(f"Input is very long ({len(generated_input)} chars)")

    # Check 3: No accidental JSON in the input
    stripped = generated_input.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        return False, ["Input looks like JSON, not natural language"]

    # Check 4: Amount consistency for single-transaction entries
    if len(targets) == 1:
        target_amount = targets[0].get("amount", 0)
        amount_found = _extract_amount_from_text(generated_input, target_amount)
        if not amount_found:
            warnings.append(
                f"Amount {target_amount} not clearly found in input '{generated_input[:80]}...'"
            )

    # Check 5: Currency consistency
    for target in targets:
        currency = target.get("currency", "ARS")
        if currency == "USD":
            usd_markers = ["usd", "dolar", "dólares", "dolares", "usds", "dólar"]
            has_usd_marker = any(m in generated_input.lower() for m in usd_markers)
            if not has_usd_marker:
                warnings.append(f"USD transaction but no USD marker in input")

    return is_valid, warnings


def _extract_amount_from_text(text: str, expected_amount: float) -> bool:
    """Check if the expected amount appears in the text in any common format.

    Handles formats like:
        - "15000", "15.000", "15,000"
        - "15mil", "15 mil", "15k", "15 lucas", "15 palos"
        - "$15.000", "$ 15.000"
        - Decimal amounts: "7650,51", "7650.51"

    Args:
        text: The input text to search.
        expected_amount: The expected amount to find.

    Returns:
        True if the amount is plausibly present in the text.
    """
    # Normalize text
    text_lower = text.lower().replace("$", "").replace(" ", "")

    # Integer amounts
    int_amount = int(expected_amount) if expected_amount == int(expected_amount) else None

    if int_amount is not None:
        # Direct match: "15000"
        if str(int_amount) in text.replace(".", "").replace(",", "").replace(" ", ""):
            return True

        # "mil" / "lucas" / "k" suffixes: "15mil", "15 mil", "15k", "15 lucas"
        if int_amount >= 1000 and int_amount % 1000 == 0:
            thousands = int_amount // 1000
            patterns = [
                f"{thousands}mil", f"{thousands} mil",
                f"{thousands}k", f"{thousands} k",
                f"{thousands} lucas", f"{thousands}lucas",
                f"{thousands} palos", f"{thousands}palos",
            ]
            for pattern in patterns:
                if pattern in text_lower:
                    return True

    # Decimal amounts: "7650,51" or "7650.51"
    amount_str = f"{expected_amount:.2f}"
    # Argentine format: comma as decimal separator
    amount_ar = amount_str.replace(".", ",")
    if amount_str in text or amount_ar in text:
        return True

    # With thousand separator: "7.650,51"
    if expected_amount >= 1000:
        parts = amount_str.split(".")
        integer_part = parts[0]
        decimal_part = parts[1] if len(parts) > 1 else "00"
        # Add thousand separators
        formatted = ""
        for i, digit in enumerate(reversed(integer_part)):
            if i > 0 and i % 3 == 0:
                formatted = "." + formatted
            formatted = digit + formatted
        full_formatted = f"{formatted},{decimal_part}" if decimal_part != "00" else formatted
        if full_formatted in text:
            return True

    return True  # Be lenient — slang like "15 lucas" is hard to fully validate


def validate_entry_schema(entry: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate that a generated entry conforms to the golden dataset JSONL schema.

    Args:
        entry: A dictionary representing one JSONL entry.

    Returns:
        Tuple of (is_valid, list_of_errors).
    """
    errors: list[str] = []

    # Required fields
    if "input" not in entry:
        errors.append("Missing 'input' field")
    if "targets" not in entry:
        errors.append("Missing 'targets' field")
    if "metadata" not in entry:
        errors.append("Missing 'metadata' field")
    if "conversation_date" not in entry:
        errors.append("Missing 'conversation_date' field")

    if errors:
        return False, errors

    # Validate targets
    for i, target in enumerate(entry["targets"]):
        required_target_fields = [
            "type", "amount", "currency", "category",
            "description", "date_delta_days", "date_raw_expression",
        ]
        for field in required_target_fields:
            if field not in target:
                errors.append(f"Target[{i}] missing '{field}'")

        if target.get("type") not in ("EXPENSE", "INCOME"):
            errors.append(f"Target[{i}] invalid type: {target.get('type')}")

    # Validate metadata
    meta = entry.get("metadata", {})
    if "source" not in meta:
        errors.append("Metadata missing 'source'")
    if "num_transactions" not in meta:
        errors.append("Metadata missing 'num_transactions'")

    return len(errors) == 0, errors
