"""
dataset_formatter.py — Convert forward_validated.jsonl to ChatML format for SLM fine-tuning.

Reads the synthetic dataset and transforms each record into the ChatML 3-turn structure:
  { "messages": [system, user, assistant] }

The system prompt is extracted from the winning DSPy-optimized program and stripped of
chain-of-thought (CoT) instructions, since for SFT we want the model to emit JSON directly.

Usage:
    python dataset_formatter.py \
        --input ../04_Synthetic_Data/generated/forward_validated.jsonl \
        --output data/train_chatml.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# System prompt: extracted from optimized_extractor_20260302_113921.json
# (signature.instructions), with the CoT reasoning line removed.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a personal finance information extractor specialized in Argentine "
    "(Rioplatense) Spanish. Given an informal message, possibly spanning multiple "
    "lines, with slang, varied number formats, and mixed currencies, analyze it "
    "and extract all financial transactions.\n\n"

    'Normalize colloquial amounts: "mil" = 1\u202f000, "lucas" = 1\u202f000 ARS '
    '(e.g. "15 lucas" = 15\u202f000), "gambas" = 100 ARS, "palos/palo" = '
    '1\u202f000\u202f000 ARS, "mangos"/"pe"/"pesitos" indicate ARS currency. '
    "Assume ARS if no currency is specified. Be robust to commas/dots in numbers.\n\n"

    'Infer type (EXPENSE vs INCOME) from verbs and context (e.g. "pagué", '
    '"me gasté" → EXPENSE; "cobré", "me pagaron" → INCOME). Apply category '
    'disambiguation: "facturas" are pastries (Supermercado_Despensa) unless '
    'clearly a utility bill (Vivienda_Servicios); "el chino"/"verduleria"/'
    '"almacen"/"despensa" → Supermercado_Despensa; "pedidos ya"/"rappi"/"lomito"/'
    '"pizza" → Comida_Comprada. Valid Categories: Comida_Comprada, '
    "Supermercado_Despensa, Deporte_Fitness, Educacion_Capacitacion, "
    "Inversiones_Finanzas, Vivienda_Servicios, Hogar_Mascotas, "
    "Salario_Honorarios, Salud_Bienestar, Transporte, Ocio_Entretenimiento, "
    "Regalos_Otros, Financiero_Tarjetas, Ropa_Accesorios. "
    "Valid Currencies: ARS, USD, EUR.\n\n"

    "Handle dates: if no date is mentioned, set date_delta_days = 0 and "
    'date_raw_expression = null; for relative references like "hoy" set 0, '
    '"ayer" set -1, and include the exact phrase in date_raw_expression. '
    "For absolute calendar dates (e.g. \"15/03\", \"el lunes 3 de julio\"), "
    "set date_delta_days = null and date_raw_expression to the exact phrase"
    "—do not compute a delta.\n\n"

    "If the message contains no financial content, return an empty JSON array.\n\n"

    "Return a JSON array of one object per transaction with exactly these fields:\n"
    "- description (string, cleaned brief description)\n"
    "- amount (float, positive)\n"
    "- currency (string, one of Valid Currencies)\n"
    "- category (string, one of Valid Categories)\n"
    '- type ("EXPENSE" or "INCOME")\n'
    "- date_delta_days (integer or null as above)\n"
    "- date_raw_expression (string or null)"
)


def format_record(record: dict) -> dict:
    """Convert a single forward_validated record into ChatML format.

    Args:
        record: A dict with at least 'input' and 'targets' keys.

    Returns:
        A dict with a 'messages' key containing the ChatML 3-turn structure.
    """
    user_message: str = record["input"]
    targets: list = record["targets"]

    # Serialize targets as compact JSON (no extra whitespace, preserve unicode)
    assistant_response: str = json.dumps(targets, ensure_ascii=False)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_response},
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert forward_validated.jsonl to ChatML format for SFT."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="../04_Synthetic_Data/generated/forward_validated.jsonl",
        help="Path to the input JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/train_chatml.jsonl",
        help="Path to the output ChatML JSONL file.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Process ---
    total: int = 0
    txn_counts: dict[int, int] = {}  # num_transactions -> count of records
    skipped: int = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line_num, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"⚠️  Skipping line {line_num}: JSON parse error — {e}",
                      file=sys.stderr)
                skipped += 1
                continue

            if "input" not in record or "targets" not in record:
                print(f"⚠️  Skipping line {line_num}: missing 'input' or 'targets'",
                      file=sys.stderr)
                skipped += 1
                continue

            chatml_record = format_record(record)
            fout.write(json.dumps(chatml_record, ensure_ascii=False) + "\n")

            num_txns = len(record["targets"])
            txn_counts[num_txns] = txn_counts.get(num_txns, 0) + 1
            total += 1

    # --- Stats ---
    print(f"\n{'='*50}")
    print(f"✅ ChatML conversion complete")
    print(f"{'='*50}")
    print(f"Input:   {input_path}")
    print(f"Output:  {output_path}")
    print(f"Total:   {total:,} records")
    if skipped:
        print(f"Skipped: {skipped:,} records")
    print(f"\nTransaction distribution:")
    for n_txns in sorted(txn_counts.keys()):
        label = "non-transactional" if n_txns == 0 else f"{n_txns} transaction(s)"
        print(f"  {label}: {txn_counts[n_txns]:,} records")


if __name__ == "__main__":
    main()
