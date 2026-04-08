import argparse
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
import time
import numpy as np
import litellm
from typing import Optional
from dotenv import load_dotenv

# Load .env from script directory
load_dotenv(Path(__file__).parent / ".env")

# Import Evaluation types
from evaluator import Evaluator
from schemas import PredictionEntry, PredictionTarget

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Same categories as core/prompts
VALID_CATEGORIES = [
    "Comida_Comprada", "Supermercado_Despensa", "Deporte_Fitness",
    "Educacion_Capacitacion", "Inversiones_Finanzas", "Vivienda_Servicios",
    "Hogar_Mascotas", "Salario_Honorarios", "Salud_Bienestar",
    "Transporte", "Ocio_Entretenimiento", "Regalos_Otros",
    "Financiero_Tarjetas", "Ropa_Accesorios"
]

VALID_CURRENCIES = ["ARS", "USD", "EUR"]

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

def build_system_prompt() -> str:
    return SYSTEM_PROMPT

def build_user_prompt(content: str, conversation_date: str) -> str:
    return content

def build_few_shot_examples(dataset: list[dict], exclude_idx: int, n: int, rng: random.Random) -> list[dict]:
    """Build few-shot example messages from the golden dataset.
    
    Returns a list of alternating user/assistant message dicts
    to inject between the system prompt and the real user input.
    """
    candidates = [r for i, r in enumerate(dataset) if i != exclude_idx and r.get("targets")]
    selected = rng.sample(candidates, min(n, len(candidates)))
    
    messages = []
    for example in selected:
        messages.append({"role": "user", "content": example["input"]})
        messages.append({"role": "assistant", "content": json.dumps(example["targets"], ensure_ascii=False)})
    return messages

def parse_model_response(raw_response: str) -> Optional[list[PredictionTarget]]:
    cleaned = raw_response.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        parsed_json = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
        
    items = parsed_json if isinstance(parsed_json, list) else [parsed_json]
    
    targets = []
    for item in items:
        if not isinstance(item, dict):
            return None
                    
        target: PredictionTarget = {
            "type": item.get("type", "EXPENSE").upper(),
            "amount": float(item.get("amount", 0) or 0.0),
            "currency": item.get("currency", "ARS"),
            "category": item.get("category", ""),
            "description": item.get("description", ""),
            "date_delta_days": item.get("date_delta_days"),
            "date_raw_expression": item.get("date_raw_expression")
        }
        targets.append(target)
        
    return targets

def main():
    parser = argparse.ArgumentParser(description="Evaluate a model against golden_dataset.jsonl")
    parser.add_argument("--model", type=str, required=True, help="LiteLLM model string (e.g. gpt-5-mini, openrouter/qwen/qwen3.5-4b)")
    parser.add_argument("--limit", type=int, default=None, help="Number of random dataset entries to test (default: all)")
    parser.add_argument("--dataset", type=str, default="../golden_dataset.jsonl", help="Path to golden_dataset.jsonl")
    parser.add_argument("--api-base", type=str, default=None, help="Optional litellm api_base for standard local models")
    parser.add_argument("--api-key", type=str, default=None, help="Optional API key for HuggingFace/OpenAI endpoints")
    parser.add_argument("--no-prompt", action="store_true", help="Evaluate a fine-tuned model by passing only the user input directly, skipping the system prompt.")
    parser.add_argument("--few-shot", type=int, default=0, help="Number of few-shot examples to inject from the golden dataset (default: 0 = zero-shot)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible sampling")
    parser.add_argument("--debug", action="store_true", help="Enable verbose liteLLM debugging")
    args = parser.parse_args()

    if args.debug:
        litellm.set_verbose = True

    # Set seed for reproducibility
    rng = random.Random(args.seed)
    if args.seed is not None:
        random.seed(args.seed)
        logger.info(f"Random seed set to {args.seed}")

    evaluator = Evaluator(args.dataset)
    with open(args.dataset, 'r', encoding='utf-8') as f:
        all_records = [json.loads(line) for line in f if line.strip()]

    # Keep full dataset for few-shot sampling, but select subset for evaluation
    if args.limit and args.limit < len(all_records):
        logger.info(f"Randomly sampling {args.limit} entries from the dataset of {len(all_records)}.")
        eval_indices = rng.sample(range(len(all_records)), args.limit)
        dataset_records = [all_records[i] for i in eval_indices]
    else:
        logger.info(f"Evaluating all {len(all_records)} entries.")
        eval_indices = list(range(len(all_records)))
        dataset_records = all_records

    system_prompt = build_system_prompt()
    predictions: list[PredictionEntry] = []
    latencies: list[float] = []

    mode_info = " (PROMPT-LESS / FINE-TUNED MODE)" if args.no_prompt else ""
    if args.few_shot > 0:
        mode_info += f" (FEW-SHOT: {args.few_shot} examples)"
    logger.info(f"Starting evaluation run with model: {args.model}{mode_info}")
    
    # gpt-5 family (excluding gpt-5.1) only supports temperature=1.0
    model_lower = args.model.lower()
    if "gpt-5" in model_lower and "gpt-5.1" not in model_lower:
        temp = 1.0
        logger.info(f"Using temperature=1.0 (required for gpt-5 reasoning models)")
    else:
        temp = 0.0

    litellm_kwargs = {
        "model": args.model,
        "temperature": temp,
    }
    if args.api_base:
        litellm_kwargs["api_base"] = args.api_base
        # Fix custom OpenAI Endpoint in LiteLLM for local runs
        if not args.model.startswith("openai/"):
            litellm_kwargs["custom_llm_provider"] = "openai"
    if args.api_key:
        litellm_kwargs["api_key"] = args.api_key

    for i, record in enumerate(dataset_records):
        input_text = record["input"]
        conv_date = record.get("conversation_date", "2025-01-01")
        
        logger.info(f"Processing ({i+1}/{len(dataset_records)}): '{input_text}'")
        
        if args.no_prompt:
            messages = [
                {"role": "user", "content": input_text}
            ]
        else:
            user_prompt = build_user_prompt(input_text, conv_date)
            messages = [
                {"role": "system", "content": system_prompt},
            ]
            # Inject few-shot examples if requested
            if args.few_shot > 0:
                orig_idx = eval_indices[i] if i < len(eval_indices) else -1
                few_shot_msgs = build_few_shot_examples(all_records, orig_idx, args.few_shot, rng)
                messages.extend(few_shot_msgs)
            messages.append({"role": "user", "content": user_prompt})
        
        start_time = time.time()
        try:
            response = litellm.completion(
                messages=messages,
                **litellm_kwargs
            )
            raw_output = response.choices[0].message.content
        except Exception as e:
            logger.error(f"LiteLLM call failed for input '{input_text}': {e}")
            logger.error("--- DEBUG INFO ---")
            logger.error(f"URL Base sent: {litellm_kwargs.get('api_base')}")
            logger.error(f"Payload messages: {json.dumps(messages, ensure_ascii=False)}")
            logger.error(f"LiteLLM Kwargs: {litellm_kwargs}")
            logger.error("------------------")
            raw_output = ""
        finally:
            latency = time.time() - start_time
            latencies.append(latency)
            
        # Instead of strict JSON formatting we let Litellm handle standard models.
        # Format mapping generates our expected PredictionTarget type.
        # By formatting output as json, we can get pure representation.
        # But some models won't support response_format strict json easily, so parsing is vital
        
        parsed_targets = parse_model_response(raw_output)
        # Even if parsed is None, the pure json score handles it
        
        # NOTE: the StrictJSONScore of the Evaluator expects raw JSON list, 
        # so if it returns a block of code, StrictJSONScore checks that.
        # Actually StrictJSONScore accepts blocks since we cleaned it above for parse_targets.
        # However, Evaluator itself parses raw_response directly.
        # So we must ensure litellm returns pure JSON without markup!
        
        predictions.append({
            "input": input_text,
            "raw_response": raw_output,
            "parsed_targets": parsed_targets
        })

    logger.info("Evaluation complete! Generating report...")
    report = evaluator.evaluate(predictions)
    
    p50_latency = float(np.percentile(latencies, 50)) if latencies else 0.0
    p99_latency = float(np.percentile(latencies, 99)) if latencies else 0.0
    
    report["latency_stats"] = {
        "p50_seconds": round(p50_latency, 3),
        "p99_seconds": round(p99_latency, 3)
    }
    
    stats = report["error_statistics"]
    n_entries = stats["total_entries"]
    n_expected = stats["total_expected_transactions"]

    def _pct(errors: int, denominator: int) -> str:
        if denominator == 0:
            return "N/A"
        return f"{errors / denominator:.1%}"

    print("\n=== EVALUATION REPORT ===")
    print(f"Model: {args.model}")
    print(f"Total Entries Evaluated: {report['total_entries']}")
    print(f"F1 Score:          {report['f1_score']:.4f}")
    print(f"Precision:         {report['precision']:.4f}")
    print(f"Recall:            {report['recall']:.4f}")
    print(f"Strict JSON Score: {report['strict_json_score']:.2%}")
    print(f"Entity Accuracy:   {report['entity_accuracy']:.2%}")
    print(f"Category Match:    {report['category_match']:.2%}")
    print(f"Latency (p50):     {report['latency_stats']['p50_seconds']:.3f}s")
    print(f"Latency (p99):     {report['latency_stats']['p99_seconds']:.3f}s")
    print()
    print("=== ERROR BREAKDOWN POR CAMPO ===")
    print(f"  (denominador: {n_expected} transacciones esperadas | {stats['total_matched_pairs']} pares matcheados)")
    print(f"  Hard Gate failures (JSON inválido): {stats['hard_gate_failures']:>4} / {n_entries} entries  ({_pct(stats['hard_gate_failures'], n_entries)})")
    print(f"  Amount errors:                      {stats['amount_errors']:>4} / {n_expected} pares    ({_pct(stats['amount_errors'], n_expected)})")
    print(f"  Currency errors:                    {stats['currency_errors']:>4} / {n_expected} pares    ({_pct(stats['currency_errors'], n_expected)})")
    print(f"  Type errors:                        {stats['type_errors']:>4} / {n_expected} pares    ({_pct(stats['type_errors'], n_expected)})")
    print(f"  Category errors:                    {stats['category_errors']:>4} / {n_expected} pares    ({_pct(stats['category_errors'], n_expected)})")
    print(f"  Date delta errors:                  {stats['date_delta_errors']:>4} / {n_expected} pares    ({_pct(stats['date_delta_errors'], n_expected)})")
    print(f"  Date expression errors:             {stats['date_expression_errors']:>4} / {n_expected} pares    ({_pct(stats['date_expression_errors'], n_expected)})")
    print("=================================\n")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = args.model.replace("/", "_")
    output_filename = f"results_{safe_model_name}_{timestamp}.json"
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Detailed results saved to {output_filename}")

if __name__ == "__main__":
    main()
