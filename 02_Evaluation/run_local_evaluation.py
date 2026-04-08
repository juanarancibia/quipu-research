import argparse
import json
import logging
import random
from datetime import datetime
import time
import numpy as np
import torch
import os
from typing import Optional
from unsloth import FastLanguageModel
from transformers import AutoTokenizer, pipeline

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

SIMPLE_TRANSACTION_PROMPT = """
Sos un extractor de información de finanzas personales. Analizá el mensaje y extraé las transacciones en formato JSON.

Devolvé un array de objetos con estos campos exactos:
- "description": descripción (string).
- "amount": número decimal positivo (float).
- "currency": "ARS" o la que se mencione (string).
- "category": debe ser exactamente una de las Categorías Válidas (string).
- "type": "EXPENSE" o "INCOME" (string).
- "date_delta_days": null para fechas fijas, 0 para hoy, o -N para días pasados (ej: ayer es -1) (integer o null).
- "date_raw_expression": null si no hay fecha, o la frase exacta usada (string o null).

Categorías Válidas: {category_list}
Monedas Válidas: {currency_list}

Respondé únicamente con JSON válido.
"""

HUMAN_PROMPT = """
Mensaje recibido: "{content}"

Analizá el mensaje según las instrucciones previas y devolvé solo el JSON correspondiente.
"""

def build_system_prompt() -> str:
    return SIMPLE_TRANSACTION_PROMPT.format(
        category_list=", ".join(VALID_CATEGORIES),
        currency_list=", ".join(VALID_CURRENCIES)
    )

def build_user_prompt(content: str, conversation_date: str) -> str:
    
    return HUMAN_PROMPT.format(
        content=content,
    )

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

def load_local_model(model_id: str):
    logger.info(f"Loading local model {model_id} using Unsloth FastLanguageModel...")
    
    # 1. FastLanguageModel detecta automáticamente si le pasas un modelo base o un adaptador LoRA.
    # Se encarga de ensamblarlo y pasarlo al hardware correcto de forma óptima.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=2048, # Ajústalo si necesitas más contexto
        dtype=None,        # Automáticamente usará bfloat16 si la GPU (como tu RTX 5090) lo soporta
        load_in_4bit=False,  # Mantener apagado como en tu script de entrenamiento
    )
    
    # 2. Habilitar modo inferencia (ESTA ES LA MAGIA)
    # Optimiza el modelo, reduce el uso de VRAM y activa soporte nativo 2x más rápido
    FastLanguageModel.for_inference(model)
    
    logger.info(f"Model successfully loaded on device: {model.device}")
    
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="Evaluate a local model against golden_dataset.jsonl")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model ID or path to local model directory")
    parser.add_argument("--limit", type=int, default=None, help="Number of random dataset entries to test (default: all)")
    parser.add_argument("--dataset", type=str, default="../golden_dataset.jsonl", help="Path to golden_dataset.jsonl")
    parser.add_argument("--no-prompt", action="store_true", help="Evaluate a fine-tuned model by passing only the user input directly, skipping the system prompt.")
    args = parser.parse_args()

    evaluator = Evaluator(args.dataset)
    with open(args.dataset, 'r', encoding='utf-8') as f:
        dataset_records = [json.loads(line) for line in f if line.strip()]

    if args.limit and args.limit < len(dataset_records):
        logger.info(f"Randomly sampling {args.limit} entries from the dataset of {len(dataset_records)}.")
        dataset_records = random.sample(dataset_records, args.limit)
    else:
        logger.info(f"Evaluating all {len(dataset_records)} entries.")

    system_prompt = build_system_prompt()
    predictions: list[PredictionEntry] = []
    latencies: list[float] = []

    mode_info = " (PROMPT-LESS / FINE-TUNED MODE)" if args.no_prompt else ""
    logger.info(f"Starting evaluation run with model: {args.model}{mode_info}")
    
    # Load Model
    model, tokenizer = load_local_model(args.model)

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
                {"role": "user", "content": user_prompt}
            ]
        
        # Format the prompt using the model's chat template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        start_time = time.time()
        try:
            # Generate response directly on device
            inputs = tokenizer(text=prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            input_length = inputs.input_ids.shape[1]
            raw_output = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
            
        except Exception as e:
            logger.error(f"Local inference failed for input '{input_text}': {e}")
            import traceback
            traceback.print_exc()
            raw_output = ""
        finally:
            latency = time.time() - start_time
            latencies.append(latency)
            
        parsed_targets = parse_model_response(raw_output)
        
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
    output_filename = f"results_{safe_model_name}_local_{timestamp}.json"
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Detailed results saved to {output_filename}")

if __name__ == "__main__":
    main()
