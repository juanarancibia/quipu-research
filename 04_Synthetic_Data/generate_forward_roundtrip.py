import argparse
import asyncio
import json
import logging
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import litellm

# Add paths to allow importing from 01_Data_Acquisition
sys.path.append(str(Path(__file__).parent.parent / "01_Data_Acquisition"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Rule-Based Intent Generation ---

def load_categories() -> Dict[str, List[Dict[str, str]]]:
    """Loads valid categories from the central JSON file."""
    categories_path = Path(__file__).parent.parent / "01_Data_Acquisition" / "data" / "categories.json"
    try:
        with open(categories_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load categories from {categories_path}: {e}")
        # Fallback to a minimal set if file not found
        return {
            "expense": [{"name": "Supermercado_Despensa"}],
            "income": [{"name": "Salario_Honorarios"}]
        }

CATEGORIES = load_categories()
EXPENSE_CATEGORIES = CATEGORIES.get("expense", [])
INCOME_CATEGORIES = CATEGORIES.get("income", [])

VALID_CATEGORIES = list(set([cat["name"] for cat in EXPENSE_CATEGORIES + INCOME_CATEGORIES]))
VALID_CATEGORIES_STR = ", ".join(VALID_CATEGORIES)

# Predefined tuples for (date_delta_days, date_raw_expression) and their weights
DATE_TUPLES: List[Tuple[Optional[int], Optional[str]]] = [
    (0, None),                                     # Implicit date
    (0, "hoy"),                                    # Today explicit
    (-1, "ayer"),                                  # Yesterday
    (-2, "anteayer"),                              # Day before yesterday
    (-3, "hace 3 dias"),                           # 3 days ago
    (-5, "hace 5 dias"),                           # 5 days ago
    (-7, "hace una semana"),                       # Last week
    (-10, "hace 10 dias"),                         # 10 days ago
    (-14, "hace dos semanas"),                     # Two weeks ago
    (-30, "el mes pasado"),                        # Last month
    ("ABSOLUTE", None),                            # Absolute date (generated dynamically)
    ("ABSOLUTE_SHORT", None),                      # Absolute date short format (3/01)
    ("ABSOLUTE_LONG", None),                       # Absolute date long format (15/02/2025)
]
DATE_WEIGHTS = [0.29, 0.10, 0.15, 0.08, 0.05, 0.04, 0.08, 0.04, 0.05, 0.04, 0.03, 0.02, 0.03]

# Targeted mode: zero out implicit+explicit today, redistribute to non-today dates
# Order matches DATE_TUPLES: implicit(0), hoy(0), ayer(-1), anteayer(-2), 3d(-3), 5d(-5),
#   semana(-7), 10d(-10), 2sem(-14), mes(-30), abs, abs_short, abs_long
DATE_WEIGHTS_TARGETED = [0.0, 0.0, 0.20, 0.12, 0.10, 0.08, 0.15, 0.08, 0.10, 0.05, 0.05, 0.03, 0.04]

# Module-level flag set by CLI
_TARGETED = False

# Spanish month names for absolute date generation
_MONTH_NAMES = [
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"
]

def _generate_random_absolute_date(fmt: str = "natural") -> Tuple[Optional[int], str]:
    """Generates a random absolute date expression within the last 90 days."""
    days_ago = random.randint(1, 90)
    target_date = datetime.now() - timedelta(days=days_ago)
    
    if fmt == "short":
        expression = f"{target_date.day}/{target_date.month:02d}"
    elif fmt == "long":
        expression = f"{target_date.day:02d}/{target_date.month:02d}/{target_date.year}"
    else:
        month_name = _MONTH_NAMES[target_date.month - 1]
        expression = f"el {target_date.day} de {month_name}"
    
    return (None, expression)

def generate_single_intent(message_tone: str) -> Dict[str, Any]:
    """Generates a single random valid transaction intent based on a specific tone."""
    if _TARGETED:
        # Targeted: 90% EXPENSE to compensate income over-representation
        tx_type = random.choices(["EXPENSE", "INCOME"], weights=[0.90, 0.10], k=1)[0]
    else:
        # 11 expense categories vs 4 income → ~73% / ~27% to equalize per-category representation
        tx_type = random.choices(["EXPENSE", "INCOME"], weights=[0.73, 0.27], k=1)[0]
    
    # Use a weighted distribution for more realistic transaction amounts
    amount_ranges = [
        (50, 2000),          # Micro (café, kiosko, propina)
        (2000, 15000),       # Pequeño (delivery, bondi, sube)
        (15000, 80000),      # Medio-bajo (super chico, farmacia)
        (80000, 300000),     # Medio (super grande, ropa, servicios)
        (300000, 1000000),   # Alto (alquiler, sueldo promedio)
        (1000000, 5000000)   # Muy alto (sueldo alto, inversión)
    ]
    range_weights = [0.15, 0.25, 0.25, 0.20, 0.10, 0.05]
    selected_range = random.choices(amount_ranges, weights=range_weights, k=1)[0]
    
    amount = round(random.uniform(selected_range[0], selected_range[1]), 2)
    
    if message_tone == "SLANG_CHAOTIC":
        # Bias heavily towards whole numbers
        if random.random() < 0.8:
            amount = int(amount)
    elif message_tone == "CASUAL_EXACT":
        # Mix of whole numbers and exact decimals
        if random.random() < 0.5:
            amount = int(amount)
    else: # FORMAL_BANK_COPY
        # Always keep exact decimals
        pass

    # Ensure it's an integer if it cleanly ends in .00
    if amount == int(amount):
        amount = int(amount)

    currency = random.choices(["ARS", "USD"], weights=[0.8, 0.2], k=1)[0]
    
    if tx_type == "EXPENSE":
        if _TARGETED:
            # Boost underrepresented categories: Hogar_Mascotas and Deporte_Fitness
            expense_weights = []
            for cat in EXPENSE_CATEGORIES:
                if cat["name"] in ("Hogar_Mascotas", "Deporte_Fitness"):
                    expense_weights.append(5.0)  # 5x weight
                else:
                    expense_weights.append(1.0)
            cat_obj = random.choices(EXPENSE_CATEGORIES, weights=expense_weights, k=1)[0]
        else:
            cat_obj = random.choice(EXPENSE_CATEGORIES)
    else:
        cat_obj = random.choice(INCOME_CATEGORIES)
        
    category = cat_obj["name"]
    category_context = cat_obj.get("description", "")

    weights = DATE_WEIGHTS_TARGETED if _TARGETED else DATE_WEIGHTS
    date_delta, date_raw = random.choices(DATE_TUPLES, weights=weights, k=1)[0]
    
    # Handle absolute date generation dynamically
    if date_delta == "ABSOLUTE":
        date_delta, date_raw = _generate_random_absolute_date(fmt="natural")
    elif date_delta == "ABSOLUTE_SHORT":
        date_delta, date_raw = _generate_random_absolute_date(fmt="short")
    elif date_delta == "ABSOLUTE_LONG":
        date_delta, date_raw = _generate_random_absolute_date(fmt="long")

    return {
        "type": tx_type,
        "amount": amount,
        "currency": currency,
        "category": category,
        "description": f"Generated {tx_type.lower()} for {category}",
        "date_delta_days": date_delta,
        "date_raw_expression": date_raw,
        "_category_context": category_context,
        "_message_tone": message_tone
    }

def generate_random_intent() -> List[Dict[str, Any]]:
    """
    Generates a list of valid transaction intents.
    20% of the time it returns a multi-intent list (2 or 3 items).
    80% of the time it returns a single-intent list.
    """
    message_tone = random.choices(["SLANG_CHAOTIC", "CASUAL_EXACT", "FORMAL_BANK_COPY"], weights=[0.6, 0.2, 0.2], k=1)[0]
    
    is_multi = random.random() < 0.20
    if is_multi:
        num_intents = random.randint(2, 3)
        return [generate_single_intent(message_tone) for _ in range(num_intents)]
    else:
        return [generate_single_intent(message_tone)]


# --- Teacher Model Generation ---

TEACHER_SYSTEM_PROMPT = """
Sos un sistema de generación de mensajes sintéticos (finanzas personales).
Tu tarea es generar un UNICO texto que represente exactamente la(s) transacción(es) en el JSON proporcionado, siguiendo el TONO especificado.

REGLAS GLOBALES ESTRICTAS:
1. NUNCA menciones fechas calendario literales.
2. SI "date_raw_expression" es distinto de null, DEBES usar exactamente esa expresión. Si es null, NO menciones tiempo.
3. SI "currency" es USD, dejá claro que son dólares. Si es ARS, omitilo o usá moneda local acorde al tono.
4. USÁ el campo "_category_context" para inventar motivos MUY ESPECÍFICOS. NUNCA repitas el nombre literal de la categoría.
5. LOS NÚMEROS EXACTOS (amount) o decimales presentes en el JSON DEBEN ser reproducidos tal cual o de manera equivalente (ej: 4500000 puede ser 4.5 palos en SLANG, pero en FORMAL debe ser $4.500.000,00).
6. NUNCA uses palabras vagas o aproximativas antes de un monto (PROHIBIDO: "casi", "más o menos", "como", "unos", "aproximadamente", "alrededor de"). El número debe quedar claro y sin ambigüedad.

REGLA DE TONO (CRÍTICA):
El JSON incluirá un campo "_message_tone". Debes adaptar TODO tu estilo de redacción a esto:
- SLANG_CHAOTIC: Usá lunfardo argentino extremo (lucas, k, palo, mango, guita). Incluí errores ortográficos intencionales y abreviaciones (xq, dps, tmb, q, bue, jaja). Cero puntuación formal, sin mayúsculas al inicio.
- CASUAL_EXACT: Hablá normal, como un mensaje estándar de WhatsApp. Sin lunfardo extremo, pero conversacional. Usá buena puntuación.
- FORMAL_BANK_COPY: Texto robótico, altamente formal. Simulá una notificación de banco, factura o comprobante digital (ej: "Usted ha realizado un pago de $4,222,823.23 en concepto de..."). NO uses lenguaje conversacional ni saludos.

REGLA DE LONGITUD (CRÍTICA):
El usuario te pedirá una longitud específica ("SHORT", "MEDIUM", o "LONG").
- Si es SHORT: Mensaje telegráfico, 2 a 5 palabras máximo (ej: "3 lucas de pan", "1500 nafta"). Sin saludos ni relleno.
- Si es MEDIUM: Una sola oración corta, al grano (ej: "che recien pague 50k de la luz").
- Si es LONG: Modo rant/descargo, contá un poco la historia de por qué gastaste eso, quejate de los precios.
"""

async def generate_chaotic_message(intent_json: List[Dict[str, Any]], model: str) -> str:
    # 70% cortos, 20% medios, 10% largos
    length_constraint = random.choices(["SHORT", "MEDIUM", "LONG"], weights=[0.70, 0.20, 0.10], k=1)[0]
    
    intent_str = json.dumps(intent_json, ensure_ascii=False, indent=2)
    user_prompt = f"LONGITUD EXIGIDA: {length_constraint}\n\nJSON Intent a representar:\n{intent_str}"

    try:
        kwargs = {"temperature": 0.9} 
        if "gpt-5" in model.lower():
            kwargs["temperature"] = 1.0
            
        response = await litellm.acompletion(
            model=model,
            messages=[
                {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            **kwargs
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Teacher LLM generation failed: {e}")
        return ""

EXTRACTOR_SYSTEM_PROMPT = """
You are a personal finance information extractor specialized in Argentine (Rioplatense) Spanish. Given an informal message, possibly spanning multiple lines, with slang, varied number formats, and mixed currencies, analyze it and extract all financial transactions.

Normalize colloquial amounts: "mil" = 1 000, "lucas" = 1 000 ARS (e.g. "15 lucas" = 15 000), "gambas" = 100 ARS, "palos/palo" = 1 000 000 ARS, "mangos"/"pe"/"pesitos" indicate ARS currency. Assume ARS if no currency is specified. Be robust to commas/dots in numbers.

Infer type (EXPENSE vs INCOME) from verbs and context (e.g. "pagué", "me gasté" -> EXPENSE; "cobré", "me pagaron", "reintegro" -> INCOME). Apply category disambiguation: "facturas" are pastries (Supermercado_Despensa) unless clearly a utility bill (Vivienda_Servicios); "el chino"/"verduleria"/"almacen"/"despensa" -> Supermercado_Despensa; "pedidos ya"/"rappi"/"lomito"/"pizza" -> Comida_Comprada. 
CRITICAL: You MUST use one of these EXACT Valid Categories: [VALID_CATEGORIES]. DO NOT invent your own categories.
Valid Currencies: ARS, USD, EUR.

Handle dates: if no date is mentioned, set date_delta_days = 0 and date_raw_expression = null; for relative references like "hoy" set 0, "ayer" set -1, and include the exact phrase in date_raw_expression. For absolute calendar dates (e.g. "15/03", "el lunes 3 de julio"), set date_delta_days = null and date_raw_expression to the exact phrase—do not compute a delta.

If the message contains no financial content, return an empty JSON array.

Return a JSON array of one object per transaction with exactly these fields:
- description (string, cleaned brief description)
- amount (float, positive)
- currency (string, one of Valid Currencies)
- category (string, one of Valid Categories)
- type ("EXPENSE" or "INCOME")
- date_delta_days (integer or null as above)
- date_raw_expression (string or null)

OUTPUT ONLY A VALID JSON ARRAY. No explanations, no markdown blocks.

EXAMPLES:
Message: 81557,50 compra de Bondiolas para cumpleaños
Output:
[
  {
    "description": "compra de Bondiolas para cumpleaños",
    "amount": 81557.50,
    "currency": "ARS",
    "category": "Supermercado_Despensa",
    "type": "EXPENSE",
    "date_delta_days": 0,
    "date_raw_expression": null
  }
]

Message: ayer cobré 120 lucas del laburo
Output:
[
  {
    "description": "cobro del laburo",
    "amount": 120000.0,
    "currency": "ARS",
    "category": "Salario_Honorarios",
    "type": "INCOME",
    "date_delta_days": -1,
    "date_raw_expression": "ayer"
  }
]
"""

async def extract_intent(message: str, extractor_model: str) -> List[Dict[str, Any]]:
    """Runs the LLM extractor on the generated message."""
    try:
        prompt_content = EXTRACTOR_SYSTEM_PROMPT.replace("[VALID_CATEGORIES]", VALID_CATEGORIES_STR)
        
        response = await litellm.acompletion(
            model=extractor_model,
            messages=[
                {"role": "system", "content": prompt_content},
                {"role": "user", "content": f"Message: {message}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        raw_json_str = response.choices[0].message.content.strip()
        
        # Remove markdown array wrappings if LLM ignores instruction
        if raw_json_str.startswith("```json"):
            raw_json_str = raw_json_str[7:-3].strip()
        elif raw_json_str.startswith("```"):
            raw_json_str = raw_json_str[3:-3].strip()

        # Handle models wrapping array in {"transactions": [...]} because of json_object format
        parsed = json.loads(raw_json_str)
        if isinstance(parsed, dict):
            # Try to uncover the array
            for key, val in parsed.items():
                if isinstance(val, list):
                    return val
            return [parsed]
        
        return parsed
    except Exception as e:
        logger.warning(f"Extraction failed or JSON malformed: {e}")
        return []

# --- Validation Logic ---

def validate_round_trip(original: List[Dict[str, Any]], extracted: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """
    Compares the original intent with the extracted intent.
    Returns (is_valid, error_reason).
    """
    if len(original) != len(extracted):
        return False, f"Length mismatch: expected {len(original)}, got {len(extracted)}"

    # Validate that all extracted amounts are numeric before sorting
    for e in extracted:
        try:
            float(e.get("amount", 0))
        except (ValueError, TypeError):
            return False, f"Non-numeric amount in extraction: {e.get('amount')}"

    # Basic sort by amount to match them up if out of order
    sorted_orig = sorted(original, key=lambda x: float(x.get("amount", 0)))
    sorted_ext = sorted(extracted, key=lambda x: float(x.get("amount", 0)))

    for o, e in zip(sorted_orig, sorted_ext):
        if str(o["type"]).upper() != str(e.get("type", "")).upper():
            return False, f"Type mismatch: {o['type']} != {e.get('type')}"
        
        # Compare amounts with a relative tolerance (e.g. 5%) to allow for natural language rounding 
        # (e.g. 4.9 palos for 4928101.11 is extracted as 4900000)
        o_amount = float(o["amount"])
        e_amount = float(e.get("amount", 0))
        
        # If the expected amount is not 0, see if they are within 5% of each other.
        # If it is 0, they should be exactly equal (or very close).
        if o_amount > 0:
            diff_ratio = abs(o_amount - e_amount) / o_amount
            if diff_ratio > 0.05:
                # Still check absolute diff in case of very small numbers
                if abs(o_amount - e_amount) > 10.0:
                    return False, f"Amount mismatch: {o['amount']} != {e.get('amount')}"
        elif abs(o_amount - e_amount) > 0.1:
            return False, f"Amount mismatch: {o['amount']} != {e.get('amount')}"
            
        if str(o["currency"]).upper() != str(e.get("currency", "")).upper():
            return False, f"Currency mismatch: {o['currency']} != {e.get('currency')}"
            
        if str(o["category"]).upper() != str(e.get("category", "")).upper():
            return False, f"Category mismatch: {o['category']} != {e.get('category')}"
            
        # DSPy optimization might not perfectly capture the original delta or raw. 
        # Check delta only if they are both numbers, though it's safer to just loosely accept it if dates are close
        o_delta = o.get("date_delta_days")
        e_delta = e.get("date_delta_days")
        
        # Normalize null/None handling
        if o_delta == "None" or o_delta == "null": o_delta = None
        if e_delta == "None" or e_delta == "null": e_delta = None
        
        if o_delta != e_delta:
            # Check if it was extracted as 0 instead of None (or vice versa), which are often semantically similar
            if not ((o_delta in (None, 0)) and (e_delta in (None, 0))):
                return False, f"Date Delta mismatch: {o_delta} != {e_delta}"

    return True, ""

# --- Main Batch Flow ---

async def process_batch(batch_size: int, teacher_model: str, extractor_model: str) -> Tuple[List[Dict], List[Dict]]:
    """Processes a batch of iterations concurrently."""
    valid_entries = []
    invalid_entries = []

    async def _process_single():
        # 1. Generate Intent
        intent = generate_random_intent()
        
        # 2. Generate Chaotic Text
        message = await generate_chaotic_message(intent, teacher_model)
        if not message or len(message) > 500:
            # Strip internal fields before logging
            for item in intent:
                item.pop("_category_context", None)
                item.pop("_message_tone", None)
            error_msg = "Teacher LLM Generation Failed (e.g. API Error or Unsupported Parameter)" if not message else f"Message too long ({len(message)} chars > 500 max)"
            return (False, {
                "original_intent": intent,
                "generated_text": message or "",
                "extracted_intent": [],
                "error": error_msg
            })
            
        # Strip the temporary contextual fields before validation and saving
        for item in intent:
            item.pop("_category_context", None)
            item.pop("_message_tone", None)
            
        # 3. Extract Intent
        extracted = await extract_intent(message, extractor_model)
        
        # 4. Validate
        is_valid, error = validate_round_trip(intent, extracted)
        
        result_pkg = {
            "original_intent": intent,
            "generated_text": message,
            "extracted_intent": extracted,
            "error": error
        }

        if is_valid:
            sorted_orig = sorted(intent, key=lambda x: float(x.get("amount", 0)))
            sorted_ext = sorted(extracted, key=lambda x: float(x.get("amount", 0)))
            
            # Build final targets using EXTRACTED values (what the text actually said)
            # not original intent (which was just a guide for the teacher LLM).
            # NEVER fall back to original intent values — if the extractor didn't find it,
            # it means the text didn't contain it.
            final_targets = []
            for o, e in zip(sorted_orig, sorted_ext):
                final_targets.append({
                    "type": e.get("type", o["type"]),
                    "amount": e.get("amount", o["amount"]),
                    "currency": e.get("currency", o["currency"]),
                    "category": e.get("category", o["category"]),
                    "description": e.get("description") or o["description"],
                    "date_delta_days": e.get("date_delta_days"),
                    "date_raw_expression": e.get("date_raw_expression"),
                })

            # Format suitable for golden dataset appending
            final_valid_entry = {
                "input": message,
                "targets": final_targets,
                "conversation_date": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "metadata": {
                    "source": "synthetic_forward_roundtrip",
                    "num_transactions": len(intent)
                }
            }
            return (True, final_valid_entry)
            
        # If we fall through the if is_valid block, return the invalid result
        return (False, result_pkg)

    tasks = [_process_single() for _ in range(batch_size)]
    results = await asyncio.gather(*tasks)

    for res in results:
        if res is None:
            continue
        valid, data = res
        if valid:
            valid_entries.append(data)
        else:
            invalid_entries.append(data)

    return valid_entries, invalid_entries

async def main_async(args):
    """Main async orchestrator."""
    print(f"🤖 Teacher Model: {args.teacher_model}")
    print(f"🧠 Extractor Model: {args.extractor_model}")
    print(f"🔢 Total examples to generate attempts for: {args.limit}")
    print(f"📦 Batch size: {args.batch_size}")
    print()

    all_valid = []
    all_invalid = []
    total_attempts = 0

    base_dir = Path(__file__).parent
    valid_path = base_dir / "generated" / "forward_validated.jsonl"
    invalid_path = base_dir / "logs" / "forward_discarded.jsonl"
    
    valid_path.parent.mkdir(parents=True, exist_ok=True)
    invalid_path.parent.mkdir(parents=True, exist_ok=True)

    # Process in chunks to respect batch_size and manage rate limits
    while total_attempts < args.limit:
        current_batch_size = min(args.batch_size, args.limit - total_attempts)
        logger.info(f"Processing batch of {current_batch_size} (Total {total_attempts}/{args.limit})...")
        
        # Execute batch
        batch_valid, batch_invalid = await process_batch(current_batch_size, args.teacher_model, args.extractor_model)
        
        all_valid.extend(batch_valid)
        all_invalid.extend(batch_invalid)
        total_attempts += current_batch_size

        # Save to disk incrementally
        with open(valid_path, "a", encoding="utf-8") as f:
            for entry in batch_valid:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                
        with open(invalid_path, "a", encoding="utf-8") as f:
            for entry in batch_invalid:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Small delay to respect rate limits between batches
        if total_attempts < args.limit:
            await asyncio.sleep(args.delay)

    # Print Final Status
    print(f"\n📊 Generation Summary:")
    print(f"   Total Attempts: {total_attempts}")
    print(f"   Valid Matches (Yield): {len(all_valid)}")
    print(f"   Invalid Fails: {len(all_invalid)}")
    
    if total_attempts > 0:
        yield_rate = (len(all_valid) / total_attempts) * 100
        print(f"   Yield Rate: {yield_rate:.2f}%")
        
    print(f"\n💾 Valid entries appended to: {valid_path}")
    print(f"💡 Invalid entries logged to: {invalid_path}")
    
    # Workaround: Allow a small amount of time for underlying transport loops (like aiohttp/httpx)
    # to close their connections properly before the main asyncio loop shuts down.
    await asyncio.sleep(0.250)


def main():
    global _TARGETED
    parser = argparse.ArgumentParser(description="Generate synthetic data via Forward Round-Trip Evaluation.")
    parser.add_argument("--limit", type=int, default=10, help="Total number of attempts to generate.")
    parser.add_argument("--batch-size", type=int, default=5, help="Concurrent batch size.")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay in seconds between batches to avoid rate limits.")
    parser.add_argument("--teacher-model", type=str, default="gpt-4.1-mini", help="Teacher LLM to generate messy text (e.g. gpt-4.1-mini).")
    parser.add_argument("--extractor-model", type=str, default="gpt-4.1-nano", help="Predictor LLM to extract JSON intent (e.g. gpt-4.1-nano).")
    parser.add_argument("--targeted", action="store_true", help="Targeted mode: forces non-today dates, boosts underrepresented categories, reduces INCOME.")
    args = parser.parse_args()

    if args.targeted:
        _TARGETED = True
        print("🎯 TARGETED MODE: Generating data to fill distribution gaps")

    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()
