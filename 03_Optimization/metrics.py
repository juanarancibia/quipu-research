import sys
import os
import json
import dspy

# Add 02_Evaluation to path so we can import the Evaluator
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "02_Evaluation"))
from evaluator import Evaluator
from schemas import PredictionEntry, PredictionTarget

# Global evaluator instance to prevent reloading the dataset
_evaluator_instance = None

def get_evaluator():
    global _evaluator_instance
    if _evaluator_instance is None:
        _evaluator_instance = Evaluator(os.path.join(os.path.dirname(__file__), "..", "golden_dataset.jsonl"))
    return _evaluator_instance

def parse_prediction(raw_json_str: str) -> list[PredictionTarget]:
    cleaned = raw_json_str.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return []
        
    items = parsed if isinstance(parsed, list) else [parsed]
    targets = []
    for item in items:
        if not isinstance(item, dict):
            continue
        
        target: PredictionTarget = {
            "type": item.get("type", "EXPENSE"),
            "amount": float(item.get("amount", 0) or 0.0),
            "currency": item.get("currency", "ARS"),
            "category": item.get("category", ""),
            "description": item.get("description", ""),
            "date_delta_days": item.get("date_delta_days"),
            "date_raw_expression": item.get("date_raw_expression")
        }
        targets.append(target)
    return targets

def quipu_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """
    Evaluates a DSPy prediction using the exact business rules:
    - Hard Gate: 0.0 if invalid json (now more lenient with markdown during optimization).
    - Score: 50% F1-Score + 50% Entity Accuracy
    """
    evaluator = get_evaluator()
    expected_targets = example.targets
    raw_response = getattr(pred, "financial_transactions_json", "")
    
    if not raw_response:
        return 0.0

    # 1. Clean and Parse first
    # This allows us to be lenient with markdown during optimization
    predicted_targets = parse_prediction(raw_response)
    
    # Re-serialize the cleaned version to check if it's at least valid JSON
    # but we will check the ORIGINAL for strict business rules if needed.
    # For optimization, we want to score the CONTENT.
    
    # We'll create a cleaned version for the strict_json_score to pass if the content is good
    try:
        cleaned_json = json.dumps(predicted_targets) if predicted_targets is not None else ""
    except Exception:
        cleaned_json = ""

    # 1. Hard Gate (Strict JSON)
    # We pass the cleaned_json to avoid failing on markdown, 
    # but the evaluator still checks for required fields.
    strict_score = evaluator.strict_json_score(cleaned_json)
    if strict_score == 0.0:
        return 0.0
        
    # 2. Build entry for evaluation
    pred_entry = {
        "input": example.message,
        "raw_response": raw_response, # We keep raw here for debugging/logs
        "parsed_targets": predicted_targets
    }
    
    # Temporarily force the expected targets for precise evaluation on this single input
    evaluator._golden[example.message] = expected_targets
    
    try:
        report = evaluator.evaluate([pred_entry])
    except Exception:
        # Resiliency: Return 0.0 without crashing on malformed data
        return 0.0
        
    # 3. Ponderaciones de Negocio (50% F1 / 50% Entity Accuracy)
    score = (report["f1_score"] * 0.50) + (report["entity_accuracy"] * 0.50)
    
    return float(score)
