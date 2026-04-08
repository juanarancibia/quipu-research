import sys
import os
import json
import dspy
sys.path.append(os.path.abspath("../02_Evaluation"))
from dspy_modules.programs import TransactionExtractor
from run_optimization import load_dspy_dataset

dspy.settings.configure(lm=dspy.LM("openai/gpt-5-mini", max_tokens=16000, temperature=1.0, response_format={"type": "json_object"}))

dataset = load_dspy_dataset("../golden_dataset.jsonl")
program = TransactionExtractor(use_cot=True)
# program.load("compiled_programs/optimized_extractor_20260302_100222.json")

for ex in dataset[:10]:
    pred = program(message=ex.message)
    print(f"Input: {ex.message}")
    print(f"Expected Date Delta: {[t.get('date_delta_days') for t in ex.targets]}")
    try:
        parsed = json.loads(pred.financial_transactions_json)
        print(f"Predicted Date Delta: {[t.get('date_delta_days') for t in parsed]}")
    except:
        print("Failed to parse")
    print("-" * 30)
