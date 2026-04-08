import json
import os
import sys
import dspy
from dotenv import load_dotenv

# Add paths
sys.path.append(os.path.abspath("../02_Evaluation"))
sys.path.append(os.path.abspath("."))

from dspy_modules.programs import TransactionExtractor
from metrics import quipu_metric, get_evaluator
from run_optimization import load_dspy_dataset

def main():
    load_dotenv()
    
    # Configure DSPy
    # Using Minimax as requested by user
    model_str = "openrouter/minimax/minimax-m2.5"
    lm = dspy.LM(model_str, max_tokens=1500)
    dspy.settings.configure(lm=lm)
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dspy_dataset("../golden_dataset.jsonl")
    
    # Take 5 examples to debug
    debug_set = dataset[:5]
    
    extractor = TransactionExtractor(use_cot=True)
    evaluator = get_evaluator()
    
    print(f"\n--- Debugging 5 examples with {model_str} (CoT=True) ---\n")
    
    for i, ex in enumerate(debug_set):
        print(f"Example {i+1}:")
        print(f"Input: {ex.message}")
        
        try:
            pred = extractor(message=ex.message)
            score = quipu_metric(ex, pred)
            
            print(f"Score: {score:.4f}")
            print(f"Thought: {getattr(pred, 'reasoning', 'N/A')}")
            print(f"Predicted JSON: {pred.financial_transactions_json}")
            print(f"Expected Targets: {json.dumps(ex.targets, indent=2)}")
            
            # Detailed breakdown if score < 1
            if score < 1.0:
                # We can call evaluate on a single entry to get the breakdown
                # quipu_metric already does something similar but doesn't print all details
                # Let's just print the parsed targets to see if they align
                from metrics import parse_prediction
                parsed = parse_prediction(pred.financial_transactions_json)
                print(f"Parsed Predictions: {json.dumps(parsed, indent=2)}")
                
        except Exception as e:
            print(f"Error: {e}")
        
        print("-" * 30)

if __name__ == "__main__":
    main()
