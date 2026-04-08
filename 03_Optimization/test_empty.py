from metrics import quipu_metric, parse_prediction, get_evaluator
from dspy import Example, Prediction
import json

ex = Example(message="Dale dale", targets=[])
pred = Prediction(financial_transactions_json="[]")
raw = pred.financial_transactions_json
parsed = parse_prediction(raw)
print("Parsed:", parsed)

cleaned = json.dumps(parsed) if parsed is not None else ""
print("Cleaned:", cleaned)

evaluator = get_evaluator()
strict = evaluator.strict_json_score(cleaned)
print("Strict Score:", strict)

evaluator._golden[ex.message] = ex.targets
report = evaluator.evaluate([{"input": ex.message, "raw_response": raw, "parsed_targets": parsed}])
print("F1:", report["f1_score"])
print("EA:", report["entity_accuracy"])

score = quipu_metric(ex, pred)
print("Final Score:", score)
