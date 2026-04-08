import dspy
import os
from dotenv import load_dotenv

load_dotenv()

# Test json mode with Minimax via LiteLLM kwargs
model_str = "openrouter/minimax/minimax-m2.5"
lm = dspy.LM(model_str, max_tokens=1500, response_format={"type": "json_object"})
dspy.settings.configure(lm=lm)

# Simple Predict
class SimpleExtract(dspy.Signature):
    """Extract information as a JSON array."""
    text: str = dspy.InputField()
    json_output: str = dspy.OutputField(desc="A JSON array")

predictor = dspy.Predict(SimpleExtract)
try:
    result = predictor(text="I bought an apple for 5 dollars")
    print("SUCCESS!")
    print(result.json_output)
except Exception as e:
    print("FAILED!")
    print(str(e))
