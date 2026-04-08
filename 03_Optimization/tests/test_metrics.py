import sys
import os
import dspy
import pytest

# Add current dir to path to import metrics
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from metrics import quipu_metric

def test_quipu_metric_perfect_score():
    """Test that a perfect prediction returns score 1.0"""
    example = dspy.Example(
        message="Ayer gasté 1500 en la farmacia",
        targets=[{
            "type": "EXPENSE",
            "amount": 1500.0,
            "currency": "ARS",
            "category": "Salud_Bienestar",
            "description": "la farmacia",
            "date_delta_days": -1,
            "date_raw_expression": "Ayer"
        }]
    ).with_inputs("message", "current_date")

    # The prediction simulates model output in JSON format
    pred_json = '''
    [
      {
        "type": "EXPENSE",
        "amount": 1500,
        "currency": "ARS",
        "category": "Salud_Bienestar",
        "description": "la farmacia",
        "date_delta_days": -1,
        "date_raw_expression": "Ayer"
      }
    ]
    '''
    pred = dspy.Prediction(financial_transactions_json=pred_json)

    score = quipu_metric(example, pred)
    assert score == 1.0

def test_quipu_metric_bad_prediction():
    """Test that an empty prediction on a positive example returns 0.0"""
    example = dspy.Example(
        message="Gasté 1500",
        targets=[{
            "type": "EXPENSE",
            "amount": 1500.0,
            "currency": "ARS",
            "category": "Regalos_Otros",
            "description": "gasto",
            "date_delta_days": None,
            "date_raw_expression": None
        }]
    ).with_inputs("message", "current_date")
    
    pred = dspy.Prediction(financial_transactions_json="[]")
    score = quipu_metric(example, pred)
    assert score == 0.0

