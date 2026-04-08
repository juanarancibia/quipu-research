"""
Unit tests for the Evaluator class.

Run with:
    source venv/bin/activate && pytest tests/test_evaluator.py -v
"""

import json
import sys
from pathlib import Path

import pytest

# Add parent directory to path so evaluator can be imported from 02_Evaluation/
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluator import Evaluator
from schemas import PredictionEntry, PredictionTarget


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture()
def golden_path(tmp_path: Path) -> Path:
    """Creates a minimal golden_dataset.jsonl for testing."""
    entries = [
        {
            "input": "Gaste 200 pesos en facturas",
            "conversation_date": "2025-06-16",
            "targets": [
                {
                    "type": "EXPENSE",
                    "amount": 200.0,
                    "currency": "ARS",
                    "category": "Vivienda_Servicios",
                    "description": "Facturas",
                    "date_delta_days": 0,
                    "date_raw_expression": None,
                }
            ],
            "metadata": {"saved_at": "2026-02-20", "source": "curator_ui", "num_transactions": 1},
        },
        {
            "input": "Ayer compré pan por 500",
            "conversation_date": "2025-06-17",
            "targets": [
                {
                    "type": "EXPENSE",
                    "amount": 500.0,
                    "currency": "ARS",
                    "category": "Supermercado_Despensa",
                    "description": "Pan",
                    "date_delta_days": -1,
                    "date_raw_expression": "Ayer",
                }
            ],
            "metadata": {"saved_at": "2026-02-20", "source": "curator_ui", "num_transactions": 1},
        },
        {
            "input": "El 4 de julio pagué obra social",
            "conversation_date": "2025-07-05",
            "targets": [
                {
                    "type": "EXPENSE",
                    "amount": 15000.0,
                    "currency": "ARS",
                    "category": "Salud_Bienestar",
                    "description": "Obra social",
                    "date_delta_days": None,
                    "date_raw_expression": "El 4 de julio",
                }
            ],
            "metadata": {"saved_at": "2026-02-20", "source": "curator_ui", "num_transactions": 1},
        },
        {
            "input": "Pague luz y gas",
            "conversation_date": "2025-06-18",
            "targets": [
                {
                    "type": "EXPENSE",
                    "amount": 14000.0,
                    "currency": "ARS",
                    "category": "Vivienda_Servicios",
                    "description": "Luz",
                    "date_delta_days": 0,
                    "date_raw_expression": None,
                },
                {
                    "type": "EXPENSE",
                    "amount": 7000.0,
                    "currency": "ARS",
                    "category": "Vivienda_Servicios",
                    "description": "Gas",
                    "date_delta_days": 0,
                    "date_raw_expression": None,
                },
            ],
            "metadata": {"saved_at": "2026-02-20", "source": "curator_ui", "num_transactions": 2},
        },
        {
            "input": "hoal querido como estas?",
            "conversation_date": "2025-07-01",
            "targets": [],
            "metadata": {"saved_at": "2026-02-21", "source": "curator_ui", "num_transactions": 0},
        },
    ]
    p = tmp_path / "golden_dataset.jsonl"
    p.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in entries), encoding="utf-8")
    return p


@pytest.fixture()
def evaluator(golden_path: Path) -> Evaluator:
    return Evaluator(str(golden_path))


# ============================================================================
# StrictJSONScore
# ============================================================================


def test_strict_json_score_valid_json(evaluator: Evaluator) -> None:
    """Valid single transaction JSON returns 1.0."""
    raw = json.dumps(
        {
            "type": "EXPENSE",
            "amount": 200.0,
            "currency": "ARS",
            "category": "Vivienda_Servicios",
            "description": "Facturas",
        }
    )
    assert evaluator.strict_json_score(raw) == 1.0


def test_strict_json_score_valid_array(evaluator: Evaluator) -> None:
    """Valid JSON array of transactions returns 1.0."""
    raw = json.dumps(
        [
            {
                "type": "EXPENSE",
                "amount": 200.0,
                "currency": "ARS",
                "category": "Vivienda_Servicios",
                "description": "Facturas",
            }
        ]
    )
    assert evaluator.strict_json_score(raw) == 1.0


def test_strict_json_score_markdown_wrapped(evaluator: Evaluator) -> None:
    """Markdown-fenced JSON returns 0.0."""
    raw = '```json\n{"type": "EXPENSE", "amount": 200.0, "currency": "ARS", "category": "Vivienda_Servicios", "description": "Facturas"}\n```'
    assert evaluator.strict_json_score(raw) == 0.0


def test_strict_json_score_conversational_text(evaluator: Evaluator) -> None:
    """Conversational text without JSON returns 0.0."""
    raw = "El modelo detectó un gasto de 200 pesos en facturas."
    assert evaluator.strict_json_score(raw) == 0.0


def test_strict_json_score_missing_required_fields(evaluator: Evaluator) -> None:
    """JSON missing required fields returns 0.0."""
    raw = json.dumps({"type": "EXPENSE", "amount": 200.0})
    assert evaluator.strict_json_score(raw) == 0.0


# ============================================================================
# EntityAccuracy
# ============================================================================


def test_entity_accuracy_exact_match(evaluator: Evaluator) -> None:
    """Perfect prediction returns 1.0."""
    predicted: list[PredictionTarget] = [
        {
            "type": "EXPENSE",
            "amount": 200.0,
            "currency": "ARS",
            "category": "Vivienda_Servicios",
            "description": "Facturas",
            "date_delta_days": 0,
            "date_raw_expression": None,
        }
    ]
    expected = [
        {
            "type": "EXPENSE",
            "amount": 200.0,
            "currency": "ARS",
            "category": "Vivienda_Servicios",
            "description": "Facturas",
            "date_delta_days": 0,
            "date_raw_expression": None,
        }
    ]
    score = evaluator.entity_accuracy(predicted, expected)
    assert score == 1.0


def test_entity_accuracy_amount_mismatch(evaluator: Evaluator) -> None:
    """Wrong amount should drop the score below 1.0."""
    predicted: list[PredictionTarget] = [
        {
            "type": "EXPENSE",
            "amount": 999.0,  # wrong
            "currency": "ARS",
            "category": "Vivienda_Servicios",
            "description": "Facturas",
            "date_delta_days": 0,
            "date_raw_expression": None,
        }
    ]
    expected = [
        {
            "type": "EXPENSE",
            "amount": 200.0,
            "currency": "ARS",
            "category": "Vivienda_Servicios",
            "description": "Facturas",
            "date_delta_days": 0,
            "date_raw_expression": None,
        }
    ]
    score = evaluator.entity_accuracy(predicted, expected)
    assert 0.0 < score < 1.0


# ============================================================================
# CategoryMatch
# ============================================================================


def test_category_match_exact(evaluator: Evaluator) -> None:
    """Exact category match returns 1.0."""
    predicted: list[PredictionTarget] = [
        {
            "type": "EXPENSE",
            "amount": 200.0,
            "currency": "ARS",
            "category": "Vivienda_Servicios",
            "description": "x",
            "date_delta_days": 0,
            "date_raw_expression": None,
        }
    ]
    expected = [{"category": "Vivienda_Servicios"}]
    assert evaluator.category_match(predicted, expected) == 1.0


def test_category_match_case_insensitive(evaluator: Evaluator) -> None:
    """Category match is case-insensitive."""
    predicted: list[PredictionTarget] = [
        {
            "type": "EXPENSE",
            "amount": 200.0,
            "currency": "ARS",
            "category": "vivienda_servicios",
            "description": "x",
            "date_delta_days": 0,
            "date_raw_expression": None,
        }
    ]
    expected = [{"category": "Vivienda_Servicios"}]
    assert evaluator.category_match(predicted, expected) == 1.0


def test_category_match_wrong(evaluator: Evaluator) -> None:
    """Wrong category returns 0.0."""
    predicted: list[PredictionTarget] = [
        {
            "type": "EXPENSE",
            "amount": 200.0,
            "currency": "ARS",
            "category": "Transporte",
            "description": "x",
            "date_delta_days": 0,
            "date_raw_expression": None,
        }
    ]
    expected = [{"category": "Vivienda_Servicios"}]
    assert evaluator.category_match(predicted, expected) == 0.0


# ============================================================================
# Temporal Scoring
# ============================================================================


def test_date_match_both_null(evaluator: Evaluator) -> None:
    """No date in input or prediction: temporal score is 1.0."""
    result = evaluator._score_temporal(0, None, 0, None)
    assert result["temporal_score"] == 1.0


def test_date_match_relative_exact(evaluator: Evaluator) -> None:
    """Exact relative date match: temporal score is 1.0."""
    result = evaluator._score_temporal(-1, "Ayer", -1, "Ayer")
    assert result["temporal_score"] == 1.0


def test_date_match_absolute_substring(evaluator: Evaluator) -> None:
    """'4 de julio' is substring of 'El 4 de julio': expression match is 1.0."""
    result = evaluator._score_temporal(None, "4 de julio", None, "El 4 de julio")
    assert result["date_expression_match"] == 1.0
    assert result["date_delta_match"] == 1.0  # both None


def test_date_match_delta_mismatch(evaluator: Evaluator) -> None:
    """delta=-1 predicted but expected delta=0: 0 on delta, full score on expression."""
    result = evaluator._score_temporal(-1, None, 0, None)
    assert result["date_delta_match"] == 0.0
    assert result["date_expression_match"] == 1.0  # both None → match
    assert result["temporal_score"] < 1.0


def test_date_match_one_null_one_set(evaluator: Evaluator) -> None:
    """Golden expected no date (raw=None) but model hallucinated one: expression score 0.0."""
    result = evaluator._score_temporal(0, "ayer", 0, None)
    assert result["date_expression_match"] == 0.0


# ============================================================================
# F1-Score / TP / FP / FN
# ============================================================================


def test_f1_more_predictions_than_expected(evaluator: Evaluator) -> None:
    """Model predicts 2 transactions but golden has 1: 1 TP, 1 FP, 0 FN."""
    predicted: list[PredictionTarget] = [
        {"type": "EXPENSE", "amount": 200.0, "currency": "ARS", "category": "x", "description": "x", "date_delta_days": 0, "date_raw_expression": None},
        {"type": "EXPENSE", "amount": 999.0, "currency": "ARS", "category": "x", "description": "x", "date_delta_days": 0, "date_raw_expression": None},
    ]
    expected = [{"type": "EXPENSE", "amount": 200.0, "currency": "ARS", "category": "x", "description": "x"}]
    counts = evaluator._transaction_counts(predicted, expected)
    assert counts["true_positives"] == 1
    assert counts["false_positives"] == 1
    assert counts["false_negatives"] == 0


def test_f1_fewer_predictions_than_expected(evaluator: Evaluator) -> None:
    """Model predicts 1 but golden has 2: 1 TP, 0 FP, 1 FN."""
    predicted: list[PredictionTarget] = [
        {"type": "EXPENSE", "amount": 14000.0, "currency": "ARS", "category": "x", "description": "x", "date_delta_days": 0, "date_raw_expression": None},
    ]
    expected = [
        {"type": "EXPENSE", "amount": 14000.0, "currency": "ARS", "category": "x", "description": "x"},
        {"type": "EXPENSE", "amount": 7000.0, "currency": "ARS", "category": "x", "description": "x"},
    ]
    counts = evaluator._transaction_counts(predicted, expected)
    assert counts["true_positives"] == 1
    assert counts["false_positives"] == 0
    assert counts["false_negatives"] == 1


def test_f1_non_transactional_correct(evaluator: Evaluator) -> None:
    """Golden is empty AND prediction is empty: both 0 counts (no TP/FP/FN)."""
    counts = evaluator._transaction_counts([], [])
    assert counts["true_positives"] == 0
    assert counts["false_positives"] == 0
    assert counts["false_negatives"] == 0


def test_f1_non_transactional_false_positive(evaluator: Evaluator) -> None:
    """Golden empty but model predicts a transaction: 0 TP, 1 FP, 0 FN."""
    predicted: list[PredictionTarget] = [
        {"type": "EXPENSE", "amount": 200.0, "currency": "ARS", "category": "x", "description": "x", "date_delta_days": 0, "date_raw_expression": None},
    ]
    counts = evaluator._transaction_counts(predicted, [])
    assert counts["true_positives"] == 0
    assert counts["false_positives"] == 1
    assert counts["false_negatives"] == 0


# ============================================================================
# Full Report Integration Test
# ============================================================================


def test_evaluate_full_report_perfect_predictions(evaluator: Evaluator) -> None:
    """
    Feeding exact copies of the golden targets as predictions should yield
    perfect scores across all metrics.
    """
    predictions: list[PredictionEntry] = [
        PredictionEntry(
            input="Gaste 200 pesos en facturas",
            raw_response=json.dumps(
                {"type": "EXPENSE", "amount": 200.0, "currency": "ARS", "category": "Vivienda_Servicios", "description": "Facturas"}
            ),
            parsed_targets=[
                PredictionTarget(
                    type="EXPENSE",
                    amount=200.0,
                    currency="ARS",
                    category="Vivienda_Servicios",
                    description="Facturas",
                    date_delta_days=0,
                    date_raw_expression=None,
                )
            ],
        ),
    ]
    report = evaluator.evaluate(predictions)
    assert report["strict_json_score"] == 1.0
    assert report["entity_accuracy"] == 1.0
    assert report["category_match"] == 1.0
    assert report["f1_score"] == 1.0
    assert report["precision"] == 1.0
    assert report["recall"] == 1.0


def test_evaluate_full_report_empty_predictions(evaluator: Evaluator) -> None:
    """
    Feeding empty parsed_targets for a transactional entry should yield
    F1=0 for that entry.
    """
    predictions: list[PredictionEntry] = [
        PredictionEntry(
            input="Gaste 200 pesos en facturas",
            raw_response="",
            parsed_targets=[],
        ),
    ]
    report = evaluator.evaluate(predictions)
    assert report["f1_score"] == 0.0
    assert report["entity_accuracy"] == 0.0


# ============================================================================
# ErrorBreakdown & FieldErrorStats
# ============================================================================


def _make_target(**overrides) -> PredictionTarget:
    """Helper to build a PredictionTarget with sensible defaults."""
    base: PredictionTarget = {
        "type": "EXPENSE",
        "amount": 200.0,
        "currency": "ARS",
        "category": "Vivienda_Servicios",
        "description": "Facturas",
        "date_delta_days": 0,
        "date_raw_expression": None,
    }
    base.update(overrides)  # type: ignore[typeddict-item]
    return base


def test_error_breakdown_perfect_match(evaluator: Evaluator) -> None:
    """All fields match: error_breakdown has one entry with all flags False."""
    predicted = [_make_target()]
    expected = [
        {
            "type": "EXPENSE",
            "amount": 200.0,
            "currency": "ARS",
            "category": "Vivienda_Servicios",
            "date_delta_days": 0,
            "date_raw_expression": None,
        }
    ]
    breakdowns = evaluator._compute_field_errors(predicted, expected)
    assert len(breakdowns) == 1
    bd = breakdowns[0]
    assert bd["amount_error"] is False
    assert bd["currency_error"] is False
    assert bd["type_error"] is False
    assert bd["category_error"] is False
    assert bd["date_delta_error"] is False
    assert bd["date_expression_error"] is False


def test_error_breakdown_category_wrong(evaluator: Evaluator) -> None:
    """Wrong category: only category_error is True."""
    predicted = [_make_target(category="Transporte")]
    expected = [
        {
            "type": "EXPENSE",
            "amount": 200.0,
            "currency": "ARS",
            "category": "Vivienda_Servicios",
            "date_delta_days": 0,
            "date_raw_expression": None,
        }
    ]
    breakdowns = evaluator._compute_field_errors(predicted, expected)
    assert len(breakdowns) == 1
    bd = breakdowns[0]
    assert bd["category_error"] is True
    assert bd["amount_error"] is False
    assert bd["type_error"] is False
    assert bd["date_delta_error"] is False


def test_error_breakdown_date_delta_wrong(evaluator: Evaluator) -> None:
    """Wrong date_delta_days: only date_delta_error is True."""
    predicted = [_make_target(date_delta_days=-5, date_raw_expression="Ayer")]
    expected = [
        {
            "type": "EXPENSE",
            "amount": 200.0,
            "currency": "ARS",
            "category": "Vivienda_Servicios",
            "date_delta_days": -1,
            "date_raw_expression": "Ayer",
        }
    ]
    breakdowns = evaluator._compute_field_errors(predicted, expected)
    assert len(breakdowns) == 1
    assert breakdowns[0]["date_delta_error"] is True
    assert breakdowns[0]["date_expression_error"] is False
    assert breakdowns[0]["amount_error"] is False
    assert breakdowns[0]["category_error"] is False


def test_error_breakdown_empty_inputs_returns_no_breakdowns(evaluator: Evaluator) -> None:
    """No predicted or no expected: returns empty list (no pairs to compare)."""
    assert evaluator._compute_field_errors([], []) == []
    assert evaluator._compute_field_errors([_make_target()], []) == []
    assert evaluator._compute_field_errors([], [{"amount": 200.0}]) == []


def test_error_statistics_fn_counts_as_all_field_errors(evaluator: Evaluator) -> None:
    """
    A FN (model predicted 0 transactions, 1 expected) must add 1 to every
    field error counter in the aggregated error_statistics.
    """
    predictions: list[PredictionEntry] = [
        PredictionEntry(
            input="Gaste 200 pesos en facturas",
            raw_response='{"type":"EXPENSE","amount":200,"currency":"ARS","category":"x","description":"x"}',
            parsed_targets=[],  # model produced nothing -> 1 FN
        ),
    ]
    report = evaluator.evaluate(predictions)
    st = report["error_statistics"]

    assert st["total_expected_transactions"] == 1
    assert st["total_matched_pairs"] == 0
    assert st["amount_errors"] == 1
    assert st["currency_errors"] == 1
    assert st["type_errors"] == 1
    assert st["category_errors"] == 1
    assert st["date_delta_errors"] == 1
    assert st["date_expression_errors"] == 1


def test_error_statistics_hard_gate_count(evaluator: Evaluator) -> None:
    """Entries with markdown-wrapped JSON increment hard_gate_failures."""
    markdown_response = (
        '```json\n[{"type":"EXPENSE","amount":200.0,"currency":"ARS",'
        '"category":"Vivienda_Servicios","description":"Facturas"}]\n```'
    )
    predictions: list[PredictionEntry] = [
        PredictionEntry(
            input="Gaste 200 pesos en facturas",
            raw_response=markdown_response,
            parsed_targets=[_make_target()],
        ),
    ]
    report = evaluator.evaluate(predictions)
    assert report["error_statistics"]["hard_gate_failures"] == 1


def test_error_statistics_aggregation(evaluator: Evaluator) -> None:
    """
    Two entries: one perfect, one with a wrong category.
    error_statistics should show exactly 1 category_error out of 2 expected transactions.
    """
    perfect_response = (
        '[{"type":"EXPENSE","amount":500.0,"currency":"ARS",'
        '"category":"Supermercado_Despensa","description":"Pan"}]'
    )
    category_wrong_response = (
        '[{"type":"EXPENSE","amount":200.0,"currency":"ARS",'
        '"category":"Transporte","description":"Facturas"}]'
    )
    predictions: list[PredictionEntry] = [
        PredictionEntry(
            input="Ayer compré pan por 500",
            raw_response=perfect_response,
            parsed_targets=[
                _make_target(
                    amount=500.0,
                    category="Supermercado_Despensa",
                    date_delta_days=-1,
                    date_raw_expression="Ayer",
                )
            ],
        ),
        PredictionEntry(
            input="Gaste 200 pesos en facturas",
            raw_response=category_wrong_response,
            parsed_targets=[_make_target(category="Transporte")],
        ),
    ]
    report = evaluator.evaluate(predictions)
    st = report["error_statistics"]

    assert st["total_expected_transactions"] == 2
    assert st["total_matched_pairs"] == 2
    assert st["category_errors"] == 1
    assert st["amount_errors"] == 0
    assert st["type_errors"] == 0
    assert st["hard_gate_failures"] == 0
