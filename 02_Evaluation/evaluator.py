"""
Evaluator for financial NLU model outputs.

Compares model predictions against the golden dataset using three metrics:
  - StrictJSONScore: validates that the model returns clean parseable JSON
  - EntityAccuracy: weighted scoring of financial, classification, and temporal fields
  - CategoryMatch: convenience metric for category accuracy alone

Plus transaction-level F1-Score, Precision, and Recall.
"""

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

from schemas import (
    EntryEvaluationResult,
    ErrorBreakdown,
    EvaluationReport,
    FieldErrorStats,
    PredictionEntry,
    PredictionTarget,
    TemporalScore,
    TransactionCounts,
)

# Fields required in every model prediction for StrictJSONScore to pass.
_REQUIRED_FIELDS = {"type", "amount", "currency", "category", "description"}

# EntityAccuracy weights. Must sum to 1.0.
_WEIGHTS = {
    # Finanzas (50%)
    "amount": 0.35,
    "currency": 0.15,
    # Clasificación (30%)
    "type": 0.15,
    "category": 0.15,
    # Contexto Temporal (20%): computed as a unit internally.
    "temporal": 0.20,
}

# Within the temporal unit: delta (70%) + expression (30%).
_TEMPORAL_WEIGHTS = {"delta": 0.70, "expression": 0.30}


class Evaluator:
    """
    Evaluates model predictions against the curated golden dataset.

    Usage:
        evaluator = Evaluator("../golden_dataset.jsonl")
        report = evaluator.evaluate(predictions)
    """

    def __init__(self, golden_dataset_path: str) -> None:
        """
        Load the golden dataset from a JSONL file into memory.

        Args:
            golden_dataset_path: Absolute or relative path to golden_dataset.jsonl.

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If a line is not valid JSON.
        """
        path = Path(golden_dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Golden dataset not found: {path}")

        self._golden: dict[str, list[dict]] = {}
        with path.open(encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {i}: {exc}") from exc
                input_text = entry.get("input", "")
                self._golden[input_text] = entry.get("targets", [])

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def evaluate(self, predictions: list[PredictionEntry]) -> EvaluationReport:
        """
        Evaluate a list of model predictions against the golden dataset.

        Args:
            predictions: List of PredictionEntry dicts, each with an 'input'
                that must match a golden dataset entry.

        Returns:
            EvaluationReport with aggregated scores and per-entry details.
        """
        per_entry_results: list[EntryEvaluationResult] = []
        total_tp = total_fp = total_fn = 0
        total_entity_accuracy = 0.0
        total_category_match = 0.0
        total_strict_json = 0.0

        # FieldErrorStats accumulators
        hard_gate_failures = 0
        total_expected_transactions = 0
        total_matched_pairs = 0
        amount_errors = currency_errors = type_errors = 0
        category_errors = date_delta_errors = date_expression_errors = 0

        for pred in predictions:
            result = self._evaluate_entry(pred)
            per_entry_results.append(result)

            total_strict_json += result["strict_json_score"]
            total_entity_accuracy += result["entity_accuracy"]
            total_category_match += result["category_match"]
            counts = result["transaction_counts"]
            total_tp += counts["true_positives"]
            total_fp += counts["false_positives"]
            total_fn += counts["false_negatives"]

            # Accumulate field-level error stats
            if result["strict_json_score"] == 0.0:
                hard_gate_failures += 1
            total_expected_transactions += counts["expected"]
            breakdowns = result["error_breakdown"]
            total_matched_pairs += len(breakdowns)
            for bd in breakdowns:
                if bd["amount_error"]:
                    amount_errors += 1
                if bd["currency_error"]:
                    currency_errors += 1
                if bd["type_error"]:
                    type_errors += 1
                if bd["category_error"]:
                    category_errors += 1
                if bd["date_delta_error"]:
                    date_delta_errors += 1
                if bd["date_expression_error"]:
                    date_expression_errors += 1
            # FN transactions count as errors on all fields
            fn_count = counts["false_negatives"]
            amount_errors += fn_count
            currency_errors += fn_count
            type_errors += fn_count
            category_errors += fn_count
            date_delta_errors += fn_count
            date_expression_errors += fn_count

        n = len(predictions)
        
        if total_tp == 0 and total_fp == 0 and total_fn == 0:
            # True Negative: The model correctly identified 0 transactions when 0 were expected
            precision = 1.0
            recall = 1.0
            f1 = 1.0
        else:
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        error_statistics = FieldErrorStats(
            hard_gate_failures=hard_gate_failures,
            total_entries=n,
            total_expected_transactions=total_expected_transactions,
            total_matched_pairs=total_matched_pairs,
            amount_errors=amount_errors,
            currency_errors=currency_errors,
            type_errors=type_errors,
            category_errors=category_errors,
            date_delta_errors=date_delta_errors,
            date_expression_errors=date_expression_errors,
        )

        return EvaluationReport(
            total_entries=n,
            strict_json_score=total_strict_json / n if n > 0 else 0.0,
            entity_accuracy=total_entity_accuracy / n if n > 0 else 0.0,
            category_match=total_category_match / n if n > 0 else 0.0,
            f1_score=round(f1, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            error_statistics=error_statistics,
            per_entry_results=per_entry_results,
        )

    # -------------------------------------------------------------------------
    # StrictJSONScore
    # -------------------------------------------------------------------------

    def strict_json_score(self, raw_response: str) -> float:
        """
        Return 1.0 if the raw response is clean JSON with required fields, 0.0 otherwise.

        Penalizes:
          - Markdown-wrapped JSON (```json ... ```)
          - Conversational text
          - JSON missing required fields (type, amount, currency, category, description)
        """
        stripped = raw_response.strip()

        # Reject any markdown code fences
        if re.search(r"```", stripped):
            return 0.0

        # Try parsing the response or the first JSON object/array found
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return 0.0

        # Support both a single dict and a list of dicts
        items = parsed if isinstance(parsed, list) else [parsed]
        
        # An empty list is a valid structural response for a non-transactional message
        # Let the F1 score handle whether an empty list was actually expected or not.
        if not items and isinstance(parsed, list):
            return 1.0

        for item in items:
            if not isinstance(item, dict):
                return 0.0
            if not _REQUIRED_FIELDS.issubset(item.keys()):
                return 0.0

        return 1.0

    # -------------------------------------------------------------------------
    # EntityAccuracy
    # -------------------------------------------------------------------------

    def entity_accuracy(
        self,
        predicted: list[PredictionTarget],
        expected: list[dict],
    ) -> float:
        """
        Compute the EntityAccuracy for one entry.

        Matches each predicted transaction to its closest expected counterpart
        greedily (by amount match first), then scores each matched pair using
        the weighted formula (Financial 50%, Classification 30%, Temporal 20%).

        Unmatched predictions and unmatched expected transactions both count as 0.
        """
        if not expected and not predicted:
            return 1.0  # Both correctly identified as non-transactional

        if not expected or not predicted:
            return 0.0

        matched_expected = set()
        scores = []

        for pred_t in predicted:
            best_score = 0.0
            best_idx = -1

            for i, exp_t in enumerate(expected):
                if i in matched_expected:
                    continue
                score = self._score_transaction_pair(pred_t, exp_t)
                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx >= 0:
                matched_expected.add(best_idx)
                scores.append(best_score)
            else:
                scores.append(0.0)

        # Unmatched expected transactions contribute 0 each
        n_total = max(len(predicted), len(expected))
        return sum(scores) / n_total

    def category_match(
        self,
        predicted: list[PredictionTarget],
        expected: list[dict],
    ) -> float:
        """
        Convenience metric: average exact (case-insensitive) category match across transactions.
        """
        if not expected and not predicted:
            return 1.0
        if not expected or not predicted:
            return 0.0

        matched_expected = set()
        matches = []

        for pred_t in predicted:
            pred_cat = (pred_t.get("category") or "").strip().lower()
            for i, exp_t in enumerate(expected):
                if i in matched_expected:
                    continue
                exp_cat = (exp_t.get("category") or "").strip().lower()
                if pred_cat == exp_cat:
                    matched_expected.add(i)
                    matches.append(1.0)
                    break
            else:
                matches.append(0.0)

        n_total = max(len(predicted), len(expected))
        return sum(matches) / n_total

    # -------------------------------------------------------------------------
    # F1 Score helpers
    # -------------------------------------------------------------------------

    def _transaction_counts(
        self,
        predicted: list[PredictionTarget],
        expected: list[dict],
    ) -> TransactionCounts:
        """
        Compute TP/FP/FN at transaction level.

        A True Positive requires amount AND type to match exactly.
        """
        matched_expected = set()
        tp = 0

        for pred_t in predicted:
            pred_amount = pred_t.get("amount")
            pred_type = (pred_t.get("type") or "").upper()

            for i, exp_t in enumerate(expected):
                if i in matched_expected:
                    continue
                exp_amount = exp_t.get("amount")
                exp_type = (exp_t.get("type") or "").upper()

                if pred_amount == exp_amount and pred_type == exp_type:
                    matched_expected.add(i)
                    tp += 1
                    break

        fp = len(predicted) - tp
        fn = len(expected) - tp

        return TransactionCounts(
            predicted=len(predicted),
            expected=len(expected),
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
        )

    # -------------------------------------------------------------------------
    # Internal scoring helpers
    # -------------------------------------------------------------------------

    def _evaluate_entry(self, pred: PredictionEntry) -> EntryEvaluationResult:
        """Evaluate a single prediction entry."""
        raw = pred.get("raw_response", "")
        strict = self.strict_json_score(raw)

        golden_targets = self._golden.get(pred["input"], [])
        predicted_targets: list[PredictionTarget] = pred.get("parsed_targets") or []

        entity_acc = self.entity_accuracy(predicted_targets, golden_targets)
        cat_match = self.category_match(predicted_targets, golden_targets)
        counts = self._transaction_counts(predicted_targets, golden_targets)
        error_breakdown = self._compute_field_errors(predicted_targets, golden_targets)

        return EntryEvaluationResult(
            input=pred["input"],
            raw_response=raw,
            parsed_targets=predicted_targets,
            strict_json_score=strict,
            entity_accuracy=round(entity_acc, 4),
            category_match=round(cat_match, 4),
            transaction_counts=counts,
            error_breakdown=error_breakdown,
        )

    def _compute_field_errors(
        self,
        predicted: list[PredictionTarget],
        expected: list[dict],
    ) -> list[ErrorBreakdown]:
        """
        Return per-field error flags for each greedy-matched (predicted, expected) pair.

        Uses the same greedy matching as entity_accuracy() (best score first).
        Unmatched expected transactions (FN) are not included here; the caller
        in evaluate() adds them as errors on all fields separately.
        """
        if not predicted or not expected:
            return []

        matched_expected: set[int] = set()
        breakdowns: list[ErrorBreakdown] = []

        for pred_t in predicted:
            best_score = -1.0
            best_idx = -1

            for i, exp_t in enumerate(expected):
                if i in matched_expected:
                    continue
                score = self._score_transaction_pair(pred_t, exp_t)
                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx < 0:
                continue  # FP — no expected counterpart

            matched_expected.add(best_idx)
            exp_t = expected[best_idx]

            temporal = self._score_temporal(
                pred_t.get("date_delta_days"),
                pred_t.get("date_raw_expression"),
                exp_t.get("date_delta_days"),
                exp_t.get("date_raw_expression"),
            )

            breakdowns.append(
                ErrorBreakdown(
                    amount_error=pred_t.get("amount") != exp_t.get("amount"),
                    currency_error=(
                        (pred_t.get("currency") or "").strip().upper()
                        != (exp_t.get("currency") or "").strip().upper()
                    ),
                    type_error=(
                        (pred_t.get("type") or "").strip().upper()
                        != (exp_t.get("type") or "").strip().upper()
                    ),
                    category_error=(
                        (pred_t.get("category") or "").strip().lower()
                        != (exp_t.get("category") or "").strip().lower()
                    ),
                    date_delta_error=temporal["date_delta_match"] < 1.0,
                    date_expression_error=temporal["date_expression_match"] < 1.0,
                )
            )

        return breakdowns

    def _score_transaction_pair(
        self, pred: PredictionTarget, expected: dict
    ) -> float:
        """
        Score a matched pair (predicted, expected) using the weighted formula.

        Weights:
          Financial (50%): amount (35%) + currency (15%)
          Classification (30%): type (15%) + category (15%)
          Temporal (20%): date_delta_days + date_raw_expression as a unit
        """
        amount_match = 1.0 if pred.get("amount") == expected.get("amount") else 0.0
        currency_match = (
            1.0
            if (pred.get("currency") or "").strip().upper()
            == (expected.get("currency") or "").strip().upper()
            else 0.0
        )
        type_match = (
            1.0
            if (pred.get("type") or "").strip().upper()
            == (expected.get("type") or "").strip().upper()
            else 0.0
        )
        category_match = (
            1.0
            if (pred.get("category") or "").strip().lower()
            == (expected.get("category") or "").strip().lower()
            else 0.0
        )

        temporal = self._score_temporal(
            pred.get("date_delta_days"),
            pred.get("date_raw_expression"),
            expected.get("date_delta_days"),
            expected.get("date_raw_expression"),
        )

        return (
            _WEIGHTS["amount"] * amount_match
            + _WEIGHTS["currency"] * currency_match
            + _WEIGHTS["type"] * type_match
            + _WEIGHTS["category"] * category_match
            + _WEIGHTS["temporal"] * temporal["temporal_score"]
        )

    def _score_temporal(
        self,
        pred_delta: Optional[int],
        pred_expr: Optional[str],
        exp_delta: Optional[int],
        exp_expr: Optional[str],
    ) -> TemporalScore:
        """
        Score temporal context as a unit: delta (70%) + expression (30%).

        Rules for date_delta_days:
          - Both None → 1.0 (absolute date, model correctly left it null)
          - Both equal ints → 1.0
          - Mismatch → 0.0

        Rules for date_raw_expression:
          - Both None → 1.0 (no date mentioned)
          - One None, one not → 0.0
          - Both set → normalized substring check:
              lowercase + strip, then check if either is contained in the other.
        """
        # Delta match
        if pred_delta is None and exp_delta is None:
            delta_score = 1.0
        elif pred_delta == exp_delta:
            delta_score = 1.0
        else:
            delta_score = 0.0

        # Expression match
        if pred_expr is None and exp_expr is None:
            expr_score = 1.0
        elif pred_expr is None or exp_expr is None:
            expr_score = 0.0
        else:
            pred_norm = pred_expr.strip().lower()
            exp_norm = exp_expr.strip().lower()
            if pred_norm == exp_norm or pred_norm in exp_norm or exp_norm in pred_norm:
                expr_score = 1.0
            else:
                # Partial fuzzy fallback via SequenceMatcher
                expr_score = SequenceMatcher(None, pred_norm, exp_norm).ratio()

        temporal_score = (
            _TEMPORAL_WEIGHTS["delta"] * delta_score
            + _TEMPORAL_WEIGHTS["expression"] * expr_score
        )

        return TemporalScore(
            date_delta_match=delta_score,
            date_expression_match=expr_score,
            temporal_score=round(temporal_score, 4),
        )
