"""
Type definitions for the Evaluation Pipeline.

These types define the shape of predictions fed into the Evaluator,
and the structured reports it produces.
"""

from typing import Optional, TypedDict


# ============================================================================
# Input Types
# ============================================================================


class PredictionTarget(TypedDict):
    """A single transaction predicted by a model for evaluation."""

    type: str  # "EXPENSE" or "INCOME"
    amount: float
    currency: str
    category: str
    description: str
    date_delta_days: Optional[int]  # None for absolute dates, 0 if no date, -N/+N if relative
    date_raw_expression: Optional[str]  # None if no date mentioned


class PredictionEntry(TypedDict):
    """A model's full response for a single golden dataset entry."""

    input: str  # Must match exactly a golden dataset input for lookup
    raw_response: str  # The raw string the model returned
    parsed_targets: Optional[list[PredictionTarget]]  # None if JSON parsing failed entirely


# ============================================================================
# Per-transaction Result Types
# ============================================================================


class TransactionCounts(TypedDict):
    """TP/FP/FN counts for a single evaluated entry."""

    predicted: int
    expected: int
    true_positives: int
    false_positives: int
    false_negatives: int


class TemporalScore(TypedDict):
    """Breakdown of the temporal component score for a matched transaction pair."""

    date_delta_match: float   # 1.0 if exact int match (or both None)
    date_expression_match: float  # 1.0 if normalized substring match (or both None)
    temporal_score: float     # Weighted: delta (70%) + expression (30%)


class ErrorBreakdown(TypedDict):
    """Per-field error flags for a single matched transaction pair."""

    amount_error: bool
    currency_error: bool
    type_error: bool
    category_error: bool
    date_delta_error: bool
    date_expression_error: bool


class EntryEvaluationResult(TypedDict):
    """Full evaluation result for a single golden dataset entry."""

    input: str
    raw_response: str
    parsed_targets: Optional[list[PredictionTarget]]
    strict_json_score: float          # 1.0 or 0.0
    entity_accuracy: float            # Weighted score across matched pairs
    category_match: float             # Standalone convenience metric
    transaction_counts: TransactionCounts
    error_breakdown: list[ErrorBreakdown]  # Per-field errors for each matched pair


# ============================================================================
# Aggregate Report Types
# ============================================================================


class FieldErrorStats(TypedDict):
    """
    Aggregated field-level error counts across all evaluated entries.

    Denominator: total_expected_transactions (all expected across the dataset).
    FN transactions (model missed them entirely) count as an error on ALL fields.
    FP transactions (model hallucinated) are excluded from field-level counts
    since there is no expected counterpart to compare against.
    """

    hard_gate_failures: int       # Entries where strict_json_score == 0.0
    total_entries: int            # Total entries evaluated
    total_expected_transactions: int  # Sum of expected transactions across entries
    total_matched_pairs: int      # Pairs greedy-matched for field comparison
    amount_errors: int
    currency_errors: int
    type_errors: int
    category_errors: int
    date_delta_errors: int
    date_expression_errors: int


class EvaluationReport(TypedDict):
    """Aggregated evaluation results across all entries."""

    total_entries: int
    strict_json_score: float          # % of valid JSON responses
    entity_accuracy: float            # Avg across all entries
    category_match: float             # Avg across all entries
    f1_score: float                   # Micro-averaged transaction-level F1
    precision: float                  # Micro-averaged transaction-level Precision
    recall: float                     # Micro-averaged transaction-level Recall
    error_statistics: FieldErrorStats  # Aggregated field-level error breakdown
    per_entry_results: list[EntryEvaluationResult]
