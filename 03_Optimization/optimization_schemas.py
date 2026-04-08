from typing import Any, Optional, TypedDict


class OptimizationLog(TypedDict):
    module_name: str
    teleprompter: str
    started_at: str
    ended_at: str
    total_tokens: int
    total_cost_usd: float
    best_score: float
    best_program_path: str
    error_statistics: Optional[dict[str, Any]]  # FieldErrorStats from the Evaluator
