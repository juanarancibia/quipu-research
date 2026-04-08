import json
import logging
from datetime import datetime
from typing import Any, Optional

from optimization_schemas import OptimizationLog

# Set up standard logger
logger = logging.getLogger("OptimizationTracker")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
if not logger.handlers:
    logger.addHandler(ch)

class OptimizationTracker:
    """
    Tracks tokens, cost (USD) and other metrics during DSPy optimization runs.
    Litellm's completion method provides `usage` which we can sum up.
    Since DSPy wraps Litellm, we can hook into Litellm's callback/observability or 
    keep track dynamically if possible. DSPy exposes history.
    """

    def __init__(self, model_name: str, teleprompter_name: str):
        self.model_name = model_name
        self.teleprompter_name = teleprompter_name
        self.start_time = datetime.now()
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0
        self.total_cost_usd = 0.0

    def add_usage(self, prompt_tokens: int, completion_tokens: int, cost: float):
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_cost_usd += cost

    def log_progress(self, message: str):
        logger.info(message)

    def finalize_and_save(
        self,
        best_score: float,
        output_path: str,
        best_program_path: str,
        error_statistics: Optional[dict[str, Any]] = None,
    ) -> OptimizationLog:
        end_time = datetime.now()

        log_entry: OptimizationLog = {
            "module_name": self.model_name,
            "teleprompter": self.teleprompter_name,
            "started_at": self.start_time.isoformat(),
            "ended_at": end_time.isoformat(),
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "total_cost_usd": self.total_cost_usd,
            "best_score": best_score,
            "best_program_path": best_program_path,
            "error_statistics": error_statistics,
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(log_entry, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Optimization finished. Log saved to {output_path}")
        logger.info(f"Total Tokens: {log_entry['total_tokens']} | Total Cost: ${log_entry['total_cost_usd']:.4f}")
        return log_entry
