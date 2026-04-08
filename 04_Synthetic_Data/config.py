"""
Configuration for the Synthetic Data Generation module.

Supports auto-detection of available LLM providers based on environment
variables, with manual override via CLI arguments.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from this module's directory
load_dotenv(Path(__file__).parent / ".env")


# ============================================================================
# Provider Detection
# ============================================================================

def detect_available_model() -> str:
    """Auto-detect the best available LLM model based on environment variables.

    Priority order:
        1. OpenAI (OPENAI_API_KEY) → gpt-4o-mini
        2. OpenRouter (OPENROUTER_API_KEY) → openrouter/openai/gpt-4o-mini

    Returns:
        A LiteLLM-compatible model string.

    Raises:
        RuntimeError: If no API keys are found in the environment.
    """
    if os.getenv("OPENAI_API_KEY"):
        return "gpt-4o-mini"
    if os.getenv("OPENROUTER_API_KEY"):
        return "openrouter/openai/gpt-4o-mini"
    raise RuntimeError(
        "No LLM API key found. Set OPENAI_API_KEY or OPENROUTER_API_KEY in your .env file."
    )


# ============================================================================
# Generation Config
# ============================================================================

@dataclass
class GenerationConfig:
    """Configuration for synthetic data generation.

    Attributes:
        model: LiteLLM model string (auto-detected or manual override).
        golden_dataset_path: Path to the golden_dataset.jsonl file.
        output_path: Path where generated synthetic data will be saved.
        variants_per_entry: Number of input variants to generate per output.
        temperature: LLM temperature for generation (higher = more creative).
        batch_size: Number of entries to process per LLM call.
        source_tag: Metadata tag for generated entries.
    """

    model: str = ""
    golden_dataset_path: Path = Path(__file__).parent.parent / "golden_dataset.jsonl"
    output_path: Path = Path(__file__).parent / "generated" / "reverse_generated.jsonl"
    variants_per_entry: int = 3
    temperature: float = 0.8
    batch_size: int = 5
    source_tag: str = "synthetic_reverse"

    def __post_init__(self) -> None:
        """Auto-detect model if not manually specified."""
        if not self.model:
            self.model = detect_available_model()
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
