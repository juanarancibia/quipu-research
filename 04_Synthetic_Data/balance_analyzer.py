"""
Balance Analyzer — Dataset gap identification and generation planning.

Analyzes the golden_dataset.jsonl to find underrepresented categories,
types, and features, then produces a prioritized generation plan.
"""

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Target thresholds — what "balanced" looks like
# ============================================================================

BALANCE_TARGETS: Dict[str, int] = {
    # Expense categories
    "EXPENSE:Supermercado_Despensa": 55,
    "EXPENSE:Comida_Comprada": 40,
    "EXPENSE:Transporte": 30,
    "EXPENSE:Vivienda_Servicios": 25,
    "EXPENSE:Salud_Bienestar": 25,
    "EXPENSE:Ocio_Entretenimiento": 25,
    "EXPENSE:Hogar_Mascotas": 25,
    "EXPENSE:Deporte_Fitness": 25,
    "EXPENSE:Regalos_Otros": 25,
    "EXPENSE:Educacion_Capacitacion": 25,
    "EXPENSE:Financiero_Tarjetas": 25,
    # Income categories
    "INCOME:Inversiones_Finanzas": 25,
    "INCOME:Salario_Honorarios": 20,
    "INCOME:Ventas_Negocios": 20,
    "INCOME:Regalos_Otros": 15,
}

# Feature balance targets
FEATURE_TARGETS: Dict[str, int] = {
    "usd_transactions": 20,
    "non_transactional": 20,
    "relative_date_expressions": 40,
    "multi_transaction_entries": 35,
}


@dataclass
class CategoryGap:
    """Describes the gap for a specific type:category combo."""

    type: str
    category: str
    current_count: int
    target_count: int

    @property
    def gap(self) -> int:
        return max(0, self.target_count - self.current_count)

    @property
    def key(self) -> str:
        return f"{self.type}:{self.category}"


@dataclass
class FeatureGap:
    """Describes the gap for a dataset feature (e.g., USD, relative dates)."""

    feature_name: str
    current_count: int
    target_count: int

    @property
    def gap(self) -> int:
        return max(0, self.target_count - self.current_count)


@dataclass
class BalanceReport:
    """Full balance analysis of the dataset."""

    total_entries: int
    total_transactions: int
    category_gaps: List[CategoryGap]
    feature_gaps: List[FeatureGap]
    entries_by_category: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    @property
    def total_category_gap(self) -> int:
        return sum(g.gap for g in self.category_gaps)

    @property
    def total_feature_gap(self) -> int:
        return sum(g.gap for g in self.feature_gaps)

    def summary(self) -> str:
        """Return a human-readable summary of the balance analysis."""
        lines = [
            "📊 Balance Report",
            f"   Total entries: {self.total_entries}",
            f"   Total transactions: {self.total_transactions}",
            "",
            f"📉 Category Gaps (total needed: {self.total_category_gap} new transactions):",
        ]
        for g in sorted(self.category_gaps, key=lambda x: x.gap, reverse=True):
            if g.gap > 0:
                lines.append(f"   {g.key}: {g.current_count}/{g.target_count} → need {g.gap}")

        lines.append("")
        lines.append(f"📉 Feature Gaps (total needed: {self.total_feature_gap} new entries):")
        for g in sorted(self.feature_gaps, key=lambda x: x.gap, reverse=True):
            if g.gap > 0:
                lines.append(
                    f"   {g.feature_name}: {g.current_count}/{g.target_count} → need {g.gap}"
                )

        return "\n".join(lines)


def analyze_dataset(dataset_path: Path) -> BalanceReport:
    """Analyze the golden dataset for balance gaps.

    Args:
        dataset_path: Path to golden_dataset.jsonl.

    Returns:
        A BalanceReport with category and feature gap analysis.
    """
    entries: List[Dict[str, Any]] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    # Count categories
    category_counts: Counter = Counter()
    entries_by_category: Dict[str, List[Dict[str, Any]]] = {}

    usd_count = 0
    non_tx_count = 0
    relative_date_count = 0
    multi_tx_count = 0
    total_transactions = 0

    for entry in entries:
        targets = entry.get("targets", [])
        total_transactions += len(targets)

        if not targets:
            non_tx_count += 1
            continue

        if len(targets) > 1:
            multi_tx_count += 1

        for target in targets:
            key = f"{target['type']}:{target['category']}"
            category_counts[key] += 1

            if key not in entries_by_category:
                entries_by_category[key] = []
            entries_by_category[key].append(entry)

            if target.get("currency") == "USD":
                usd_count += 1

            delta = target.get("date_delta_days")
            raw_expr = target.get("date_raw_expression")
            if (delta is not None and delta != 0) or raw_expr is not None:
                relative_date_count += 1

    # Build category gaps
    category_gaps: List[CategoryGap] = []
    for key, target_count in BALANCE_TARGETS.items():
        type_str, cat = key.split(":", 1)
        current = category_counts.get(key, 0)
        category_gaps.append(CategoryGap(
            type=type_str,
            category=cat,
            current_count=current,
            target_count=target_count,
        ))

    # Build feature gaps
    feature_gaps: List[FeatureGap] = [
        FeatureGap("usd_transactions", usd_count, FEATURE_TARGETS["usd_transactions"]),
        FeatureGap("non_transactional", non_tx_count, FEATURE_TARGETS["non_transactional"]),
        FeatureGap(
            "relative_date_expressions",
            relative_date_count,
            FEATURE_TARGETS["relative_date_expressions"],
        ),
        FeatureGap(
            "multi_transaction_entries",
            multi_tx_count,
            FEATURE_TARGETS["multi_transaction_entries"],
        ),
    ]

    return BalanceReport(
        total_entries=len(entries),
        total_transactions=total_transactions,
        category_gaps=category_gaps,
        feature_gaps=feature_gaps,
        entries_by_category=entries_by_category,
    )


def select_entries_for_generation(
    report: BalanceReport,
    entries: List[Dict[str, Any]],
) -> List[Tuple[Dict[str, Any], int]]:
    """Select entries from the dataset that should be used as seeds for generation.

    Prioritizes entries from underrepresented categories. Returns a list of
    (entry, num_variants) tuples.

    Args:
        report: The balance analysis report.
        entries: All entries from the golden dataset.

    Returns:
        List of (entry, num_variants) tuples, sorted by priority (biggest gaps first).
    """
    generation_plan: List[Tuple[Dict[str, Any], int]] = []
    used_inputs: set = set()

    # Sort category gaps by size (biggest gaps first)
    sorted_gaps = sorted(report.category_gaps, key=lambda g: g.gap, reverse=True)

    for gap in sorted_gaps:
        if gap.gap <= 0:
            continue

        # Find seed entries for this category
        seed_entries = report.entries_by_category.get(gap.key, [])
        if not seed_entries:
            logger.warning(f"No seed entries for {gap.key} — cannot generate.")
            continue

        # Calculate how many variants we need per seed entry
        remaining_gap = gap.gap
        for seed in seed_entries:
            if remaining_gap <= 0:
                break
            if seed["input"] in used_inputs:
                continue

            variants_needed = min(5, remaining_gap)  # Cap at 5 variants per seed
            generation_plan.append((seed, variants_needed))
            used_inputs.add(seed["input"])
            remaining_gap -= variants_needed

    return generation_plan


if __name__ == "__main__":
    from config import GenerationConfig

    config = GenerationConfig()
    report = analyze_dataset(config.golden_dataset_path)
    print(report.summary())

    plan = select_entries_for_generation(report, [])
    print(f"\n🎯 Generation plan: {len(plan)} seed entries")
    for entry, n_variants in plan[:10]:
        targets = entry.get("targets", [])
        cat = targets[0]["category"] if targets else "non-tx"
        print(f"   {cat}: '{entry['input'][:60]}...' → {n_variants} variants")
