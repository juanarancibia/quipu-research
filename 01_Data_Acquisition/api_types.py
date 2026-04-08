"""
Type definitions for Dataset Curator API responses and data structures.
"""

from typing import TypedDict, List, Dict, Literal, Optional


# ============================================================================
# Transaction and Target Types
# ============================================================================

class Transaction(TypedDict):
    """A single financial transaction (expense or income)."""
    type: Literal["expense", "income"]
    amount: float
    currency: str
    category: str
    description: str
    date_delta_days: Optional[int]  # null if date is absolute/complex, 0 if no date mentioned, -1/1/etc if relative
    date_raw_expression: Optional[str]  # null if no date mentioned, "ayer" if relative, "el 4 de julio" if absolute


class Target(TypedDict):
    """A target transaction in the golden dataset."""
    type: Literal["EXPENSE", "INCOME"]
    amount: float
    currency: str
    category: str
    description: str
    date_delta_days: Optional[int]  # null if date is absolute/complex, 0 if no date mentioned, -1/1/etc if relative
    date_raw_expression: Optional[str]  # null if no date mentioned, "ayer" if relative, "el 4 de julio" if absolute


# ============================================================================
# Conversation Types
# ============================================================================

class ConversationOutput(TypedDict):
    """Output from bot response in tentative dataset."""
    type: str
    amount: float
    currency: str
    category: str
    description: str
    date_delta_days: Optional[int]  # null if date is absolute/complex, 0 if no date mentioned, -1/1/etc if relative
    date_raw_expression: Optional[str]  # null if no date mentioned, "ayer" if relative, "el 4 de julio" if absolute


class Conversation(TypedDict):
    """A conversation entry from tentative dataset."""
    input: str
    conversation_date: str  # Date when user sent the message (YYYY-MM-DD)
    outputs: List[ConversationOutput]
    saved: bool  # Added by API


# ============================================================================
# Golden Dataset Types
# ============================================================================

class GoldenDatasetMetadata(TypedDict):
    """Metadata for golden dataset entries."""
    saved_at: str
    source: str
    num_transactions: int


class GoldenDatasetEntry(TypedDict):
    """A complete entry in the golden dataset."""
    input: str
    conversation_date: str  # Date when user sent the message (YYYY-MM-DD)
    targets: List[Target]
    metadata: GoldenDatasetMetadata


# ============================================================================
# API Request Types
# ============================================================================

class SaveEntryRequest(TypedDict):
    """Request body for saving an entry to golden dataset."""
    input: str
    conversation_date: str  # Date when user sent the message (YYYY-MM-DD)
    targets: List[Transaction]


# ============================================================================
# API Response Types
# ============================================================================

class TypeDistribution(TypedDict):
    """Distribution counts by transaction type."""
    expense: int
    income: int


class DatasetStats(TypedDict):
    """Statistics for a dataset."""
    total: int
    by_type: Dict[str, int]


class StatsResponse(TypedDict):
    """Response from /api/stats endpoint."""
    tentative: DatasetStats
    golden: DatasetStats
    saved_count: int
    remaining: int


class ConversationsResponse(TypedDict):
    """Response from /api/conversations endpoint."""
    conversations: List[Conversation]
    total: int
    saved_count: int


class SaveResponse(TypedDict):
    """Response from /api/save endpoint."""
    success: bool
    message: str


class ErrorResponse(TypedDict):
    """Error response from API."""
    error: str


# ============================================================================
# Categories Types
# ============================================================================

class CategoryItem(TypedDict):
    """A single category with name and description."""
    name: str
    description: str


class Categories(TypedDict):
    """Available categories for expenses and incomes."""
    expense: List[CategoryItem]
    income: List[CategoryItem]


# ============================================================================
# Insights Types
# ============================================================================

class TypeCount(TypedDict):
    """Count of transactions by type."""
    EXPENSE: int
    INCOME: int


class TypePercentages(TypedDict):
    """Percentage distribution by type."""
    EXPENSE: float
    INCOME: float


class TypeDistributionDetail(TypedDict):
    """Detailed type distribution."""
    counts: TypeCount
    percentages: TypePercentages


class CategoryDistribution(TypedDict):
    """Distribution of transactions across categories."""
    EXPENSE: Dict[str, int]
    INCOME: Dict[str, int]


class AmountStatistics(TypedDict):
    """Statistical measures for transaction amounts."""
    min: float
    max: float
    avg: float
    median: float


class AmountStatisticsByType(TypedDict):
    """Amount statistics separated by type."""
    EXPENSE: AmountStatistics
    INCOME: AmountStatistics


class Recommendation(TypedDict):
    """A recommendation for improving dataset balance."""
    type: Literal["success", "warning", "info"]
    message: str


class InsightsResponse(TypedDict):
    """Response from /api/insights endpoint."""
    total_entries: int
    total_transactions: int
    type_distribution: TypeDistributionDetail
    category_distribution: CategoryDistribution
    transactions_per_entry: Dict[int, int]
    amount_statistics: AmountStatisticsByType
    currency_distribution: Dict[str, int]
    recommendations: List[Recommendation]


class EmptyInsightsResponse(TypedDict):
    """Response when no data is available."""
    total_entries: Literal[0]
    total_transactions: Literal[0]
    message: str


# ============================================================================
# Combined Response Types
# ============================================================================

# Union types for endpoints that can return different shapes
InsightsResponseUnion = InsightsResponse | EmptyInsightsResponse
SaveResponseUnion = SaveResponse | ErrorResponse
