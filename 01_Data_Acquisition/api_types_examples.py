"""
Example usage of API types for type checking and validation.
"""

from api_types import (
    Transaction,
    Target,
    GoldenDatasetEntry,
    SaveEntryRequest,
    InsightsResponse,
    Recommendation
)


# Example 1: Creating a transaction (from UI)
def create_transaction_example() -> Transaction:
    """Example of creating a transaction object."""
    transaction: Transaction = {
        "type": "expense",
        "amount": 1500.50,
        "currency": "ARS",
        "category": "Supermercado_Despensa",
        "description": "Compra en supermercado",
        "date_delta_days": 0,  # No date mentioned, assume same day
        "date_raw_expression": None  # No date mentioned in input
    }
    return transaction


# Example 2: Creating a target (for golden dataset)
def create_target_example() -> Target:
    """Example of creating a target object."""
    target: Target = {
        "type": "EXPENSE",
        "amount": 1500.50,
        "currency": "ARS",
        "category": "Supermercado_Despensa",
        "description": "Compra en supermercado",
        "date_delta_days": 0,  # No date mentioned, assume same day
        "date_raw_expression": None  # No date mentioned in input
    }
    return target


# Example 3: Creating a complete golden dataset entry
def create_golden_entry_example() -> GoldenDatasetEntry:
    """Example of creating a complete golden dataset entry."""
    entry: GoldenDatasetEntry = {
        "input": "Compré en el super por 1500 pesos",
        "conversation_date": "2026-02-19",
        "targets": [
            {
                "type": "EXPENSE",
                "amount": 1500.50,
                "currency": "ARS",
                "category": "Supermercado_Despensa",
                "description": "Compra en supermercado",
                "date_delta_days": 0,
                "date_raw_expression": None
            }
        ],
        "metadata": {
            "saved_at": "2026-02-19 10:30:00",
            "source": "curator_ui",
            "num_transactions": 1
        }
    }
    return entry


# Example 4: Creating a save request
def create_save_request_example() -> SaveEntryRequest:
    """Example of creating a save request."""
    request: SaveEntryRequest = {
        "input": "Pagué luz y gas ayer, 5000 y 3000 respectivamente",
        "conversation_date": "2026-02-19",
        "targets": [
            {
                "type": "expense",
                "amount": 5000.0,
                "currency": "ARS",
                "category": "Vivienda_Servicios",
                "description": "Luz",
                "date_delta_days": -1,  # Yesterday
                "date_raw_expression": "ayer"
            },
            {
                "type": "expense",
                "amount": 3000.0,
                "currency": "ARS",
                "category": "Vivienda_Servicios",
                "description": "Gas",
                "date_delta_days": -1,  # Yesterday
                "date_raw_expression": "ayer"
            }
        ]
    }
    return request


# Example 5: Processing insights response
def process_insights_example(response: InsightsResponse) -> None:
    """Example of processing an insights response."""
    print(f"Total entries: {response['total_entries']}")
    print(f"Total transactions: {response['total_transactions']}")
    
    # Access type distribution
    expenses_pct = response['type_distribution']['percentages']['EXPENSE']
    incomes_pct = response['type_distribution']['percentages']['INCOME']
    print(f"Expenses: {expenses_pct}%, Incomes: {incomes_pct}%")
    
    # Process recommendations
    for rec in response['recommendations']:
        icon = "✅" if rec['type'] == 'success' else "⚠️" if rec['type'] == 'warning' else "ℹ️"
        print(f"{icon} {rec['message']}")


# Example 6: Type-safe recommendation creation
def create_recommendation(
    rec_type: Recommendation['type'],
    message: str
) -> Recommendation:
    """Create a type-safe recommendation."""
    recommendation: Recommendation = {
        'type': rec_type,
        'message': message
    }
    return recommendation


if __name__ == "__main__":
    # Demo usage
    print("=== Example API Types Usage ===\n")
    
    print("1. Transaction:")
    tx = create_transaction_example()
    print(f"   {tx}\n")
    
    print("2. Target:")
    target = create_target_example()
    print(f"   {target}\n")
    
    print("3. Golden Entry:")
    entry = create_golden_entry_example()
    print(f"   Input: {entry['input']}")
    print(f"   Targets: {len(entry['targets'])} transaction(s)\n")
    
    print("4. Save Request:")
    req = create_save_request_example()
    print(f"   Input: {req['input']}")
    print(f"   Targets: {len(req['targets'])} transaction(s)\n")
    
    print("5. Recommendation:")
    rec = create_recommendation('success', 'Dataset is well balanced!')
    print(f"   {rec}\n")
