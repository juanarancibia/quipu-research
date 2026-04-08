#!/usr/bin/env python3
"""
Dataset Curator - Simple UI for reviewing and curating golden dataset
"""

from flask import Flask, render_template, request, jsonify
import json
import os
from datetime import datetime
from api_types import (
    Categories,
    ConversationsResponse,
    StatsResponse,
    SaveResponse,
    ErrorResponse,
    InsightsResponse,
    EmptyInsightsResponse,
    GoldenDatasetEntry,
    Target
)

app = Flask(__name__)

# Paths
TENTATIVE_DATASET_PATH = 'data/tentative_dataset.json'
GOLDEN_DATASET_PATH = '../golden_dataset.jsonl'
CATEGORIES_PATH = 'data/categories.json'

def load_categories() -> Categories:
    """Load categories from JSON file."""
    with open(CATEGORIES_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_tentative_dataset():
    """Load the tentative dataset."""
    with open(TENTATIVE_DATASET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['conversations']

def load_golden_dataset():
    """Load existing golden dataset entries."""
    golden_entries = []
    if os.path.exists(GOLDEN_DATASET_PATH):
        with open(GOLDEN_DATASET_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    golden_entries.append(json.loads(line))
    return golden_entries

def get_saved_inputs():
    """Get set of inputs that are already in golden dataset."""
    golden_entries = load_golden_dataset()
    return {entry['input'] for entry in golden_entries}

def save_to_golden_dataset(entry):
    """Append an entry to the golden dataset."""
    os.makedirs(os.path.dirname(GOLDEN_DATASET_PATH) if os.path.dirname(GOLDEN_DATASET_PATH) else '.', exist_ok=True)
    
    with open(GOLDEN_DATASET_PATH, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

@app.route('/')
def index():
    """Main curator interface."""
    return render_template('curator.html')

@app.route('/api/conversations')
def get_conversations() -> ConversationsResponse:
    """Get all conversations with saved status."""
    conversations = load_tentative_dataset()
    golden_entries = load_golden_dataset()
    
    # Create a map of saved entries by input
    saved_map = {entry['input']: entry for entry in golden_entries}
    
    # Mark which ones are already saved and replace with golden dataset data if available
    for conv in conversations:
        if conv['input'] in saved_map:
            # Replace with data from golden dataset to show the curated version
            golden_entry = saved_map[conv['input']]
            conv['saved'] = True
            conv['outputs'] = golden_entry.get('targets', [])
            conv['conversation_date'] = golden_entry.get('conversation_date', '')
        else:
            conv['saved'] = False
    
    response: ConversationsResponse = {
        'conversations': conversations,
        'total': len(conversations),
        'saved_count': len(saved_map)
    }
    return jsonify(response)

@app.route('/api/categories')
def get_categories() -> Categories:
    """Get available categories."""
    categories = load_categories()
    return jsonify(categories)

@app.route('/api/save', methods=['POST'])
def save_entry() -> SaveResponse | tuple[ErrorResponse, int]:
    """Save an entry to golden dataset."""
    data = request.json
    
    # Validate required fields
    if 'input' not in data or 'targets' not in data:
        error: ErrorResponse = {'error': 'Missing required fields'}
        return jsonify(error), 400
    
    if not isinstance(data['targets'], list):
        error: ErrorResponse = {'error': 'targets must be an array'}
        return jsonify(error), 400
    
    # Validate each target (only if targets array is not empty)
    for target in data['targets']:
        required_fields = ['type', 'amount', 'currency', 'category', 'description']
        if not all(field in target for field in required_fields):
            error: ErrorResponse = {'error': 'Each target must have type, amount, currency, category, and description'}
            return jsonify(error), 400
        
        # date_raw_expression: default to None if not provided or empty
        if 'date_raw_expression' not in target or target['date_raw_expression'] == '':
            target['date_raw_expression'] = None
        
        # date_delta_days logic:
        # - If explicitly set (including 0), keep it as is
        # - If null/empty AND date_raw_expression is null → default to 0
        # - If null/empty but date_raw_expression has value → keep as null
        if 'date_delta_days' not in target or target['date_delta_days'] == '' or target['date_delta_days'] is None:
            if target['date_raw_expression'] is None:
                target['date_delta_days'] = 0
            else:
                target['date_delta_days'] = None
    
    # Create entry in golden dataset format
    targets: list[Target] = [
        {
            'type': target['type'].upper(),  # type: ignore
            'amount': float(target['amount']),
            'currency': target['currency'],
            'category': target['category'],
            'description': target['description'],
            'date_delta_days': int(target['date_delta_days']) if target['date_delta_days'] is not None else None,
            'date_raw_expression': target['date_raw_expression']  # Can be None/null
        }
        for target in data['targets']
    ]
    
    entry: GoldenDatasetEntry = {
        'input': data['input'],
        'conversation_date': data['conversation_date'],
        'targets': targets,
        'metadata': {
            'saved_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'source': 'curator_ui',
            'num_transactions': len(data['targets'])
        }
    }
    
    # Check if already exists
    saved_inputs = get_saved_inputs()
    if entry['input'] in saved_inputs:
        error: ErrorResponse = {'error': 'Entry already exists in golden dataset'}
        return jsonify(error), 400
    
    # Save to file
    save_to_golden_dataset(entry)
    
    response: SaveResponse = {'success': True, 'message': 'Entry saved to golden dataset'}
    return jsonify(response)

@app.route('/api/stats')
def get_stats() -> StatsResponse:
    """Get statistics about the datasets."""
    conversations = load_tentative_dataset()
    golden_entries = load_golden_dataset()
    saved_inputs = get_saved_inputs()
    
    # Count by type
    tentative_by_type = {}
    for conv in conversations:
        for output in conv.get('outputs', [conv.get('output', {})]):  # Support both formats
            t = output.get('type', 'unknown')
            tentative_by_type[t] = tentative_by_type.get(t, 0) + 1
    
    golden_by_type = {}
    for entry in golden_entries:
        # Support both 'target' (old) and 'targets' (new) formats
        targets = entry.get('targets', [entry.get('target')] if entry.get('target') else [])
        for target in targets:
            if target:
                t = target.get('type', 'unknown')
                golden_by_type[t] = golden_by_type.get(t, 0) + 1
    
    response: StatsResponse = {
        'tentative': {
            'total': len(conversations),
            'by_type': tentative_by_type
        },
        'golden': {
            'total': len(golden_entries),
            'by_type': golden_by_type
        },
        'saved_count': len(saved_inputs),
        'remaining': len(conversations) - len(saved_inputs)
    }
    return jsonify(response)

@app.route('/api/insights')
def get_insights() -> InsightsResponse | EmptyInsightsResponse:
    """Get detailed insights about the golden dataset for balancing."""
    golden_entries = load_golden_dataset()
    categories = load_categories()
    
    if len(golden_entries) == 0:
        empty_response: EmptyInsightsResponse = {
            'total_entries': 0,
            'total_transactions': 0,
            'message': 'No entries in golden dataset yet'
        }
        return jsonify(empty_response)
    
    # Initialize counters
    total_transactions = 0
    type_count = {'EXPENSE': 0, 'INCOME': 0}
    category_distribution = {'EXPENSE': {}, 'INCOME': {}}
    transactions_per_entry = {}
    amount_ranges = {'EXPENSE': [], 'INCOME': []}
    currency_count = {}
    
    # Initialize all categories with 0
    for cat in categories['expense']:
        # Support both old format (string) and new format (object with name)
        cat_name = cat if isinstance(cat, str) else cat['name']
        category_distribution['EXPENSE'][cat_name] = 0
    for cat in categories['income']:
        cat_name = cat if isinstance(cat, str) else cat['name']
        category_distribution['INCOME'][cat_name] = 0
    
    # Analyze each entry
    for entry in golden_entries:
        targets = entry.get('targets', [entry.get('target')] if entry.get('target') else [])
        num_targets = len([t for t in targets if t])
        
        # Count transactions per entry
        transactions_per_entry[num_targets] = transactions_per_entry.get(num_targets, 0) + 1
        
        for target in targets:
            if not target:
                continue
                
            total_transactions += 1
            
            # Type distribution
            t_type = target.get('type', 'UNKNOWN').upper()
            type_count[t_type] = type_count.get(t_type, 0) + 1
            
            # Category distribution
            category = target.get('category', 'Unknown')
            if t_type in category_distribution:
                category_distribution[t_type][category] = category_distribution[t_type].get(category, 0) + 1
            
            # Amount ranges
            amount = target.get('amount', 0)
            if t_type in amount_ranges:
                amount_ranges[t_type].append(amount)
            
            # Currency distribution
            currency = target.get('currency', 'Unknown')
            currency_count[currency] = currency_count.get(currency, 0) + 1
    
    # Calculate statistics for amounts
    def get_amount_stats(amounts):
        if not amounts:
            return {}
        return {
            'min': min(amounts),
            'max': max(amounts),
            'avg': sum(amounts) / len(amounts),
            'median': sorted(amounts)[len(amounts) // 2]
        }
    
    # Remove categories with 0 count for cleaner display
    category_distribution['EXPENSE'] = {k: v for k, v in category_distribution['EXPENSE'].items() if v > 0}
    category_distribution['INCOME'] = {k: v for k, v in category_distribution['INCOME'].items() if v > 0}
    
    # Calculate balance metrics
    total = type_count.get('EXPENSE', 0) + type_count.get('INCOME', 0)
    expense_pct = (type_count.get('EXPENSE', 0) / total * 100) if total > 0 else 0
    income_pct = (type_count.get('INCOME', 0) / total * 100) if total > 0 else 0
    
    # Recommendations
    recommendations = []
    
    # Balance between expense/income
    if expense_pct > 70:
        recommendations.append({
            'type': 'warning',
            'message': f'Dataset heavily skewed towards expenses ({expense_pct:.1f}%). Consider adding more income examples.'
        })
    elif income_pct > 70:
        recommendations.append({
            'type': 'warning',
            'message': f'Dataset heavily skewed towards incomes ({income_pct:.1f}%). Consider adding more expense examples.'
        })
    else:
        recommendations.append({
            'type': 'success',
            'message': f'Good balance between expenses ({expense_pct:.1f}%) and incomes ({income_pct:.1f}%).'
        })
    
    # Category coverage
    expense_categories_used = len(category_distribution['EXPENSE'])
    income_categories_used = len(category_distribution['INCOME'])
    total_expense_categories = len(categories['expense'])
    total_income_categories = len(categories['income'])
    
    if expense_categories_used < total_expense_categories * 0.5:
        recommendations.append({
            'type': 'info',
            'message': f'Only {expense_categories_used}/{total_expense_categories} expense categories have examples. Consider diversifying.'
        })
    
    if income_categories_used < total_income_categories * 0.5:
        recommendations.append({
            'type': 'info',
            'message': f'Only {income_categories_used}/{total_income_categories} income categories have examples. Consider diversifying.'
        })
    
    # Multi-transaction coverage
    multi_transaction_entries = sum(count for num, count in transactions_per_entry.items() if num > 1)
    single_transaction_entries = transactions_per_entry.get(1, 0)
    
    if multi_transaction_entries < len(golden_entries) * 0.2:
        recommendations.append({
            'type': 'info',
            'message': f'Only {multi_transaction_entries} entries have multiple transactions. Consider adding more complex examples.'
        })
    
    response: InsightsResponse = {
        'total_entries': len(golden_entries),
        'total_transactions': total_transactions,
        'type_distribution': {
            'counts': type_count,
            'percentages': {
                'EXPENSE': round(expense_pct, 1),
                'INCOME': round(income_pct, 1)
            }
        },
        'category_distribution': category_distribution,
        'transactions_per_entry': dict(sorted(transactions_per_entry.items())),
        'amount_statistics': {
            'EXPENSE': get_amount_stats(amount_ranges['EXPENSE']),
            'INCOME': get_amount_stats(amount_ranges['INCOME'])
        },
        'currency_distribution': currency_count,
        'recommendations': recommendations
    }
    return jsonify(response)

@app.route('/insights')
def insights_page():
    """Insights dashboard page."""
    return render_template('insights.html')

if __name__ == '__main__':
    print("🚀 Starting Dataset Curator UI...")
    print(f"📂 Tentative dataset: {TENTATIVE_DATASET_PATH}")
    print(f"📂 Golden dataset: {GOLDEN_DATASET_PATH}")
    print("🌐 Open http://localhost:5000 in your browser")
    print()
    app.run(debug=True, host='0.0.0.0', port=5000)
