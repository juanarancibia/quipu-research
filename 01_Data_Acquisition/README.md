# 01 Data Acquisition: Golden Dataset Construction

This phase focuses on building a golden dataset by parsing WhatsApp chat conversations with the Quipu bot to extract input/output pairs for manual review and curation.

## Files

- `parse_whatsapp_chat.py` - Python script to parse WhatsApp chat exports into structured JSON
- `dataset_curator.py` - Flask web UI for reviewing and curating the golden dataset
- `start_curator.sh` - Script to start the curator UI
- `files/WhatsApp Chat with Quipu.txt` - WhatsApp chat export file
- `tentative_dataset.json` - Generated dataset for manual review (not tracked in git)
- `templates/curator.html` - HTML template for the curator UI

## Overview

The parser extracts **only** conversations where:
1. User sends a text message about an expense or income
2. Bot responds with structured data (expense/income parsing)

Other interactions (errors, confirmations, media, system messages) are filtered out.

## Usage

### Basic Usage

Parse the entire chat history:

```bash
python3 parse_whatsapp_chat.py -i files/WhatsApp\ Chat\ with\ Quipu.txt -o tentative_dataset.json
```

### With Date Filtering

Parse conversations within a specific date range:

```bash
python3 parse_whatsapp_chat.py \
  -i files/WhatsApp\ Chat\ with\ Quipu.txt \
  -o tentative_dataset.json \
  --start-date 2025-06-20 \
  --end-date 2025-06-30
```

Parse from a specific date onwards (get latest bot version responses):

```bash
python3 parse_whatsapp_chat.py \
  -i files/WhatsApp\ Chat\ with\ Quipu.txt \
  -o tentative_dataset.json \
  --start-date 2025-07-01
```

### Quick Start

You can also use the provided example script:

```bash
./example_usage.sh
```

### Command Line Options

- `-i, --input` - Path to WhatsApp chat export file (required)
- `-o, --output` - Output JSON file path (required)
- `--start-date` - Start date filter in YYYY-MM-DD format (inclusive, optional)
- `--end-date` - End date filter in YYYY-MM-DD format (inclusive, optional)
- `--indent` - JSON indentation level (default: 2)

## Output Format

The script generates a simplified JSON file with the following structure:

```json
{
  "metadata": {
    "source_file": "files/WhatsApp Chat with Quipu.txt",
    "total_conversations": 506,
    "start_date_filter": "2025-06-16",
    "end_date_filter": null,
    "export_timestamp": "2026-02-19 10:00:00",
    "note": "Datos tentativos para análisis manual. Filtrado: solo mensajes de texto con respuesta estructurada (expense/income)."
  },
  "conversations": [
    {
      "input": "Gaste 200 pesos en facturas",
      "outputs": [
        {
          "type": "expense",
          "description": "Facturas",
          "amount": 200.0,
          "currency": "ARS",
          "category": "comida",
          "bot_recorded_date": "16/06/2025 18:04",
          "status": "pending_confirmation"
        }
      ]
    },
    {
      "input": "Pague luz $14160, gas $7805 y agua $17448",
      "outputs": [
        {
          "type": "expense",
          "description": "Luz",
          "amount": 14160.0,
          "currency": "ARS",
          "category": "servicios"
        },
        {
          "type": "expense",
          "description": "Gas",
          "amount": 7805.0,
          "currency": "ARS",
          "category": "servicios"
        },
        {
          "type": "expense",
          "description": "Agua",
          "amount": 17448.0,
          "currency": "ARS",
          "category": "servicios"
        }
      ]
    }
  ]
}
```

**Note:** Each conversation can have 1 or more outputs (transactions). The parser automatically detects when the bot responds with multiple expense/income entries.

## Parsed Data Fields

### Output Structure (parsed_data)
Each conversation has an `outputs` array containing 1 or more transactions:

- `type`: "expense" or "income"
- `description`: Transaction description extracted by bot
- `amount`: Numeric amount (float)
- `currency`: Currency code (e.g., "ARS")
- `category`: Category assigned by bot
- `bot_recorded_date`: Date as recorded by bot (optional)
- `status`: "pending_confirmation" if not yet confirmed (optional)

**Multiple Transactions:** When a user reports multiple expenses/incomes in one message, all are captured in the `outputs` array.

## Workflow

### Step 1: Generate Tentative Dataset

Run the parser with appropriate date filters:

```bash
python3 parse_whatsapp_chat.py \
  -i files/WhatsApp\ Chat\ with\ Quipu.txt \
  -o tentative_dataset.json \
  --start-date 2025-06-16
```

### Step 2: Start the Curator UI

Launch the web interface for manual curation:

```bash
./start_curator.sh
```

Then open your browser at: **http://localhost:5000**

### Step 3: Review and Curate

Using the curator UI:

1. **Review** each conversation input/output pair
2. **Edit** any fields that need correction (description, category, amount, etc.)
3. **Save** quality examples to the golden dataset
4. **Filter** by status (saved/unsaved) or type (expense/income)
5. **Track** your progress with real-time statistics

The curator will:
- ✅ Show which entries are already saved (green border)
- ✅ Let you edit all fields before saving
- ✅ Prevent duplicate entries
- ✅ Save to `../golden_dataset.jsonl` automatically
- ✅ Update statistics in real-time

### Step 4: Result

Curated entries are saved to `golden_dataset.jsonl` in the project root with the format:

```json
{"input": "Gaste 200 en facturas", "target": {"type": "EXPENSE", "amount": 200.0, "currency": "ARS", "category": "comida", "description": "Facturas"}, "metadata": {"saved_at": "2026-01-27 16:30:00", "source": "curator_ui"}}
```

## Features

- ✅ Parses WhatsApp chat export format
- ✅ **Filters only text-based expense/income conversations**
- ✅ **Simplified output**: just user text + bot parsed data
- ✅ Date range filtering for dataset versioning
- ✅ Excludes media messages, confirmations, errors, and system messages
- ✅ Includes metadata for traceability

## Notes

- **Date filtering** is useful to capture only the latest version of the bot's behavior
- **Media messages are excluded** from the dataset (can't extract text from images)
- The output is meant for **manual review and curation**, not direct use as training data
- Filtered conversations focus on the core use case: natural language → structured transaction data
