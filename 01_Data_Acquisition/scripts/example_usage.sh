#!/bin/bash
# Example script to generate tentative dataset from WhatsApp chat for manual review

# Example 1: Parse entire chat history
echo "Example 1: Parsing entire chat history..."
python3 parse_whatsapp_chat.py \
  -i files/WhatsApp\ Chat\ with\ Quipu.txt \
  -o tentative_dataset.json

echo ""
echo "---"
echo ""

# Example 2: Parse only recent conversations (from July 2025 onwards)
# This is useful to get the latest version of the bot's behavior
echo "Example 2: Parsing recent conversations (from July 2025)..."
python3 parse_whatsapp_chat.py \
  -i files/WhatsApp\ Chat\ with\ Quipu.txt \
  -o tentative_dataset_recent.json \
  --start-date 2025-07-01

echo ""
echo "---"
echo ""

# Example 3: Parse specific date range
echo "Example 3: Parsing specific date range (June 16-22, 2025)..."
python3 parse_whatsapp_chat.py \
  -i files/WhatsApp\ Chat\ with\ Quipu.txt \
  -o tentative_dataset_june.json \
  --start-date 2025-06-16 \
  --end-date 2025-06-22

echo ""
echo "Done! Review the generated JSON files and select entries for golden_dataset.jsonl"
