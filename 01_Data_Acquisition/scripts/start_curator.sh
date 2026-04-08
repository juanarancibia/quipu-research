#!/bin/bash
# Start the Dataset Curator UI

cd "$(dirname "$0")"

echo "🔍 Checking dependencies..."
if ! python3 -c "import flask" 2>/dev/null; then
    echo "📦 Flask not found. Installing..."
    pip3 install flask
fi

echo ""
echo "🚀 Starting Dataset Curator UI..."
echo "📂 Working directory: $(pwd)"
echo ""

cd ..
python3 dataset_curator.py
