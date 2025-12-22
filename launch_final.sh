#!/bin/bash

echo "=========================================="
echo "🚀 Launching DocSynthesis-V1 Final Demo"
echo "=========================================="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

echo "✅ Python found: $(python3 --version)"
echo ""

# Install dependencies if needed
echo "📦 Checking dependencies..."
pip3 install -q gradio torch transformers pillow numpy opencv-python-headless 2>/dev/null

echo ""
echo "🔥 Starting Gradio server..."
echo "📍 Access the demo at: http://localhost:7860"
echo "🌐 Public URL will be displayed below"
echo ""
echo "=========================================="
echo ""

python3 gradio_app_final.py

