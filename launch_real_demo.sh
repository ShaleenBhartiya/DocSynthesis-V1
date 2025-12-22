#!/bin/bash
# Launch Real DocSynthesis-V1 Gradio Demo
# This version uses actual processing components

echo "========================================="
echo "🏆 DocSynthesis-V1 REAL Demo Launcher"
echo "IndiaAI IDP Challenge 2024"
echo "========================================="
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "🔄 Activating virtual environment..."
    source venv/bin/activate
fi

# Install minimal requirements
echo "📦 Installing dependencies..."
pip install -q gradio plotly opencv-python pillow numpy pytesseract

echo ""
echo "========================================="
echo "🚀 Launching REAL Demo..."
echo "========================================="
echo ""
echo "📱 The demo will be available at:"
echo "   - Local: http://localhost:7860"
echo "   - Public: (Gradio will generate a link)"
echo ""
echo "💡 Upload your own documents to test!"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Launch
python gradio_app_real.py

