#!/bin/bash
# Launch script for DocSynthesis-V1 Gradio Demo
# IndiaAI IDP Challenge Submission

echo "========================================="
echo "🏆 DocSynthesis-V1 Interactive Demo"
echo "IndiaAI IDP Challenge 2024"
echo "========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install/update requirements
echo "📥 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements-gradio.txt

# Check if full requirements are needed
if [ "$1" == "--full" ]; then
    echo "📥 Installing full system requirements..."
    pip install -q -r requirements.txt
fi

echo ""
echo "========================================="
echo "🚀 Launching Gradio Interface..."
echo "========================================="
echo ""
echo "📱 The demo will be available at:"
echo "   - Local: http://localhost:7860"
echo "   - Public: (Gradio will generate a shareable link)"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Launch the Gradio app
python gradio_app.py

