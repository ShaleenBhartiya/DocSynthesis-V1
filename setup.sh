#!/bin/bash

# DocSynthesis-V1 Quick Setup Script
# Automates installation and setup process

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════╗"
echo "║   DocSynthesis-V1 Setup Script                     ║"
echo "║   IndiaAI IDP Challenge Submission                 ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "📋 Checking prerequisites..."
PYTHON_VERSION=$(python3 --version 2>&1 | grep -Po '(?<=Python )(.+)')
if [[ -z "$PYTHON_VERSION" ]]; then
    echo -e "${RED}❌ Python 3 is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION found"

# Check for CUDA (optional)
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓${NC} NVIDIA GPU detected"
    GPU_AVAILABLE=true
else
    echo -e "${YELLOW}⚠${NC} No NVIDIA GPU detected (CPU mode will be used)"
    GPU_AVAILABLE=false
fi

# Create virtual environment
echo ""
echo "🔨 Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓${NC} Virtual environment created"
else
    echo -e "${YELLOW}⚠${NC} Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate
echo -e "${GREEN}✓${NC} Virtual environment activated"

# Upgrade pip
echo ""
echo "📦 Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo -e "${GREEN}✓${NC} Pip upgraded"

# Install dependencies
echo ""
echo "📥 Installing dependencies..."
echo "   This may take several minutes..."

if [ "$GPU_AVAILABLE" = true ]; then
    echo "   Installing GPU-enabled version..."
    pip install -r requirements-gpu.txt > install.log 2>&1
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} GPU dependencies installed"
    else
        echo -e "${RED}❌ GPU installation failed, falling back to CPU version${NC}"
        pip install -r requirements.txt > install.log 2>&1
    fi
else
    echo "   Installing CPU version..."
    pip install -r requirements.txt > install.log 2>&1
fi

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Dependencies installed successfully"
else
    echo -e "${RED}❌ Installation failed. Check install.log for details${NC}"
    exit 1
fi

# Create necessary directories
echo ""
echo "📁 Creating directories..."
mkdir -p data/models data/cache data/logs temp output
echo -e "${GREEN}✓${NC} Directories created"

# Setup environment file
echo ""
echo "⚙️  Setting up environment..."
if [ ! -f ".env" ]; then
    cp env.sample .env
    echo -e "${GREEN}✓${NC} Environment file created (.env)"
    echo -e "${YELLOW}   Please edit .env with your configuration${NC}"
else
    echo -e "${YELLOW}⚠${NC} .env already exists"
fi

# Download models (optional)
echo ""
read -p "📥 Download DeepSeek-OCR model? (requires ~24GB) [y/N]: " download_model
if [[ $download_model =~ ^[Yy]$ ]]; then
    echo "Downloading models..."
    python3 -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained('deepseek-ai/DeepSeek-OCR', trust_remote_code=True); AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-OCR', trust_remote_code=True)"
    echo -e "${GREEN}✓${NC} Models downloaded"
else
    echo -e "${YELLOW}⚠${NC} Skipping model download"
    echo "   Models will be downloaded on first use"
fi

# Run tests
echo ""
read -p "🧪 Run tests to verify installation? [Y/n]: " run_tests
if [[ ! $run_tests =~ ^[Nn]$ ]]; then
    echo "Running tests..."
    pytest tests/unit/ -v
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} Tests passed"
    else
        echo -e "${YELLOW}⚠${NC} Some tests failed (may be expected if models not downloaded)"
    fi
fi

# Summary
echo ""
echo "╔════════════════════════════════════════════════════╗"
echo "║   Setup Complete! 🎉                               ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""
echo "📚 Next Steps:"
echo ""
echo "1. Activate virtual environment:"
echo "   ${GREEN}source venv/bin/activate${NC}"
echo ""
echo "2. Edit configuration (if needed):"
echo "   ${GREEN}nano .env${NC}"
echo ""
echo "3. Try basic example:"
echo "   ${GREEN}python examples/basic_processing.py${NC}"
echo ""
echo "4. Start API server:"
echo "   ${GREEN}python -m src.api.server${NC}"
echo ""
echo "5. Process a document:"
echo "   ${GREEN}python main.py --input document.pdf --output results/${NC}"
echo ""
echo "📖 Documentation:"
echo "   - README.md: Overview and usage"
echo "   - GITHUB_SETUP.md: GitHub repository setup"
echo "   - PROJECT_STRUCTURE.md: Codebase organization"
echo ""
echo "🐛 Issues? Check install.log or open an issue on GitHub"
echo ""
echo "Good luck with your IndiaAI submission! 🏆"

