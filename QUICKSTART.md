# 🚀 Quick Start Guide

Get DocSynthesis-V1 up and running in minutes!

## Prerequisites

- Python 3.9+
- 16GB+ RAM
- (Optional) NVIDIA GPU with CUDA 11.8+
- 50GB free disk space

## Option 1: Automated Setup (Recommended)

```bash
# Clone the repository (or use the folder)
cd docsynthesis-v1-github

# Run setup script
./setup.sh
```

The script will:
- ✅ Check prerequisites
- ✅ Create virtual environment
- ✅ Install dependencies
- ✅ Setup directories
- ✅ Configure environment
- ✅ (Optional) Download models
- ✅ Run tests

## Option 2: Manual Setup

### Step 1: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 2: Install Dependencies

**For GPU:**
```bash
pip install -r requirements-gpu.txt
```

**For CPU:**
```bash
pip install -r requirements.txt
```

### Step 3: Setup Environment

```bash
cp env.sample .env
# Edit .env with your settings
```

### Step 4: Create Directories

```bash
mkdir -p data/models data/cache data/logs temp output
```

## First Run: Process a Document

### Using CLI

```bash
# Basic usage
python main.py --input your_document.pdf --output results/

# With translation
python main.py --input document.pdf --translate --explain

# Full pipeline
python main.py \
  --input document.pdf \
  --output results/ \
  --language hindi \
  --translate \
  --extract-fields \
  --summarize \
  --explain
```

### Using Python API

```python
from main import DocSynthesisV1

# Initialize
processor = DocSynthesisV1()

# Process document
result = processor.process(
    input_path="document.pdf",
    output_dir="results/",
    translate=True,
    extract_fields=True,
    generate_summary=True,
    explain=True
)

# Access results
print(f"Confidence: {result['ocr']['confidence']:.2%}")
print(f"Extracted: {len(result['extraction']['fields'])} fields")
```

### Using REST API

**Start server:**
```bash
python -m src.api.server
```

**Make request:**
```bash
curl -X POST http://localhost:8000/api/v1/process \
  -F "file=@document.pdf" \
  -F 'options={"translate":true,"extract_fields":true}'
```

## Docker Deployment

```bash
# Build and run
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

## Verify Installation

```bash
# Run tests
pytest tests/unit/ -v

# Check API
python -c "from main import DocSynthesisV1; print('✅ Installation successful!')"
```

## Example Outputs

After processing, you'll find:

```
results/
├── text.txt              # Extracted plain text
├── document.md           # Markdown formatted document
├── extracted_fields.json # Structured data
├── summary.txt           # Document summary
└── results.json          # Complete results with metadata
```

## Common Issues

### Issue: CUDA out of memory
**Solution:** Reduce batch size in `.env`:
```bash
BATCH_SIZE=1
```

### Issue: Model download fails
**Solution:** Download manually:
```python
from transformers import AutoModel
AutoModel.from_pretrained("deepseek-ai/DeepSeek-OCR", trust_remote_code=True)
```

### Issue: Import errors
**Solution:** Ensure virtual environment is activated:
```bash
source venv/bin/activate
```

## Next Steps

1. **Read Documentation**
   - [README.md](README.md) - Complete documentation
   - [API Reference](docs/api.md) - API documentation
   - [Architecture](docs/architecture.md) - System design

2. **Try Examples**
   - `examples/basic_processing.py` - Basic usage
   - `examples/api_client.py` - API client
   - `examples/baseline_deepseek.py` - Baseline comparison

3. **Deploy**
   - [Docker Deployment](docs/deployment.md#docker)
   - [Kubernetes](docs/deployment.md#kubernetes)
   - [Cloud](docs/deployment.md#cloud)

4. **Contribute**
   - [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guide
   - Open issues and PRs on GitHub

## Getting Help

- **Documentation**: Check `docs/` folder
- **Issues**: [GitHub Issues](https://github.com/yourusername/docsynthesis-v1/issues)
- **Examples**: Look in `examples/` folder
- **Logs**: Check `data/logs/` for error details

## Performance Tips

1. **Use GPU** for 10-20× speedup
2. **Batch processing** for multiple documents
3. **Cache results** for repeated processing
4. **Adjust workers** based on available resources

## System Requirements by Scale

| Scale | Documents/Day | RAM | GPU | Storage |
|-------|---------------|-----|-----|---------|
| Development | <1,000 | 16GB | Optional | 50GB |
| Small | 1K-10K | 32GB | T4 | 100GB |
| Medium | 10K-50K | 64GB | A10 | 500GB |
| Large | 50K-200K | 128GB | A100 | 1TB |

---

**Ready to process documents! 🎉**

For detailed documentation, see [README.md](README.md)

