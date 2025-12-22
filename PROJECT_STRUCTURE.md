# DocSynthesis-V1 Project Structure

```
docsynthesis-v1/
│
├── README.md                    # Main documentation
├── LICENSE                      # Apache 2.0 License
├── CONTRIBUTING.md             # Contribution guidelines
├── GITHUB_SETUP.md             # GitHub repository setup guide
├── requirements.txt            # Python dependencies
├── requirements-gpu.txt        # GPU-specific dependencies
├── Dockerfile                  # Docker container definition
├── docker-compose.yml          # Multi-container setup
├── env.sample                  # Environment variables template
├── .gitignore                  # Git ignore patterns
│
├── main.py                     # Main entry point / CLI
│
├── src/                        # Source code
│   ├── __init__.py
│   │
│   ├── config/                 # Configuration management
│   │   ├── __init__.py
│   │   └── settings.py         # Settings and configuration
│   │
│   ├── preprocessing/          # Image preprocessing
│   │   ├── __init__.py
│   │   ├── pipeline.py         # Main preprocessing pipeline
│   │   ├── binarization.py     # Binarization methods
│   │   ├── geometric_correction.py  # Skew correction
│   │   └── image_restoration.py     # U-Net restoration
│   │
│   ├── layout/                 # Document layout analysis
│   │   ├── __init__.py
│   │   ├── analyzer.py         # HybriDLA layout analysis
│   │   └── hybridla.py         # Hybrid diffusion-autoregressive
│   │
│   ├── ocr/                    # OCR engines
│   │   ├── __init__.py
│   │   ├── engine.py           # DeepSeek-OCR engine
│   │   ├── deepseek_ocr.py     # DeepSeek implementation
│   │   └── internvl_ocr.py     # InternVL alternative
│   │
│   ├── translation/            # Multilingual translation
│   │   ├── __init__.py
│   │   ├── nmt.py              # Many-to-one NMT
│   │   └── language_detector.py # Language detection
│   │
│   ├── extraction/             # Information extraction
│   │   ├── __init__.py
│   │   ├── extractor.py        # Structured extraction
│   │   └── summarizer.py       # Document summarization
│   │
│   ├── xai/                    # Explainable AI
│   │   ├── __init__.py
│   │   ├── explainer.py        # Multi-level XAI
│   │   └── fam.py              # Feature Alignment Metrics
│   │
│   └── api/                    # REST API
│       ├── __init__.py
│       ├── server.py           # FastAPI server
│       ├── routes.py           # API routes
│       └── models.py           # Pydantic models
│
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── unit/                   # Unit tests
│   │   ├── __init__.py
│   │   ├── test_ocr.py
│   │   ├── test_preprocessing.py
│   │   ├── test_extraction.py
│   │   └── ...
│   │
│   └── integration/            # Integration tests
│       ├── __init__.py
│       └── test_pipeline.py
│
├── examples/                   # Example scripts
│   ├── basic_processing.py     # Basic usage example
│   ├── api_client.py           # API client example
│   ├── baseline_deepseek.py    # Baseline DeepSeek script
│   └── sample_certificate.pdf  # Sample document
│
├── docs/                       # Documentation
│   ├── submission.tex          # Technical submission document
│   ├── architecture.md         # Architecture documentation
│   ├── api.md                  # API documentation
│   ├── deployment.md           # Deployment guide
│   └── assets/                 # Images and diagrams
│
├── deployment/                 # Deployment configurations
│   ├── docker/                 # Docker configs
│   ├── kubernetes/             # K8s manifests
│   └── terraform/              # Infrastructure as code
│
└── data/                       # Data directory (not in git)
    ├── models/                 # Model weights
    ├── cache/                  # Cache files
    └── logs/                   # Log files
```

## Key Components

### Core Modules

1. **Preprocessing** (`src/preprocessing/`)
   - Image restoration and enhancement
   - Watermark suppression
   - Geometric correction
   - Binarization

2. **OCR Engine** (`src/ocr/`)
   - DeepSeek-OCR with COC
   - Context optical compression
   - Multilingual recognition

3. **Layout Analysis** (`src/layout/`)
   - HybriDLA framework
   - Document structure extraction
   - Reading order detection

4. **Translation** (`src/translation/`)
   - Many-to-one NMT
   - Indic language support
   - Language detection

5. **Extraction** (`src/extraction/`)
   - Structured data extraction
   - Grounding verification
   - Document summarization

6. **XAI** (`src/xai/`)
   - Visual attention maps
   - Token attribution
   - FAM scoring

7. **API** (`src/api/`)
   - RESTful API server
   - Async processing
   - Job management

### Supporting Files

- **Configuration**: Settings management and environment variables
- **Tests**: Comprehensive unit and integration tests
- **Examples**: Usage demonstrations and tutorials
- **Documentation**: Technical docs and guides
- **Deployment**: Docker, K8s, and cloud deployment configs

## Module Dependencies

```
main.py
  └─> src.config.settings
      ├─> src.preprocessing.pipeline
      ├─> src.layout.analyzer
      ├─> src.ocr.engine
      ├─> src.translation.nmt
      ├─> src.extraction.extractor
      ├─> src.extraction.summarizer
      └─> src.xai.explainer
```

## Data Flow

```
Input Document
    ↓
Preprocessing → Restored Image
    ↓
Layout Analysis → Document Structure
    ↓
OCR Engine → Extracted Text
    ↓
Translation → English Text (if needed)
    ↓
Extraction → Structured Fields
    ↓
Summarization → Document Summary
    ↓
XAI/FAM → Explanations
    ↓
Output Results
```

## Configuration Files

- `env.sample`: Environment variables template
- `requirements.txt`: Python dependencies
- `Dockerfile`: Container definition
- `docker-compose.yml`: Multi-service orchestration

## Best Practices

1. **Imports**: Use relative imports within packages
2. **Logging**: Use the logging module, not print()
3. **Configuration**: Use Settings class, not hardcoded values
4. **Error Handling**: Catch and log exceptions properly
5. **Testing**: Write tests for new functionality
6. **Documentation**: Update docstrings and README

## Adding New Components

1. Create module in appropriate directory
2. Add `__init__.py` if new package
3. Import in parent `__init__.py`
4. Add tests in `tests/unit/`
5. Update documentation
6. Add example usage

