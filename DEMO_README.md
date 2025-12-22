# DocSynthesis-V1 Interactive Demo

## 🎯 Overview

This is the **official interactive demonstration** of DocSynthesis-V1 for the **IndiaAI Intelligent Document Processing Challenge**. The demo showcases all seven stages of our state-of-the-art document processing pipeline through an intuitive web interface.

## 🌟 Features

### Complete Pipeline Demo
- **Upload & Process**: Full end-to-end document processing
- **Real-time Metrics**: Live performance monitoring
- **Interactive Visualization**: Explore results at each stage

### Individual Component Demos
1. **Preprocessing**: Image restoration, watermark removal, geometric correction
2. **Layout Analysis**: HybriDLA document structure detection
3. **OCR Engine**: DeepSeek-OCR with Context Optical Compression
4. **Explainability**: Visual attention maps and FAM analysis
5. **Benchmarks**: Comprehensive performance comparisons

## 🚀 Quick Start

### Option 1: Simple Launch (Recommended)

**Linux/Mac:**
```bash
chmod +x launch_demo.sh
./launch_demo.sh
```

**Windows:**
```cmd
launch_demo.bat
```

The demo will automatically:
- Create a virtual environment
- Install all dependencies
- Launch the Gradio interface
- Generate a shareable public link

### Option 2: Manual Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-gradio.txt

# Launch the demo
python gradio_app.py
```

### Option 3: Docker

```bash
# Build and run
docker-compose up gradio-demo

# Access at http://localhost:7860
```

## 📱 Accessing the Demo

Once launched, the demo is available at:
- **Local**: http://localhost:7860
- **Public**: Gradio generates a shareable link (valid for 72 hours)

## 🎮 Using the Demo

### Tab 1: Complete Pipeline
1. Upload a document (PDF, JPG, PNG)
2. Configure processing options:
   - Enable/disable preprocessing
   - Choose translation language
   - Toggle field extraction
   - Enable summarization
   - Show explainability analysis
3. Click "🚀 Process Document"
4. View results across all stages

### Tab 2: Preprocessing Demo
- Upload a degraded document
- See before/after comparison
- View enhancement statistics

### Tab 3: Layout Analysis
- Upload a document
- Visualize detected layout elements
- See hierarchical structure

### Tab 4: Explainability (XAI)
- Upload a document
- Generate attention heatmaps
- View FAM score breakdown

### Tab 5: Performance Benchmarks
- Compare against competitors
- View detailed metrics
- See competitive advantages

### Tab 6: About
- Complete technical overview
- Architecture details
- Challenge alignment

## 📄 Sample Documents

For testing, use any of these document types:
- Government certificates
- Educational certificates
- Official letters
- ID documents
- Forms and applications
- Multilingual documents (Hindi, Bengali, Tamil, etc.)

**Demo documents** are available in the `examples/` directory.

## ⚙️ Configuration

The demo runs with default settings optimized for demonstration. For production use:

1. Edit `.env` file (copy from `env.sample`)
2. Configure model paths and API keys
3. Adjust processing parameters in `config/settings.py`

## 🔧 Troubleshooting

### Port Already in Use
```bash
# Change port in gradio_app.py
demo.launch(server_port=7861)  # Use different port
```

### Dependencies Issues
```bash
# Reinstall clean
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-gradio.txt
```

### GPU Not Detected
```bash
# Install GPU version of dependencies
pip install -r requirements-gpu.txt
```

### Import Errors
The demo works in **standalone mode** without the full system. To enable full integration:
```bash
./launch_demo.sh --full
```

## 🎨 Customization

### Modify Colors
Edit the `INDIA_COLORS` dictionary in `gradio_app.py`:
```python
INDIA_COLORS = {
    "blue": "#003893",
    "orange": "#FF671F",
    "green": "#138808",
    # ... customize
}
```

### Add Custom Examples
Add your documents to the interface:
```python
gr.Examples(
    examples=[
        ["examples/certificate.jpg", True, True, "Hindi", True, True, True],
        ["examples/form.pdf", True, False, "English", True, False, False],
    ],
    inputs=[input_image, enable_preprocessing, ...]
)
```

### Modify Processing
Edit the `process_document_pipeline()` function to integrate with your models.

## 📊 Performance

The demo is optimized for:
- **Responsive UI**: <100ms interface interactions
- **Fast Processing**: Simulated results in ~2 seconds
- **Scalable**: Handles documents up to 10MB
- **Concurrent Users**: Supports multiple simultaneous users

For production deployment with real models, ensure:
- GPU availability (16GB+ VRAM recommended)
- 32GB+ RAM
- Fast storage (SSD)

## 🌐 Deployment

### Local Development
```bash
python gradio_app.py
```

### Production Server
```bash
# With authentication
python gradio_app.py --auth username:password

# Custom configuration
python gradio_app.py --server-name 0.0.0.0 --server-port 8080
```

### Cloud Deployment

**Hugging Face Spaces:**
```bash
# Push to HF Spaces
git push hf main
```

**AWS/GCP/Azure:**
Use the provided Docker configuration:
```bash
docker build -t docsynthesis-demo .
docker run -p 7860:7860 docsynthesis-demo
```

## 📈 Monitoring

The demo includes built-in metrics:
- Processing time per document
- Accuracy scores
- Confidence levels
- Cost estimates

Access monitoring dashboard at `/metrics` endpoint (when enabled).

## 🤝 Contributing

To contribute to the demo:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

See `CONTRIBUTING.md` for detailed guidelines.

## 📝 License

This demo is part of DocSynthesis-V1 and is licensed under Apache 2.0.

## 🏆 Competition Submission

This demo is the official submission for the **IndiaAI Intelligent Document Processing Challenge 2024**.

### Key Differentiators:
- ✅ Complete interactive demonstration
- ✅ All seven pipeline stages showcased
- ✅ Real-time performance metrics
- ✅ Comprehensive explainability
- ✅ Professional government-grade UI
- ✅ Ready for production deployment

## 📞 Support

For questions or issues:
- **Email**: team@docsynthesis.ai
- **GitHub Issues**: [Report a bug](https://github.com/brainwaveml/docsynthesis-v1/issues)
- **Documentation**: See full docs in `docs/` directory

## 🎯 Next Steps

After exploring the demo:

1. **Read the Technical Report**: `docs/submission.pdf`
2. **Try the API**: See `examples/api_client.py`
3. **Deploy Production**: Follow `docs/deployment.md`
4. **Integrate**: Use SDK in your application

---

<div align="center">

### 🏆 Built for IndiaAI Challenge 2024

**DocSynthesis-V1** by BrainWave ML

*Transforming India's Document Processing with AI*

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Gradio](https://img.shields.io/badge/interface-Gradio-orange.svg)](https://gradio.app)

</div>

