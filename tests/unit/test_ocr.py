"""
Unit tests for OCR Engine
"""

import pytest
import numpy as np
from PIL import Image
from unittest.mock import Mock, patch
from src.ocr.engine import OCREngine
from src.config.settings import Settings


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = Settings()
    settings.model.device = "cpu"  # Use CPU for tests
    settings.model.enable_flash_attention = False
    return settings


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    # Create a simple image with text-like patterns
    img = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
    return Image.fromarray(img)


class TestOCREngine:
    """Test cases for OCR Engine."""
    
    def test_initialization(self, mock_settings):
        """Test OCR engine initialization."""
        with patch('src.ocr.engine.AutoModel'):
            with patch('src.ocr.engine.AutoTokenizer'):
                engine = OCREngine(mock_settings)
                assert engine is not None
                assert engine.model_name == mock_settings.model.deepseek_model
    
    def test_model_info(self, mock_settings):
        """Test getting model information."""
        with patch('src.ocr.engine.AutoModel'):
            with patch('src.ocr.engine.AutoTokenizer'):
                engine = OCREngine(mock_settings)
                info = engine.get_model_info()
                
                assert 'model_name' in info
                assert 'compression_ratio' in info
                assert info['compression_ratio'] == "10x"
    
    @patch('src.ocr.engine.AutoModel')
    @patch('src.ocr.engine.AutoTokenizer')
    def test_recognize_basic(self, mock_tokenizer, mock_model, mock_settings, sample_image):
        """Test basic OCR recognition."""
        # Mock the model's infer method
        mock_instance = Mock()
        mock_instance.infer.return_value = {
            'text': 'Sample text from document',
            'markdown': '# Sample text from document',
            'confidence': 0.95,
            'compression_ratio': 10.0
        }
        mock_model.from_pretrained.return_value = mock_instance
        
        engine = OCREngine(mock_settings)
        result = engine.recognize(sample_image)
        
        assert 'text' in result
        assert 'confidence' in result
        assert 'compression_ratio' in result
        assert result['confidence'] > 0
    
    def test_calculate_confidence(self, mock_settings):
        """Test confidence calculation."""
        with patch('src.ocr.engine.AutoModel'):
            with patch('src.ocr.engine.AutoTokenizer'):
                engine = OCREngine(mock_settings)
                
                # Test with optimal compression ratio
                result = {
                    'confidence': 0.95,
                    'compression_ratio': 10.0
                }
                confidence = engine._calculate_confidence(result)
                assert 0.9 <= confidence <= 1.0
                
                # Test with suboptimal compression ratio
                result = {
                    'confidence': 0.95,
                    'compression_ratio': 5.0
                }
                confidence = engine._calculate_confidence(result)
                assert 0.85 <= confidence <= 0.95


def test_ocr_integration():
    """Integration test - requires actual model (skip if not available)."""
    pytest.skip("Integration test - requires DeepSeek-OCR model")

