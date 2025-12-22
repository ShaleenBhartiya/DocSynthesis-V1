"""Configuration settings for DocSynthesis-V1."""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Model configuration settings."""
    deepseek_model: str = "deepseek-ai/DeepSeek-OCR"
    internvl_model: str = "OpenGVLab/InternVL2-8B"
    base_size: int = 1024
    image_size: int = 640
    crop_mode: bool = True
    batch_size: int = 4
    device: str = "cuda"
    enable_flash_attention: bool = True


@dataclass
class PreprocessingConfig:
    """Preprocessing configuration."""
    enable_restoration: bool = True
    enable_watermark_removal: bool = True
    enable_geometric_correction: bool = True
    enable_binarization: bool = True
    target_dpi: int = 300
    max_image_size: int = 4096


@dataclass
class LayoutConfig:
    """Layout analysis configuration."""
    model_name: str = "hybridla"
    confidence_threshold: float = 0.7
    enable_table_detection: bool = True
    enable_hierarchy_extraction: bool = True


@dataclass
class TranslationConfig:
    """Translation configuration."""
    model_type: str = "many-to-one"
    supported_languages: list = field(default_factory=lambda: [
        "hindi", "bengali", "tamil", "telugu", "gujarati", 
        "marathi", "malayalam", "kannada", "punjabi", "odia"
    ])
    target_language: str = "english"
    beam_size: int = 5
    max_length: int = 512


@dataclass
class ExtractionConfig:
    """Extraction configuration."""
    llm_model: str = "gpt-3.5-turbo"
    enable_grounding: bool = True
    confidence_threshold: float = 0.8
    max_fields: int = 50


@dataclass
class XAIConfig:
    """Explainability configuration."""
    enable_visual_attention: bool = True
    enable_token_attribution: bool = True
    enable_nl_explanation: bool = True
    compute_fam_score: bool = True


@dataclass
class APIConfig:
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    max_upload_size: int = 50 * 1024 * 1024  # 50MB
    enable_cors: bool = True
    api_key_required: bool = False


class Settings:
    """Global settings manager for DocSynthesis-V1."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize settings from environment variables and config file.
        
        Args:
            config_path: Optional path to YAML configuration file
        """
        # Load from environment variables
        self.cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "0")
        self.data_dir = Path(os.getenv("DATA_DIR", "data"))
        self.cache_dir = Path(os.getenv("CACHE_DIR", "data/cache"))
        self.log_dir = Path(os.getenv("LOG_DIR", "data/logs"))
        
        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize component configs
        self.model = ModelConfig()
        self.preprocessing = PreprocessingConfig()
        self.layout = LayoutConfig()
        self.translation = TranslationConfig()
        self.extraction = ExtractionConfig()
        self.xai = XAIConfig()
        self.api = APIConfig()
        
        # Load from config file if provided
        if config_path and Path(config_path).exists():
            self.load_config(config_path)
        
        # Override with environment variables
        self._load_from_env()
    
    def load_config(self, config_path: str):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Update configurations
        if "model" in config_data:
            self._update_dataclass(self.model, config_data["model"])
        if "preprocessing" in config_data:
            self._update_dataclass(self.preprocessing, config_data["preprocessing"])
        if "layout" in config_data:
            self._update_dataclass(self.layout, config_data["layout"])
        if "translation" in config_data:
            self._update_dataclass(self.translation, config_data["translation"])
        if "extraction" in config_data:
            self._update_dataclass(self.extraction, config_data["extraction"])
        if "xai" in config_data:
            self._update_dataclass(self.xai, config_data["xai"])
        if "api" in config_data:
            self._update_dataclass(self.api, config_data["api"])
    
    def _update_dataclass(self, obj: Any, data: Dict):
        """Update dataclass fields from dictionary."""
        for key, value in data.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        # Model config
        if os.getenv("DEEPSEEK_MODEL"):
            self.model.deepseek_model = os.getenv("DEEPSEEK_MODEL")
        if os.getenv("BATCH_SIZE"):
            self.model.batch_size = int(os.getenv("BATCH_SIZE"))
        
        # API config
        if os.getenv("API_PORT"):
            self.api.port = int(os.getenv("API_PORT"))
        if os.getenv("API_WORKERS"):
            self.api.workers = int(os.getenv("API_WORKERS"))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return {
            "model": self.model.__dict__,
            "preprocessing": self.preprocessing.__dict__,
            "layout": self.layout.__dict__,
            "translation": self.translation.__dict__,
            "extraction": self.extraction.__dict__,
            "xai": self.xai.__dict__,
            "api": self.api.__dict__
        }
    
    def save_config(self, path: str):
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

