"""
Preprocessing Pipeline for Document Restoration
Handles degraded inputs, watermarks, geometric distortions
"""

import logging
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image
import cv2

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    Multi-stage preprocessing pipeline for document restoration.
    
    Stages:
    1. Image Restoration (U-Net based)
    2. Watermark Suppression (Fourier domain)
    3. Geometric Correction (angle detection & correction)
    4. Intelligent Binarization (adaptive thresholding)
    """
    
    def __init__(self, settings):
        """Initialize preprocessing pipeline."""
        self.settings = settings
        self.config = settings.preprocessing
        logger.info("Preprocessing pipeline initialized")
    
    def process(self, input_path: str) -> Dict[str, Any]:
        """
        Process document through complete preprocessing pipeline.
        
        Args:
            input_path: Path to input document image
            
        Returns:
            Dictionary containing:
                - image: Preprocessed image
                - restored: Image after restoration
                - corrected: Image after geometric correction
                - quality_score: Quality assessment score
        """
        logger.info(f"Starting preprocessing: {input_path}")
        
        # Load image
        image = self._load_image(input_path)
        original = image.copy()
        
        results = {
            "original": original,
            "restored": None,
            "corrected": None,
            "quality_score": 0.0
        }
        
        # Stage 1: Image Restoration
        if self.config.enable_restoration:
            logger.info("Stage 1: Image restoration...")
            image = self._restore_image(image)
            results["restored"] = image
        
        # Stage 2: Watermark Suppression
        if self.config.enable_watermark_removal:
            logger.info("Stage 2: Watermark suppression...")
            image = self._suppress_watermark(image)
        
        # Stage 3: Geometric Correction
        if self.config.enable_geometric_correction:
            logger.info("Stage 3: Geometric correction...")
            image = self._correct_geometry(image)
            results["corrected"] = image
        
        # Stage 4: Binarization (optional)
        if self.config.enable_binarization:
            logger.info("Stage 4: Binarization...")
            image = self._binarize(image)
        
        # Calculate quality score
        results["quality_score"] = self._assess_quality(original, image)
        results["image"] = image
        
        logger.info(f"Preprocessing completed. Quality score: {results['quality_score']:.3f}")
        return results
    
    def _load_image(self, path: str) -> np.ndarray:
        """Load and convert image to numpy array."""
        image = Image.open(path)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if too large
        max_size = self.config.max_image_size
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.LANCZOS)
            logger.info(f"Resized image to {new_size}")
        
        return np.array(image)
    
    def _restore_image(self, image: np.ndarray) -> np.ndarray:
        """
        Restore degraded image using U-Net based approach.
        
        Note: This is a simplified version. Full implementation would use
        trained deep learning model for restoration.
        """
        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Denoise using Non-local Means Denoising
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Convert back to RGB
        restored = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        
        return restored
    
    def _suppress_watermark(self, image: np.ndarray) -> np.ndarray:
        """
        Suppress watermarks using Fourier domain filtering.
        
        Watermarks typically exist in low-frequency components.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Fourier Transform
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        
        # Create high-pass filter mask
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        
        # Define filter parameters
        radius = 30
        mask = np.ones((rows, cols), np.uint8)
        center = (crow, ccol)
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        mask[mask_area] = 0
        
        # Apply mask
        f_shift_filtered = f_shift * mask
        
        # Inverse Fourier Transform
        f_ishift = np.fft.ifftshift(f_shift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        
        # Normalize
        img_back = np.uint8(img_back / img_back.max() * 255)
        
        # Convert back to RGB
        result = cv2.cvtColor(img_back, cv2.COLOR_GRAY2RGB)
        
        return result
    
    def _correct_geometry(self, image: np.ndarray) -> np.ndarray:
        """
        Correct geometric distortions (skew, rotation).
        
        Uses Hough Transform for angle detection.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using Hough Transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        
        if lines is not None:
            # Calculate dominant angle
            angles = []
            for rho, theta in lines[:, 0]:
                angle = np.degrees(theta) - 90
                if abs(angle) < 45:  # Only consider angles within ±45°
                    angles.append(angle)
            
            if angles:
                # Use median angle
                rotation_angle = np.median(angles)
                
                # Rotate image
                if abs(rotation_angle) > 0.5:  # Only rotate if significant
                    logger.info(f"Rotating image by {rotation_angle:.2f}°")
                    h, w = image.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
                    image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                                          borderMode=cv2.BORDER_REPLICATE)
        
        return image
    
    def _binarize(self, image: np.ndarray) -> np.ndarray:
        """
        Intelligent binarization using adaptive thresholding.
        
        Better than Otsu for variable illumination.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        
        # Convert back to RGB for consistency
        result = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        
        return result
    
    def _assess_quality(self, original: np.ndarray, processed: np.ndarray) -> float:
        """
        Assess quality improvement.
        
        Uses metrics like contrast, sharpness, etc.
        """
        # Convert to grayscale
        orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        proc_gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
        
        # Calculate contrast (standard deviation)
        contrast_orig = np.std(orig_gray)
        contrast_proc = np.std(proc_gray)
        
        # Calculate sharpness (Laplacian variance)
        sharpness_orig = cv2.Laplacian(orig_gray, cv2.CV_64F).var()
        sharpness_proc = cv2.Laplacian(proc_gray, cv2.CV_64F).var()
        
        # Combine metrics (normalized to 0-1)
        contrast_score = min(contrast_proc / (contrast_orig + 1e-6), 2.0) / 2.0
        sharpness_score = min(sharpness_proc / (sharpness_orig + 1e-6), 2.0) / 2.0
        
        quality_score = (contrast_score + sharpness_score) / 2.0
        
        return min(quality_score, 1.0)

