"""
Human attribute analysis using DeepFace
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("Warning: DeepFace not available. Install with: pip install deepface")

from utils.config import DEEPFACE_MODEL, DEEPFACE_BACKEND
from utils.image_loader import convert_bgr_to_rgb


class HumanAnalyzer:
    """Analyzer for human attributes using DeepFace."""
    
    def __init__(self, model_name: str = None, backend: str = None):
        """
        Initialize human analyzer.
        
        Args:
            model_name: DeepFace model name (VGG-Face, Facenet, ArcFace, etc.)
            backend: Backend for face detection (opencv, ssd, dlib, mtcnn, retinaface)
        """
        if not DEEPFACE_AVAILABLE:
            raise ImportError(
                "DeepFace is not installed. Install with: pip install deepface"
            )
        
        self.model_name = model_name or DEEPFACE_MODEL
        self.backend = backend or DEEPFACE_BACKEND
        print(f"HumanAnalyzer initialized with model: {self.model_name}, backend: {self.backend}")
    
    def analyze_human(self, image_crop: np.ndarray) -> Dict:
        """
        Analyze human attributes from cropped image.
        
        Args:
            image_crop: Cropped image containing human (BGR format)
            
        Returns:
            Dictionary with keys:
            - age: Estimated age
            - gender: Gender classification
            - emotion: Emotion detection (optional)
            - confidence: Overall confidence
        """
        if image_crop is None or image_crop.size == 0:
            return {
                "age": None,
                "gender": None,
                "emotion": None,
                "confidence": 0.0,
                "error": "Empty image crop"
            }
        
        # Convert BGR to RGB for DeepFace
        image_rgb = convert_bgr_to_rgb(image_crop)
        
        try:
            # Analyze with DeepFace
            # Note: DeepFace expects RGB image or file path
            result = DeepFace.analyze(
                img=image_rgb,
                actions=['age', 'gender', 'emotion'],
                enforce_detection=False,  # Don't fail if face not detected
                silent=True,
                model_name=self.model_name,
                detector_backend=self.backend
            )
            
            # Handle both single dict and list of dicts
            if isinstance(result, list):
                result = result[0]
            
            # Extract attributes
            attributes = {
                "age": float(result.get("age", 0)),
                "gender": result.get("dominant_gender", "Unknown"),
                "emotion": result.get("dominant_emotion", "Unknown"),
                "confidence": 0.8,  # DeepFace doesn't provide overall confidence
                "gender_confidence": result.get("gender", {}).get(result.get("dominant_gender", ""), 0.0) if isinstance(result.get("gender"), dict) else 0.0
            }
            
            return attributes
            
        except Exception as e:
            # If face detection fails, return default values
            print(f"DeepFace analysis failed: {e}")
            return {
                "age": None,
                "gender": None,
                "emotion": None,
                "confidence": 0.0,
                "error": str(e)
            }
    
    def analyze_multiple(self, image_crops: list) -> list:
        """
        Analyze multiple human crops.
        
        Args:
            image_crops: List of cropped images
            
        Returns:
            List of attribute dictionaries
        """
        results = []
        for crop in image_crops:
            result = self.analyze_human(crop)
            results.append(result)
        return results

