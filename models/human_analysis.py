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


def preprocess_face_for_emotion(image_rgb: np.ndarray) -> np.ndarray:
    """
    Preprocess face image for better emotion detection.
    
    Args:
        image_rgb: RGB face image
        
    Returns:
        Preprocessed RGB image
    """
    # Convert to BGR for OpenCV processing
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve contrast
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    enhanced_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Apply slight bilateral filter to reduce noise while preserving edges
    denoised = cv2.bilateralFilter(enhanced_bgr, 5, 50, 50)
    
    # Convert back to RGB
    processed_rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
    
    return processed_rgb


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
    
    def _refine_emotion(self, emotion_scores: Dict[str, float], dominant: str) -> str:
        """
        Refine emotion detection with post-processing rules.
        Helps distinguish between sad, neutral, fear, and angry.
        
        Args:
            emotion_scores: Dictionary of emotion scores
            dominant: Original dominant emotion
            
        Returns:
            Refined emotion
        """
        # Get scores for commonly confused emotions
        sad = emotion_scores.get('sad', 0)
        neutral = emotion_scores.get('neutral', 0)
        fear = emotion_scores.get('fear', 0)
        angry = emotion_scores.get('angry', 0)
        happy = emotion_scores.get('happy', 0)
        
        # If happy is very low, likely a negative emotion
        if happy < 10:
            # Get top 3 negative emotions
            negative_emotions = {
                'sad': sad,
                'neutral': neutral,
                'fear': fear,
                'angry': angry
            }
            
            # If scores are close (within 15%), prefer sad over neutral/fear
            max_score = max(negative_emotions.values())
            close_emotions = [k for k, v in negative_emotions.items() if v >= max_score - 15]
            
            # If sad is in close emotions and neutral/fear is dominant, switch to sad
            if 'sad' in close_emotions and dominant in ['neutral', 'fear']:
                # Additional check: if angry is not much higher, prefer sad
                if angry < sad + 20:
                    return 'sad'
            
            # If fear and sad are close but fear is dominant, and angry is low, prefer sad
            if dominant == 'fear' and abs(fear - sad) < 10 and angry < sad:
                return 'sad'
                
            # If neutral is dominant but sad is close and angry is low, prefer sad
            if dominant == 'neutral' and sad > neutral - 10 and angry < sad:
                return 'sad'
        
        return dominant
    
    def _preprocess_for_emotion(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better emotion detection.
        
        - Enhance contrast
        - Normalize brightness
        - Reduce noise
        """
        # Convert to LAB color space for better processing
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Merge channels
        lab = cv2.merge([l, a, b])
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Light denoising to reduce noise without losing facial details
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 5, 5, 7, 21)
        
        return enhanced
    
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
        
        # Preprocess image for better emotion detection
        image_rgb = self._preprocess_for_emotion(image_rgb)
        
        try:
            # Analyze with DeepFace
            # Note: DeepFace expects RGB image or file path
            result = DeepFace.analyze(
                img_path=image_rgb,
                actions=['age', 'gender', 'emotion'],
                enforce_detection=False,  # More flexible face detection
                silent=True,
                detector_backend=self.backend
            )
            
            # Handle both single dict and list of dicts
            if isinstance(result, list):
                result = result[0]
            
            # Extract attributes with all emotion scores
            emotion_scores = result.get("emotion", {})
            dominant_emotion = result.get("dominant_emotion", "Unknown")
            
            # Post-process emotion to improve sad detection
            if emotion_scores:
                dominant_emotion = self._refine_emotion(emotion_scores, dominant_emotion)
            
            attributes = {
                "age": float(result.get("age", 0)),
                "gender": result.get("dominant_gender", "Unknown"),
                "emotion": dominant_emotion,
                "confidence": 0.8,  # DeepFace doesn't provide overall confidence
                "gender_confidence": result.get("gender", {}).get(result.get("dominant_gender", ""), 0.0) if isinstance(result.get("gender"), dict) else 0.0,
                "emotion_scores": {k: float(v) for k, v in emotion_scores.items()} if emotion_scores else {},
                "emotion_confidence": emotion_scores.get(dominant_emotion, 0.0) if emotion_scores else 0.0
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

