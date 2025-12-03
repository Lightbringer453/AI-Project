"""
YOLOv8-based object detector for humans and animals
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
from pathlib import Path

from utils.config import (
    HUMAN_CLASS_ID,
    ANIMAL_CLASS_IDS,
    ANIMAL_CLASS_NAMES,
    CONFIDENCE_THRESHOLD
)


class YOLODetector:
    """YOLOv8 detector for human and animal detection."""
    
    def __init__(self, model_path: str = "yolov8n.pt", device: Optional[str] = None):
        """
        Initialize YOLOv8 detector.
        
        Args:
            model_path: Path to YOLOv8 model weights
            device: Device to run on ('cpu', 'cuda', 'mps', etc.)
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load YOLOv8 model."""
        try:
            self.model = YOLO(self.model_path)
            if self.device:
                self.model.to(self.device)
            print(f"YOLOv8 model loaded successfully from {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLOv8 model: {e}")
    
    def detect(self, image: np.ndarray, conf_threshold: float = None) -> List[Dict]:
        """
        Detect humans and animals in image.
        
        Args:
            image: Input image as numpy array (BGR format)
            conf_threshold: Confidence threshold (uses default if None)
            
        Returns:
            List of detection dictionaries with keys:
            - bbox: (x1, y1, x2, y2)
            - class_id: COCO class ID
            - class_name: Class name
            - confidence: Confidence score
            - class_type: 'human' or 'animal'
        """
        if conf_threshold is None:
            conf_threshold = CONFIDENCE_THRESHOLD
        
        # Run inference
        results = self.model(image, conf=conf_threshold, verbose=False)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Get class and confidence
                class_id = int(box.cls[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                
                # Filter for humans and animals only
                if class_id == HUMAN_CLASS_ID:
                    detection = {
                        "bbox": (x1, y1, x2, y2),
                        "class_id": class_id,
                        "class_name": "person",
                        "confidence": confidence,
                        "class_type": "human"
                    }
                    detections.append(detection)
                elif class_id in ANIMAL_CLASS_IDS:
                    class_name = ANIMAL_CLASS_NAMES.get(class_id, "animal")
                    detection = {
                        "bbox": (x1, y1, x2, y2),
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": confidence,
                        "class_type": "animal"
                    }
                    detections.append(detection)
        
        return detections
    
    def detect_and_crop(self, image: np.ndarray, conf_threshold: float = None) -> List[Dict]:
        """
        Detect objects and return cropped regions.
        
        Args:
            image: Input image
            conf_threshold: Confidence threshold
            
        Returns:
            List of detection dictionaries with additional 'crop' key containing cropped image
        """
        detections = self.detect(image, conf_threshold)
        
        for detection in detections:
            bbox = detection["bbox"]
            crop = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            detection["crop"] = crop
        
        return detections

