"""
Detectron2-based object detector for humans and animals (alternative to YOLOv8)
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path

try:
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog
    DETECTRON2_AVAILABLE = True
except ImportError:
    DETECTRON2_AVAILABLE = False
    print("Warning: Detectron2 not available. Install with: pip install detectron2")

from utils.config import (
    HUMAN_CLASS_ID,
    ANIMAL_CLASS_IDS,
    ANIMAL_CLASS_NAMES,
    CONFIDENCE_THRESHOLD,
    DETECTRON2_CONFIG,
    DETECTRON2_WEIGHTS
)


class Detectron2Detector:
    """Detectron2 detector for human and animal detection."""
    
    def __init__(self, config_path: str = None, weights_path: str = None, device: str = "cpu"):
        """
        Initialize Detectron2 detector.
        
        Args:
            config_path: Path to config file (uses default if None)
            weights_path: Path to weights file (uses default if None)
            device: Device to run on ('cpu' or 'cuda')
        """
        if not DETECTRON2_AVAILABLE:
            raise ImportError(
                "Detectron2 is not installed. Install with: pip install detectron2"
            )
        
        self.config_path = config_path or DETECTRON2_CONFIG
        self.weights_path = weights_path or DETECTRON2_WEIGHTS
        self.device = device
        self.predictor = None
        self._load_model()
    
    def _load_model(self):
        """Load Detectron2 model."""
        try:
            cfg = get_cfg()
            
            if self.config_path.startswith("COCO"):
                cfg.merge_from_file(None)  
            else:
                cfg.merge_from_file(self.config_path)
            
            if self.weights_path.startswith("detectron2://"):
                cfg.MODEL.WEIGHTS = self.weights_path
            else:
                cfg.MODEL.WEIGHTS = self.weights_path
            
            cfg.MODEL.DEVICE = self.device
            
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
            
            self.predictor = DefaultPredictor(cfg)
            print(f"Detectron2 model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load Detectron2 model: {e}")
    
    def detect(self, image: np.ndarray, conf_threshold: float = None) -> List[Dict]:
        """
        Detect humans and animals in image.
        
        Args:
            image: Input image as numpy array (BGR format)
            conf_threshold: Confidence threshold (uses default if None)
            
        Returns:
            List of detection dictionaries
        """
        if conf_threshold is None:
            conf_threshold = CONFIDENCE_THRESHOLD
        
        outputs = self.predictor(image)
        
        instances = outputs["instances"]
        detections = []
        
        boxes = instances.pred_boxes.tensor.cpu().numpy()
        scores = instances.scores.cpu().numpy()
        classes = instances.pred_classes.cpu().numpy()
        
        for i in range(len(instances)):
            if scores[i] < conf_threshold:
                continue
            
            class_id = int(classes[i])
            confidence = float(scores[i])
            
            box = boxes[i]
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            
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
                    "class_name": class_name,
                    "confidence": confidence,
                    "class_type": "animal"
                }
                detections.append(detection)
        
        return detections
    
    def detect_and_crop(self, image: np.ndarray, conf_threshold: float = None) -> List[Dict]:
        """
        Detect objects and return cropped regions.
        """
        detections = self.detect(image, conf_threshold)
        
        for detection in detections:
            bbox = detection["bbox"]
            crop = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            detection["crop"] = crop
        
        return detections

