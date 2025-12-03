"""
Main pipeline for image detection and analysis
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import json
from datetime import datetime

from models.yolo_detector import YOLODetector
from models.human_analysis import HumanAnalyzer
from models.animal_analysis import AnimalAnalyzer
from utils.image_loader import load_image, resize_image_max_dim, crop_image
from utils.draw_results import draw_all_detections
from utils.config import OUTPUT_DIR, SAVE_ANNOTATED_IMAGES, MAX_IMAGE_SIZE


class ImageAnalysisPipeline:
    """Main pipeline for image detection and analysis."""
    
    def __init__(
        self,
        detector_type: str = "yolo",
        yolo_model_path: str = "yolov8n.pt",
        device: Optional[str] = None
    ):
        """
        Initialize the analysis pipeline.
        
        Args:
            detector_type: 'yolo' or 'detectron2'
            yolo_model_path: Path to YOLOv8 model
            device: Device to run on ('cpu', 'cuda', etc.)
        """
        self.detector_type = detector_type.lower()
        self.device = device
        
        # Initialize detector
        if self.detector_type == "yolo":
            self.detector = YOLODetector(model_path=yolo_model_path, device=device)
        elif self.detector_type == "detectron2":
            from models.detectron_detector import Detectron2Detector
            self.detector = Detectron2Detector(device=device or "cpu")
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")
        
        # Initialize analyzers
        self.human_analyzer = HumanAnalyzer()
        self.animal_analyzer = AnimalAnalyzer()
        
        # Create output directory
        Path(OUTPUT_DIR).mkdir(exist_ok=True)
        
        print(f"Pipeline initialized with {detector_type} detector")
    
    def process_image(
        self,
        image_path: Union[str, Path, np.ndarray],
        save_output: bool = None,
        draw_annotations: bool = True,
        conf_threshold: float = None
    ) -> Dict:
        """
        Process a single image through the complete pipeline.
        
        Args:
            image_path: Path to image file or numpy array
            save_output: Whether to save annotated image
            draw_annotations: Whether to draw annotations on image
            
        Returns:
            Dictionary with:
            - detections: List of detection results
            - annotated_image: Annotated image (if draw_annotations=True)
            - summary: Summary statistics
        """
        # Load image
        if isinstance(image_path, (str, Path)):
            image = load_image(image_path)
        elif isinstance(image_path, np.ndarray):
            image = image_path.copy()
        else:
            raise ValueError("Invalid image input type")
        
        # Resize if too large
        image = resize_image_max_dim(image, MAX_IMAGE_SIZE)
        
        # Step 1: Detect objects
        detections = self.detector.detect(image, conf_threshold=conf_threshold)
        
        # Step 2: Analyze each detection
        results = []
        for detection in detections:
            bbox = detection["bbox"]
            crop = crop_image(image, bbox)
            class_type = detection["class_type"]
            
            if class_type == "human":
                # Analyze human
                attributes = self.human_analyzer.analyze_human(crop)
                detection["attributes"] = attributes
            elif class_type == "animal":
                # Analyze animal
                species = detection.get("class_name", "unknown")
                detection_confidence = detection.get("confidence", None)
                attributes = self.animal_analyzer.analyze_animal(
                    crop, 
                    species, 
                    detection_confidence=detection_confidence
                )
                detection["attributes"] = attributes
            
            results.append(detection)
        
        # Step 3: Draw annotations
        annotated_image = None
        if draw_annotations:
            annotated_image = draw_all_detections(image, results, draw_attributes=True)
        
        # Step 4: Save output if requested
        if save_output is None:
            save_output = SAVE_ANNOTATED_IMAGES
        
        if save_output and annotated_image is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(OUTPUT_DIR) / f"result_{timestamp}.jpg"
            cv2.imwrite(str(output_path), annotated_image)
            print(f"Saved annotated image to {output_path}")
        
        # Create summary
        summary = {
            "total_detections": len(results),
            "humans": sum(1 for r in results if r["class_type"] == "human"),
            "animals": sum(1 for r in results if r["class_type"] == "animal"),
            "timestamp": datetime.now().isoformat()
        }
        
        return {
            "detections": results,
            "annotated_image": annotated_image,
            "summary": summary
        }
    
    def process_image_to_json(
        self,
        image_path: Union[str, Path, np.ndarray]
    ) -> str:
        """
        Process image and return JSON output.
        
        Args:
            image_path: Path to image file or numpy array
            
        Returns:
            JSON string with detection results
        """
        result = self.process_image(image_path, save_output=False, draw_annotations=False)
        
        # Convert to JSON-serializable format
        json_data = {
            "summary": result["summary"],
            "detections": []
        }
        
        for detection in result["detections"]:
            json_detection = {
                "bbox": detection["bbox"],
                "class_type": detection["class_type"],
                "class_name": detection.get("class_name", ""),
                "confidence": detection.get("confidence", 0.0),
                "attributes": detection.get("attributes", {})
            }
            json_data["detections"].append(json_detection)
        
        return json.dumps(json_data, indent=2)


def main():
    """Example usage of the pipeline."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_path>")
        print("Example: python main.py test_image.jpg")
        return
    
    image_path = sys.argv[1]
    
    # Initialize pipeline
    pipeline = ImageAnalysisPipeline(detector_type="yolo")
    
    # Process image
    print(f"Processing image: {image_path}")
    result = pipeline.process_image(image_path)
    
    # Print summary
    print("\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)
    print(f"Total detections: {result['summary']['total_detections']}")
    print(f"Humans: {result['summary']['humans']}")
    print(f"Animals: {result['summary']['animals']}")
    
    # Print detailed results
    print("\n" + "="*50)
    print("DETAILED RESULTS")
    print("="*50)
    for i, detection in enumerate(result["detections"], 1):
        print(f"\nDetection {i}:")
        print(f"  Type: {detection['class_type']}")
        print(f"  Class: {detection.get('class_name', 'N/A')}")
        print(f"  Confidence: {detection.get('confidence', 0.0):.2f}")
        print(f"  BBox: {detection['bbox']}")
        
        attributes = detection.get("attributes", {})
        if detection["class_type"] == "human":
            print(f"  Age: {attributes.get('age', 'N/A')}")
            print(f"  Gender: {attributes.get('gender', 'N/A')}")
            print(f"  Emotion: {attributes.get('emotion', 'N/A')}")
        elif detection["class_type"] == "animal":
            print(f"  Species: {attributes.get('species', 'N/A')}")
            print(f"  Breed: {attributes.get('breed', 'N/A')}")
            print(f"  Maturity: {attributes.get('maturity', 'N/A')}")
    
    # Save JSON output
    json_output = pipeline.process_image_to_json(image_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = Path(OUTPUT_DIR) / f"result_{timestamp}.json"
    with open(json_path, 'w') as f:
        f.write(json_output)
    print(f"\nJSON output saved to {json_path}")


if __name__ == "__main__":
    main()

