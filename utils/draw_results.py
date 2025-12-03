"""
Utility functions for drawing detection results and annotations on images
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional


def draw_bounding_box(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    label: str,
    confidence: float,
    color: Optional[Tuple[int, int, int]] = None
) -> np.ndarray:
    """
    Draw a single bounding box with label and confidence on image.
    
    Args:
        image: Input image
        bbox: Bounding box as (x1, y1, x2, y2)
        label: Class label
        confidence: Confidence score
        color: BGR color tuple (optional)
        
    Returns:
        Annotated image
    """
    x1, y1, x2, y2 = bbox
    
    # Default color (blue)
    if color is None:
        color = (255, 0, 0)
    
    # Draw rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    
    # Prepare label text
    label_text = f"{label}: {confidence:.2f}"
    
    # Calculate text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        label_text, font, font_scale, thickness
    )
    
    # Draw label background
    cv2.rectangle(
        image,
        (x1, y1 - text_height - baseline - 10),
        (x1 + text_width, y1),
        color,
        -1
    )
    
    # Draw label text
    cv2.putText(
        image,
        label_text,
        (x1, y1 - baseline - 5),
        font,
        font_scale,
        (255, 255, 255),
        thickness
    )
    
    return image


def draw_human_attributes(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    attributes: Dict,
    color: Optional[Tuple[int, int, int]] = None
) -> np.ndarray:
    """
    Draw human detection with attributes.
    
    Args:
        image: Input image
        bbox: Bounding box
        attributes: Dictionary with age, gender, confidence, etc.
        color: BGR color tuple
        
    Returns:
        Annotated image
    """
    if color is None:
        color = (0, 255, 0)  # Green for humans
    
    x1, y1, x2, y2 = bbox
    
    # Draw bounding box
    image = draw_bounding_box(
        image, bbox, "Human", attributes.get("confidence", 0.0), color
    )
    
    # Prepare attribute text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    
    attributes_text = []
    if "age" in attributes and attributes['age'] is not None:
        attributes_text.append(f"Age: {attributes['age']:.0f}")
    if "gender" in attributes and attributes['gender'] is not None:
        attributes_text.append(f"Gender: {attributes['gender']}")
    if "emotion" in attributes and attributes['emotion'] is not None:
        attributes_text.append(f"Emotion: {attributes['emotion']}")
    
    # Draw attributes below bounding box
    y_offset = y2 + 20
    for i, attr_text in enumerate(attributes_text):
        cv2.putText(
            image,
            attr_text,
            (x1, y_offset + i * 20),
            font,
            font_scale,
            color,
            thickness
        )
    
    return image


def draw_animal_attributes(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    attributes: Dict,
    color: Optional[Tuple[int, int, int]] = None
) -> np.ndarray:
    """
    Draw animal detection with attributes.
    
    Args:
        image: Input image
        bbox: Bounding box
        attributes: Dictionary with species, breed, maturity, confidence, etc.
        color: BGR color tuple
        
    Returns:
        Annotated image
    """
    if color is None:
        color = (0, 165, 255)  # Orange for animals
    
    x1, y1, x2, y2 = bbox
    
    # Draw bounding box
    species = attributes.get("species", "Animal")
    image = draw_bounding_box(
        image, bbox, species, attributes.get("confidence", 0.0), color
    )
    
    # Prepare attribute text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    
    attributes_text = []
    if "breed" in attributes and attributes["breed"] is not None:
        attributes_text.append(f"Breed: {attributes['breed']}")
    if "maturity" in attributes and attributes["maturity"] is not None:
        attributes_text.append(f"Maturity: {attributes['maturity']}")
    
    # Draw attributes below bounding box
    y_offset = y2 + 20
    for i, attr_text in enumerate(attributes_text):
        cv2.putText(
            image,
            attr_text,
            (x1, y_offset + i * 20),
            font,
            font_scale,
            color,
            thickness
        )
    
    return image


def draw_all_detections(
    image: np.ndarray,
    detections: List[Dict],
    draw_attributes: bool = True
) -> np.ndarray:
    """
    Draw all detections on image.
    
    Args:
        image: Input image
        detections: List of detection dictionaries
        draw_attributes: Whether to draw detailed attributes
        
    Returns:
        Annotated image
    """
    image = image.copy()
    
    for detection in detections:
        bbox = detection["bbox"]
        class_type = detection.get("class_type", "unknown")
        attributes = detection.get("attributes", {})
        
        if class_type == "human":
            if draw_attributes:
                image = draw_human_attributes(image, bbox, attributes)
            else:
                image = draw_bounding_box(
                    image, bbox, "Human", attributes.get("confidence", 0.0)
                )
        elif class_type == "animal":
            if draw_attributes:
                image = draw_animal_attributes(image, bbox, attributes)
            else:
                species = attributes.get("species", "Animal")
                image = draw_bounding_box(
                    image, bbox, species, attributes.get("confidence", 0.0)
                )
        else:
            # Generic detection
            label = detection.get("label", "Unknown")
            confidence = detection.get("confidence", 0.0)
            image = draw_bounding_box(image, bbox, label, confidence)
    
    return image

