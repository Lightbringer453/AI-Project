"""
Utility functions for loading and preprocessing images
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional
from PIL import Image


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    Load an image from file path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Image as numpy array (BGR format for OpenCV)
    """
    if isinstance(image_path, Path):
        image_path = str(image_path)
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    return image


def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Load an image from bytes.
    
    Args:
        image_bytes: Image data as bytes
        
    Returns:
        Image as numpy array (BGR format for OpenCV)
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image from bytes")
    return image


def preprocess_image(image: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Preprocess image for model input.
    
    Args:
        image: Input image as numpy array
        target_size: Optional target size (width, height)
        
    Returns:
        Preprocessed image
    """
    if target_size:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    
    return image


def resize_image_max_dim(image: np.ndarray, max_dim: int = 1920) -> np.ndarray:
    """
    Resize image maintaining aspect ratio, with max dimension constraint.
    
    Args:
        image: Input image
        max_dim: Maximum dimension (width or height)
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    
    if h > w:
        new_h = max_dim
        new_w = int(w * (max_dim / h))
    else:
        new_w = max_dim
        new_h = int(h * (max_dim / w))
    
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def crop_image(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Crop image using bounding box coordinates.
    
    Args:
        image: Input image
        bbox: Bounding box as (x1, y1, x2, y2)
        
    Returns:
        Cropped image
    """
    x1, y1, x2, y2 = bbox
    # Ensure coordinates are within image bounds
    h, w = image.shape[:2]
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    
    return image[y1:y2, x1:x2]


def convert_bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to RGB."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def convert_rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert RGB image to BGR."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def image_to_pil(image: np.ndarray) -> Image.Image:
    """Convert numpy array image to PIL Image."""
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert BGR to RGB for PIL
        image = convert_bgr_to_rgb(image)
    return Image.fromarray(image)

