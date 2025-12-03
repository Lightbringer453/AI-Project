"""
Models package for image detection and analysis
"""

from .yolo_detector import YOLODetector
from .human_analysis import HumanAnalyzer
from .animal_analysis import AnimalAnalyzer

__all__ = ['YOLODetector', 'HumanAnalyzer', 'AnimalAnalyzer']

