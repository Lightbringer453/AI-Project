"""
Configuration file for the Multi-Purpose Image Detection and Analysis System
"""

# Model paths and settings
YOLO_MODEL_PATH = "yolov8n.pt"  # Will download automatically if not present

# Detection settings
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

# Class IDs (COCO dataset)
HUMAN_CLASS_ID = 0  # person
ANIMAL_CLASS_IDS = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]  # Various animals in COCO

# Animal class mapping (COCO dataset)
ANIMAL_CLASS_NAMES = {
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "other"
}

# DeepFace settings
DEEPFACE_MODEL = "VGG-Face"  # Options: VGG-Face, Facenet, ArcFace, OpenFace, DeepFace, DeepID, Dlib
DEEPFACE_BACKEND = "opencv"  # Options: opencv, ssd, dlib, mtcnn, retinaface

# Image processing settings
IMAGE_SIZE = (640, 640)  # Standard YOLO input size
MAX_IMAGE_SIZE = 1920  # Maximum dimension for processing

# Output settings
OUTPUT_DIR = "outputs"
SAVE_ANNOTATED_IMAGES = True

