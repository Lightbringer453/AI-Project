"""
Configuration file for the Multi-Purpose Image Detection and Analysis System
"""


YOLO_MODEL_PATH = "yolov8n.pt"  

CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

HUMAN_CLASS_ID = 0  
ANIMAL_CLASS_IDS = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]  

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

DEEPFACE_MODEL = "VGG-Face"  
DEEPFACE_BACKEND = "opencv"  

IMAGE_SIZE = (640, 640)  
MAX_IMAGE_SIZE = 1920  

OUTPUT_DIR = "outputs"
SAVE_ANNOTATED_IMAGES = True

