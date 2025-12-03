# Multi-Purpose Image Detection and Analysis System

An AI-powered image analysis tool that detects humans and animals from images and provides detailed attributes for each detected class.

## Features

### Human Detection
- **Age estimation** using DeepFace
- **Gender classification** using DeepFace
- **Emotion detection** (optional)
- **Confidence scores**

### Animal Detection
- **Species classification** (dog, cat, bird, horse, cow, sheep, etc.)
- **Breed estimation** (placeholder implementation - ready for custom models)
- **Maturity state** (juvenile/adult)
- **Confidence scores**

## Project Structure

```
/project
 ├── models/
 │    ├── yolo_detector.py          # YOLOv8-based detector
 │    ├── detectron_detector.py     # Detectron2-based detector (alternative)
 │    ├── human_analysis.py          # Human attribute analysis with DeepFace
 │    └── animal_analysis.py         # Animal attribute analysis
 ├── utils/
 │    ├── image_loader.py            # Image loading and preprocessing utilities
 │    ├── draw_results.py            # Visualization utilities
 │    └── config.py                  # Configuration settings
 ├── notebooks/
 │    └── demo.ipynb                 # Jupyter notebook demo
 ├── app/
 │    ├── main.py                    # Main pipeline implementation
 │    └── interface.py               # Streamlit web interface
 ├── outputs/                        # Output directory (created automatically)
 ├── requirements.txt                # Python dependencies
 └── README.md                       # This file
```

## Installation

### 1. Clone or download this repository

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Detectron2 (optional, if you want to use Detectron2 instead of YOLOv8)

For CPU:
```bash
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.0/index.html
```

For CUDA (GPU):
```bash
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html
```

**Note**: YOLOv8 models will be downloaded automatically on first use.

## Usage

### 1. Command Line Interface

```bash
python app/main.py path/to/image.jpg
```

This will:
- Detect humans and animals in the image
- Analyze attributes for each detection
- Save annotated image to `outputs/` directory
- Save JSON results to `outputs/` directory
- Print summary and detailed results to console

### 2. Python Script

```python
from app.main import ImageAnalysisPipeline
from utils.image_loader import load_image

# Initialize pipeline
pipeline = ImageAnalysisPipeline(detector_type="yolo")

# Process image
result = pipeline.process_image("path/to/image.jpg")

# Access results
print(f"Total detections: {result['summary']['total_detections']}")
for detection in result['detections']:
    print(f"Type: {detection['class_type']}")
    print(f"Attributes: {detection['attributes']}")
```

### 3. Streamlit Web Interface

```bash
streamlit run app/interface.py
```

Then open your browser to the URL shown (typically `http://localhost:8501`).

Features:
- Upload images via web interface
- View annotated results
- See detailed attributes
- Download JSON output
- Adjustable settings (detector type, confidence threshold)

### 4. Jupyter Notebook

Open `notebooks/demo.ipynb` in Jupyter and follow the cells to:
- Load and process images
- Visualize results
- Test individual components
- Export JSON output

## Configuration

Edit `utils/config.py` to customize:

- Model paths
- Confidence thresholds
- Image processing settings
- DeepFace model and backend
- Output directory

## Models Used

### Object Detection
- **YOLOv8** (default): Fast and accurate object detection
- **Detectron2** (alternative): Facebook's detection framework

### Human Analysis
- **DeepFace**: Uses VGG-Face, Facenet, or ArcFace models for:
  - Age estimation
  - Gender classification
  - Emotion detection

### Animal Analysis
- **Rule-based heuristics** (current implementation)
- Ready for integration with custom trained models for:
  - Species classification
  - Breed identification
  - Maturity estimation

## Output Format

### JSON Output Example

```json
{
  "summary": {
    "total_detections": 2,
    "humans": 1,
    "animals": 1,
    "timestamp": "2024-01-01T12:00:00"
  },
  "detections": [
    {
      "bbox": [100, 150, 300, 450],
      "class_type": "human",
      "class_name": "person",
      "confidence": 0.95,
      "attributes": {
        "age": 25.0,
        "gender": "Man",
        "emotion": "happy",
        "confidence": 0.8
      }
    },
    {
      "bbox": [400, 200, 600, 500],
      "class_type": "animal",
      "class_name": "dog",
      "confidence": 0.87,
      "attributes": {
        "species": "dog",
        "breed": "Golden Retriever",
        "maturity": "adult",
        "confidence": 0.7
      }
    }
  ]
}
```

## Deployment

### Google Colab

1. Upload the project to Google Drive or GitHub
2. Open a Colab notebook
3. Install dependencies:
   ```python
   !pip install -r requirements.txt
   ```
4. Run the pipeline as shown in `notebooks/demo.ipynb`

### Hugging Face Spaces

1. Create a new Space with Streamlit SDK
2. Upload all project files
3. Add `requirements.txt`
4. The `app/interface.py` will automatically be used as the Streamlit app

## Troubleshooting

### DeepFace Installation Issues

If you encounter issues with DeepFace:
```bash
pip install --upgrade deepface
pip install tensorflow
```

### CUDA/GPU Support

For GPU acceleration:
- Install PyTorch with CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
- Set device in pipeline: `ImageAnalysisPipeline(device="cuda")`

### Model Download

YOLOv8 models are downloaded automatically. If download fails:
- Check internet connection
- Manually download from: https://github.com/ultralytics/assets/releases
- Place in project root or specify path in config

## Future Enhancements

- [ ] Train custom animal breed classifier
- [ ] Add more animal species support
- [ ] Implement batch processing
- [ ] Add video processing support
- [ ] Improve maturity estimation with ML models
- [ ] Add pose estimation for humans
- [ ] Support for multiple faces per person

## License

This project is provided as-is for educational and research purposes.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

- YOLOv8 by Ultralytics
- DeepFace by Sefik Ilkin Serengil
- Detectron2 by Facebook AI Research
- COCO Dataset for object detection classes

