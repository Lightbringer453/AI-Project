"""
Example usage script for the Image Detection and Analysis System

This script demonstrates how to use the pipeline programmatically.
"""

from app.main import ImageAnalysisPipeline
from utils.image_loader import load_image
import json


def main():
    """Example usage of the image analysis pipeline."""
    
    # Example 1: Initialize pipeline with YOLOv8
    print("Initializing pipeline...")
    pipeline = ImageAnalysisPipeline(detector_type="yolo")
    
    # Example 2: Process an image (replace with your image path)
    image_path = "test_image.jpg"  # Update this with your image path
    
    try:
        print(f"\nProcessing image: {image_path}")
        result = pipeline.process_image(image_path, save_output=True)
        
        # Example 3: Access results
        print("\n" + "="*50)
        print("SUMMARY")
        print("="*50)
        print(f"Total detections: {result['summary']['total_detections']}")
        print(f"Humans detected: {result['summary']['humans']}")
        print(f"Animals detected: {result['summary']['animals']}")
        
        # Example 4: Access individual detections
        print("\n" + "="*50)
        print("DETAILED RESULTS")
        print("="*50)
        for i, detection in enumerate(result['detections'], 1):
            print(f"\nDetection {i}:")
            print(f"  Type: {detection['class_type']}")
            print(f"  Class: {detection.get('class_name', 'N/A')}")
            print(f"  Confidence: {detection.get('confidence', 0.0):.2%}")
            
            attributes = detection.get('attributes', {})
            if detection['class_type'] == 'human':
                print(f"  Age: {attributes.get('age', 'N/A')}")
                print(f"  Gender: {attributes.get('gender', 'N/A')}")
                print(f"  Emotion: {attributes.get('emotion', 'N/A')}")
            elif detection['class_type'] == 'animal':
                print(f"  Species: {attributes.get('species', 'N/A')}")
                print(f"  Breed: {attributes.get('breed', 'N/A')}")
                print(f"  Maturity: {attributes.get('maturity', 'N/A')}")
        
        # Example 5: Get JSON output
        print("\n" + "="*50)
        print("JSON OUTPUT")
        print("="*50)
        json_output = pipeline.process_image_to_json(image_path)
        print(json_output)
        
        # Example 6: Save JSON to file
        with open("outputs/result.json", "w") as f:
            f.write(json_output)
        print("\nJSON saved to outputs/result.json")
        
    except FileNotFoundError:
        print(f"\nError: Image file '{image_path}' not found.")
        print("Please update the 'image_path' variable with a valid image path.")
        print("\nExample:")
        print("  image_path = 'path/to/your/image.jpg'")
    except Exception as e:
        print(f"\nError processing image: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

