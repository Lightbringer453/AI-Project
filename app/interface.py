"""
Simple Streamlit UI for image upload and analysis
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import ImageAnalysisPipeline
from utils.image_loader import load_image_from_bytes


def main():
    """Streamlit application main function."""
    st.set_page_config(
        page_title="Image Detection & Analysis",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Multi-Purpose Image Detection and Analysis System")
    st.markdown("""
    Upload an image to detect humans and animals, then analyze their attributes:
    - **Humans**: Age, Gender, Emotion
    - **Animals**: Species, Breed, Maturity
    """)
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        detector_type = st.selectbox(
            "Detector Type",
            ["yolo", "detectron2"],
            index=0,
            help="Choose between YOLOv8 or Detectron2"
        )
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum confidence for detections"
        )
        
        show_json = st.checkbox("Show JSON Output", value=False)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload an image containing humans or animals"
    )
    
    if uploaded_file is not None:
        # Load image
        image_bytes = uploaded_file.read()
        image = load_image_from_bytes(image_bytes)
        
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original Image")
            st.image(image, channels="BGR", use_container_width=True)
        
        # Process image
        with st.spinner("Analyzing image..."):
            try:
                # Initialize pipeline
                pipeline = ImageAnalysisPipeline(detector_type=detector_type)
                
                # Process image
                result = pipeline.process_image(
                    image,
                    save_output=False,
                    draw_annotations=True,
                    conf_threshold=confidence_threshold
                )
                
                # Display annotated image
                with col2:
                    st.subheader("🎯 Analysis Results")
                    if result["annotated_image"] is not None:
                        st.image(result["annotated_image"], channels="BGR", use_container_width=True)
                    else:
                        st.info("No detections found")
                
                # Display summary
                st.subheader("📊 Summary")
                summary = result["summary"]
                col_sum1, col_sum2, col_sum3 = st.columns(3)
                
                with col_sum1:
                    st.metric("Total Detections", summary["total_detections"])
                with col_sum2:
                    st.metric("Humans", summary["humans"])
                with col_sum3:
                    st.metric("Animals", summary["animals"])
                
                # Display detailed results
                if result["detections"]:
                    st.subheader("🔍 Detailed Results")
                    
                    for i, detection in enumerate(result["detections"], 1):
                        with st.expander(f"Detection {i}: {detection['class_type'].title()}"):
                            col_det1, col_det2 = st.columns(2)
                            
                            with col_det1:
                                st.write("**Detection Info:**")
                                st.write(f"- Class: {detection.get('class_name', 'N/A')}")
                                st.write(f"- Confidence: {detection.get('confidence', 0.0):.2%}")
                                st.write(f"- BBox: {detection['bbox']}")
                            
                            with col_det2:
                                attributes = detection.get("attributes", {})
                                st.write("**Attributes:**")
                                
                                if detection["class_type"] == "human":
                                    age = attributes.get('age')
                                    if age:
                                        st.write(f"- Age: {age:.0f} years")
                                    else:
                                        st.write("- Age: N/A")
                                    st.write(f"- Gender: {attributes.get('gender', 'N/A')}")
                                    st.write(f"- Emotion: {attributes.get('emotion', 'N/A')}")
                                
                                elif detection["class_type"] == "animal":
                                    st.write(f"- Species: {attributes.get('species', 'N/A')}")
                                    breed = attributes.get('breed')
                                    if breed:
                                        st.write(f"- Breed: {breed}")
                                    else:
                                        st.write("- Breed: N/A")
                                    st.write(f"- Maturity: {attributes.get('maturity', 'N/A')}")
                
                # JSON output
                if show_json:
                    st.subheader("📄 JSON Output")
                    json_output = pipeline.process_image_to_json(image)
                    st.code(json_output, language="json")
                    st.download_button(
                        label="Download JSON",
                        data=json_output,
                        file_name="analysis_result.json",
                        mime="application/json"
                    )
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.exception(e)
    
    else:
        st.info("👆 Please upload an image to get started")
        
        # Show example usage
        with st.expander("ℹ️ How to use"):
            st.markdown("""
            1. **Upload an image** using the file uploader above
            2. **Adjust settings** in the sidebar if needed
            3. **View results** including:
               - Annotated image with bounding boxes
               - Summary statistics
               - Detailed attributes for each detection
            4. **Download JSON** output if needed
            
            **Supported formats**: JPG, JPEG, PNG, BMP
            """)


if __name__ == "__main__":
    main()

