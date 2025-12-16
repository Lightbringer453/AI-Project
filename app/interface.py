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
        page_icon="�️",
        layout="wide"
    )

    # One-time splash/loading screen in the center of the page
    if "splash_done" not in st.session_state:
        st.markdown(
            """
            <style>
            .centered-loader-overlay {
                position: fixed;
                inset: 0;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                background: radial-gradient(circle at top, #f9fafb 0, #e5e7eb 40%, #d1d5db 100%);
                z-index: 9999;
                animation: fadeOut 0.3s ease-out 0.8s forwards;
            }
            .centered-loader {
                border: 10px solid #e5e7eb;
                border-top: 10px solid #2563eb;
                border-radius: 50%;
                width: 120px;
                height: 120px;
                animation: spin 1s linear infinite;
                box-shadow: 0 10px 30px rgba(0,0,0,0.15);
                margin-bottom: 24px;
            }
            .centered-loader-text {
                font-size: 24px;
                font-weight: 600;
                color: #374151;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            @keyframes fadeOut {
                to {
                    opacity: 0;
                    visibility: hidden;
                }
            }
            </style>
            <div class="centered-loader-overlay" id="splash-loader">
                <div class="centered-loader"></div>
                <div class="centered-loader-text">Loading...</div>
            </div>
            <script>
            setTimeout(function() {
                var loader = document.getElementById('splash-loader');
                if (loader) {
                    loader.style.display = 'none';
                }
            }, 1100);
            </script>
            """,
            unsafe_allow_html=True,
        )
        # Mark splash as shown
        st.session_state["splash_done"] = True
    
    st.title("Multi-Purpose Image Detection and Analysis System")
    st.markdown("""
    Upload an image to detect humans and animals, then analyze their attributes:
    - **Humans**: Age, Gender, Emotion
    - **Animals**: Species, Breed, Maturity
    """)
    
    # Detector type is fixed to YOLO
    detector_type = "yolo"
    
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
            st.subheader("Original Image")
            st.image(image, channels="BGR", use_container_width=True)
        
        # Process image
        # Create centered loading indicator
        loading_placeholder = st.empty()
        with loading_placeholder.container():
            st.markdown("""
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 300px;">
                <div class="spinner" style="border: 4px solid #f3f3f3; border-top: 4px solid #3498db; border-radius: 50%; width: 50px; height: 50px; animation: spin 1s linear infinite; margin-bottom: 20px;"></div>
                <p style="font-size: 18px; color: #666;">loading...</p>
            </div>
            <style>
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            </style>
            """, unsafe_allow_html=True)
        
        try:
            # Initialize pipeline
            pipeline = ImageAnalysisPipeline(detector_type=detector_type)
            
            # Process image
            result = pipeline.process_image(
                image,
                save_output=False,
                draw_annotations=True
            )
            
            # Clear loading indicator
            loading_placeholder.empty()
            
            # Display annotated image
            with col2:
                st.subheader("Analysis Results")
                if result["annotated_image"] is not None:
                    st.image(result["annotated_image"], channels="BGR", use_container_width=True)
                else:
                    st.info("No detections found")
            
            # Display summary
            st.subheader("Summary")
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
                st.subheader("Detailed Results")
                
                for i, detection in enumerate(result["detections"], 1):
                    with st.expander(f"Detection {i}: {detection['class_type'].title()}", expanded=True):
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

                                # Extra description for Siamese cats
                                if (
                                    isinstance(breed, str)
                                    and breed.lower() == "siamese"
                                    and str(attributes.get("species", "")).lower() == "cat"
                                ):
                                    st.markdown("""
**About Siamese Cats**

The Siamese cat is one of the first distinctly recognised breeds of Asian cat. It derives from the Wichianmat landrace. The Siamese cat is one of several varieties of cats native to Thailand (known as Siam before 1939). The original Siamese became one of the most popular breeds in Europe and North America in the 19th century. Siamese cats have a distinctive colourpoint coat, resulting from a temperature-sensitive type of albinism.

Distinct features like blue almond-shaped eyes, a triangular head shape, large ears, an elongated, slender, and muscular body, and various forms of point colouration characterise the modern-style Siamese. The modern-style Siamese's point-colouration resembles the "old-style" foundation stock. The "old-style" Siamese have a round head and body and have been re-established by multiple registries as the Thai cat. Siamese and Thai cats are selectively bred and pedigreed in multiple cat fancier and breeder organisations. The terms "Siamese" or "Thai" are used for cats from this specific breed, which are by definition all purebred cats with a known and formally registered ancestry. The ancestry registration is the cat's pedigree or "paperwork".
""")

                                # Extra description for Persian cats
                                if (
                                    isinstance(breed, str)
                                    and breed.lower() == "persian"
                                    and str(attributes.get("species", "")).lower() == "cat"
                                ):
                                    st.markdown("""
**About Persian Cats**

The Persian cat, also known as the Persian Longhair or simply Persian, is a long-haired traditional breed of cat characterised by a round face and petite, but not flat and not smashed in, muzzle. The short flat nose was created in the US from in-breeding and causes breathing difficulties in the breed, whereas, the traditional Persian breed has a petite nose which enables them to breathe without difficulties.

The first documented ancestors of Persian cats might have been imported into Italy from Khorasan as early as around 1620, but this has not been proven. Instead, there is stronger evidence for a longhaired cat breed being exported from Afghanistan and Iran/Persia from the 19th century onwards. Persian cats have been widely recognised by the North-West European cat fancy since the 19th century, and after World War II by breeders from North America, Australia and New Zealand. Some cat fancier organisations' breed standards subsume the Himalayan and Exotic Shorthair as variants of this breed, while others generally treat them as separate breeds.
""")

                                # Extra description for Golden Retrievers
                                if (
                                    isinstance(breed, str)
                                    and breed.lower() == "golden retriever"
                                    and str(attributes.get("species", "")).lower() == "dog"
                                ):
                                    st.markdown("""
**About Golden Retrievers**

The Golden Retriever is a Scottish breed of retriever dog of medium-large size. It is characterised by a gentle and affectionate nature and a striking golden coat. It is a working dog, and registration is subject to successful completion of a working trial. It is commonly kept as a companion dog and is among the most frequently registered breeds in several Western countries; some may compete in dog shows or obedience trials, or work as guide dogs.

The Golden Retriever was bred by Sir Dudley Marjoribanks at his Scottish estate Guisachan in the late nineteenth century. He cross-bred Flat-coated Retrievers with Tweed Water Spaniels, with some further infusions of Red Setter, Labrador Retriever and Bloodhound. It was recognised by the Kennel Club in 1913, and during the interwar period spread to many parts of the world.
""")

                                # Extra description for Poodles
                                if (
                                    isinstance(breed, str)
                                    and breed.lower() == "poodle"
                                    and str(attributes.get("species", "")).lower() == "dog"
                                ):
                                    st.markdown("""
**About Poodles**

The Poodle, called the Pudel in German and the Caniche in French, is a breed of water dog. The breed is divided into four varieties based on size, the Standard Poodle, Medium Poodle, Miniature Poodle and Toy Poodle, although the Medium Poodle is not universally recognised. They have a distinctive thick, curly coat that comes in many colours and patterns, with only solid colours recognised by major breed registries. Poodles are active and intelligent, and are particularly able to learn from humans. Poodles tend to live 10–18 years, with smaller varieties tending to live longer than larger ones.

The Poodle likely originated in Germany, although the Fédération Cynologique Internationale (FCI, International Canine Federation) and a minority of cynologists believe it originated in France. Similar dogs date back to at least the 17th century. Larger Poodles were originally used by wildfowl hunters to retrieve game from water, while smaller varieties were once commonly used as circus performers. Poodles were recognised by both the Kennel Club of the United Kingdom and the American Kennel Club (AKC) soon after the clubs' founding. Since the mid-20th century, Poodles have enjoyed enormous popularity as pets and show dogs – Poodles were the AKC's most registered breed from 1960 to 1982, and are now the FCI's third most registered breed. Poodles are also common at dog shows, where they often sport the popularly recognised Continental clip, with face and rear clipped close, and tufts of hair on the hocks and tail tip.
""")

                                # Extra description for Thoroughbred horses
                                if (
                                    isinstance(breed, str)
                                    and breed.lower() == "thoroughbred"
                                    and str(attributes.get("species", "")).lower() == "horse"
                                ):
                                    st.markdown("""
**About Thoroughbred Horses**

The Thoroughbred is a horse breed developed for horse racing. Although the word thoroughbred is sometimes used to refer to any breed of purebred horse, it technically refers only to the Thoroughbred breed. Thoroughbreds are considered "hot-blooded" horses that are known for their agility, speed, and spirit.

The Thoroughbred, as it is known today, was developed in 17th- and 18th-century England, when native mares were crossbred with imported stallions of Arabian, Barb, and Turkoman breeding. All modern Thoroughbreds can trace their pedigrees to three stallions originally imported into England in the 17th and 18th centuries, and to a larger number of foundation mares of mostly English breeding. During the 18th and 19th centuries, the Thoroughbred breed spread throughout the world; they were imported into North America starting in 1730 and into Australia, Europe, Japan and South America during the 19th century. Millions of Thoroughbreds exist today, and around 100,000 foals are registered each year worldwide.
""")

                                # Extra description for Holstein cows
                                if (
                                    isinstance(breed, str)
                                    and breed.lower() == "holstein"
                                    and str(attributes.get("species", "")).lower() == "cow"
                                ):
                                    st.markdown("""
**About Holstein Friesian Cattle**

The Holstein Friesian is an international breed or group of breeds of dairy cattle. It originated in Frisia, stretching from the Dutch province of North Holland to the German state of Schleswig-Holstein. It is the dominant breed in industrial dairy farming worldwide, and is found in more than 160 countries. It is known by many names, among them Holstein, Friesian and Black and White.

With the growth of the New World, a demand for milk developed in North America and South America, and dairy breeders in those regions at first imported their livestock from the Netherlands. However, after about 8,800 Friesians (black pied German cows) had been imported, Europe stopped exporting dairy animals due to disease problems.
""")

                                # Extra description for Merino sheep
                                if (
                                    isinstance(breed, str)
                                    and breed.lower() == "merino"
                                    and str(attributes.get("species", "")).lower() == "sheep"
                                ):
                                    st.markdown("""
**About Merino Sheep**

The Merino is a breed or group of breeds of domestic sheep, characterised by very fine soft wool. It was established in the Iberian Peninsula (modern Spain and Portugal) near the end of the Middle Ages, and was for several centuries kept as a strict Spanish monopoly; exports of the breed were not allowed, and those who tried risked capital punishment. During the eighteenth century, flocks were sent to the courts of a number of European countries, including France (where they developed into the Rambouillet), Hungary, the Netherlands, Prussia, Saxony and Sweden.

The Merino subsequently spread to many parts of the world, including South Africa, Australia, and New Zealand. Numerous recognised breeds, strains and variants have developed from the original type; these include, among others, the American Merino and Delaine Merino in the Americas, the Australian Merino, Booroola Merino and Peppin Merino in Oceania, and the Gentile di Puglia, Merinolandschaf and Rambouillet in Europe.
""")

                                st.write(f"- Maturity: {attributes.get('maturity', 'N/A')}")
                
        except Exception as e:
            # Clear loading indicator on error
            loading_placeholder.empty()
            st.error(f"Error processing image: {str(e)}")
            st.exception(e)
    
    else:
        st.info("Please upload an image to get started")
        
        # Show example usage
        with st.expander("How to use", expanded=True):
            st.markdown("""
            1. **Upload an image** using the file uploader above
            2. **View results** including:
               - Annotated image with bounding boxes
               - Summary statistics
               - Detailed attributes for each detection
            
            **Supported formats**: JPG, JPEG, PNG, BMP
            """)


if __name__ == "__main__":
    main()

