"""
Hayvan türlerini (species) tanımak için eğitim scripti.

Animals-10 dataset'i ile kullanım:
    python train_species_classifier.py --dataset_path animal_dataset

Bu script:
1. Her tür için özellik vektörleri oluşturur
2. ResNet18 kullanarak derin öğrenme özellikleri çıkarır  
3. Vektörleri kaydeder (models/species_vectors.pkl)
"""

import argparse
from pathlib import Path
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pickle
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


class SpeciesFeatureExtractor:
    """Hayvan türleri için özellik çıkarıcı."""
    
    def __init__(self):
        """Initialize feature extractor."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load pre-trained ResNet18
        print("Loading ResNet18 model...")
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(model.children())[:-1])
        self.feature_extractor.eval()
        self.feature_extractor.to(self.device)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print("✓ Feature extractor ready")
    
    def extract_features(self, image_path):
        """Extract deep features from image."""
        try:
            # Load image
            img = cv2.imread(str(image_path))
            if img is None:
                return None
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(img_rgb)
            
            # Preprocess
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.feature_extractor(input_tensor)
                features = features.squeeze().cpu().numpy()
                features = features.flatten()
                # Normalize
                features = features / (np.linalg.norm(features) + 1e-8)
            
            return features
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None


def build_species_vectors(dataset_path, samples_per_species=100, output_file="models/species_vectors.pkl"):
    """
    Build species reference vectors from dataset.
    
    Args:
        dataset_path: Path to dataset with species subdirectories
        samples_per_species: Number of samples per species
        output_file: Output file path
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        return False
    
    # Initialize feature extractor
    extractor = SpeciesFeatureExtractor()
    
    # Get species directories
    species_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    if len(species_dirs) == 0:
        print(f"Error: No species directories found in {dataset_path}")
        return False
    
    print(f"\nFound {len(species_dirs)} species:")
    for d in species_dirs:
        print(f"  - {d.name}")
    
    species_vectors = {}
    
    print("\n" + "=" * 70)
    print("EXTRACTING FEATURES")
    print("=" * 70)
    
    for species_dir in species_dirs:
        species_name = species_dir.name
        print(f"\nProcessing: {species_name}")
        
        # Get image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(list(species_dir.glob(ext)))
            image_files.extend(list(species_dir.glob(ext.upper())))
        
        if len(image_files) == 0:
            print(f"  Warning: No images found in {species_name}")
            continue
        
        # Sample images
        if len(image_files) > samples_per_species:
            import random
            image_files = random.sample(image_files, samples_per_species)
        
        # Extract features
        feature_list = []
        for img_path in tqdm(image_files, desc=f"  {species_name}", ncols=70):
            features = extractor.extract_features(img_path)
            if features is not None:
                feature_list.append(features)
        
        if len(feature_list) > 0:
            # Average features to get species reference vector
            species_vector = np.mean(feature_list, axis=0)
            # Normalize
            species_vector = species_vector / (np.linalg.norm(species_vector) + 1e-8)
            species_vectors[species_name] = species_vector
            print(f"  ✓ {len(feature_list)} samples processed")
        else:
            print(f"  ✗ No valid features extracted")
    
    # Save vectors
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(species_vectors, f)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n✓ Built {len(species_vectors)} species vectors")
    print(f"✓ Saved to: {output_path.absolute()}")
    print("\nSpecies:")
    for species in sorted(species_vectors.keys()):
        print(f"  - {species}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Build species reference vectors from Animals-10 dataset"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="animal_dataset",
        help="Path to dataset directory (default: animal_dataset)"
    )
    parser.add_argument(
        "--samples_per_species",
        type=int,
        default=100,
        help="Number of samples per species (default: 100)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="models/species_vectors.pkl",
        help="Output file path (default: models/species_vectors.pkl)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("SPECIES CLASSIFIER TRAINING")
    print("=" * 70)
    print(f"\nDataset: {args.dataset_path}")
    print(f"Samples per species: {args.samples_per_species}")
    print(f"Output: {args.output_file}")
    
    success = build_species_vectors(
        dataset_path=args.dataset_path,
        samples_per_species=args.samples_per_species,
        output_file=args.output_file
    )
    
    if success:
        print("\n" + "=" * 70)
        print("NEXT STEPS")
        print("=" * 70)
        print("\n1. Test the species classifier:")
        print("   python test_simple.py your_image.jpg")
        print("\n2. Run web interface:")
        print("   streamlit run app/interface.py")
        print("\n" + "=" * 70)
    else:
        print("\n✗ Training failed")


if __name__ == "__main__":
    main()
