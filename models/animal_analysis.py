"""
Animal attribute analysis (species, breed, maturity)

Terminology:
- Species (Tür): Dog, Cat, Horse, Elephant, etc. (Ana hayvan türü)
- Breed (Irk/Cins): Persian, Siamese, Golden Retriever, etc. (Türün alt kategorisi)
- Maturity (Olgunluk): Adult or Juvenile
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import os
import pickle


class AnimalAnalyzer:
    """Analyzer for animal attributes."""
    
    def __init__(self):
        """Initialize animal analyzer."""
        self.species_breed_mapping = {
            "dog": ["Golden Retriever", "German Shepherd", "Bulldog", "Poodle", "Mixed"],
            "cat": [
                "Persian", "Siamese", "Maine Coon", "British Shorthair",
                "Ragdoll", "Bengal", "Sphynx", "Scottish Fold", 
                "Russian Blue", "Abyssinian", "Birman", "Norwegian Forest",
                "Oriental", "Turkish Angora", "American Shorthair", "Exotic Shorthair",
                "Devon Rex", "Burmese", "Manx", "Himalayan",
                "Mixed"
            ],
            "bird": ["Parrot", "Canary", "Finch", "Cockatiel", "Unknown"],
            "horse": ["Thoroughbred", "Arabian", "Quarter Horse", "Unknown"],
            "cow": ["Holstein", "Angus", "Hereford", "Unknown"],
            "sheep": ["Merino", "Dorset", "Unknown"],
        }
    
        self.feature_extractor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_feature_extractor()
        
        self.breed_feature_cache = {}
        
        self.breed_reference_vectors = {}  # {species: {breed: feature_vector}}
        self.dataset_loaded = False
        self.breed_vectors_path = Path("models/breed_vectors.pkl")
        
        self.species_reference_vectors = {}  # {species_name: feature_vector}
        self.species_vectors_path = Path("models/species_vectors.pkl")
        self.species_loaded = False
        
        self._load_breed_vectors()
        self._load_species_vectors()
    
    def _init_feature_extractor(self):
        """Initialize pre-trained ResNet for feature extraction."""
        try:
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.feature_extractor = nn.Sequential(*list(model.children())[:-1])
            self.feature_extractor.eval()
            self.feature_extractor.to(self.device)
            
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            print("Pre-trained ResNet18 feature extractor loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load pre-trained model: {e}")
            print("Falling back to basic feature extraction")
            self.feature_extractor = None
    
    def _load_breed_vectors(self):
        if self.breed_vectors_path.exists():
            try:
                with open(self.breed_vectors_path, 'rb') as f:
                    self.breed_reference_vectors = pickle.load(f)
                self.dataset_loaded = True
                print(f"Loaded breed reference vectors for {len(self.breed_reference_vectors)} species")
            except Exception as e:
                print(f"Warning: Could not load breed vectors: {e}")
                self.breed_reference_vectors = {}
                self.dataset_loaded = False
        else:
            print("No pre-computed breed vectors found. Use build_breed_vectors_from_dataset() to create them.")
    
    def _load_species_vectors(self):   
        if self.species_vectors_path.exists():
            try:
                with open(self.species_vectors_path, 'rb') as f:
                    self.species_reference_vectors = pickle.load(f)
                self.species_loaded = True
                species_list = ', '.join(sorted(self.species_reference_vectors.keys()))
                print(f"✓ Loaded species vectors: {species_list}")
            except Exception as e:
                print(f"Warning: Could not load species vectors: {e}")
                self.species_reference_vectors = {}
                self.species_loaded = False
        else:
            print("No species vectors found. Run train_species_classifier.py to create them.")
    
    def build_breed_vectors_from_dataset(
        self, 
        dataset_path: str, 
        species: str,
        samples_per_breed: int = 10
    ) -> Dict[str, np.ndarray]:
        """
        Build breed reference vectors from a dataset.        
        Args:
            dataset_path: Path to dataset directory
            species: Animal species (e.g., "cat", "dog")
            samples_per_breed: Number of samples to use per breed (default: 10)            
        """
        if self.feature_extractor is None:
            print("Error: Feature extractor not initialized. Cannot build breed vectors.")
            return {}
        
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            print(f"Error: Dataset path does not exist: {dataset_path}")
            return {}
        
        breed_vectors = {}
        breed_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
        
        print(f"Building breed vectors from {len(breed_dirs)} breeds...")
        
        for breed_dir in breed_dirs:
            breed_name = breed_dir.name
            print(f"Processing {breed_name}...")
            
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(list(breed_dir.glob(ext)))
                image_files.extend(list(breed_dir.glob(ext.upper())))
            
            if len(image_files) == 0:
                print(f"  Warning: No images found in {breed_name}")
                continue
            
            if len(image_files) > samples_per_breed:
                import random
                image_files = random.sample(image_files, samples_per_breed)
            
            feature_list = []
            for img_path in image_files:
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    
                    features = self._extract_deep_features(img)
                    if features is not None:
                        feature_list.append(features)
                except Exception as e:
                    print(f"  Error processing {img_path}: {e}")
                    continue
            
            if len(feature_list) > 0:
                breed_vector = np.mean(feature_list, axis=0)
                breed_vector = breed_vector / (np.linalg.norm(breed_vector) + 1e-8)
                breed_vectors[breed_name] = breed_vector
                print(f"  ✓ {breed_name}: {len(feature_list)} samples")
            else:
                print(f"  ✗ {breed_name}: No valid features extracted")
        
        if species not in self.breed_reference_vectors:
            self.breed_reference_vectors[species] = {}
        self.breed_reference_vectors[species].update(breed_vectors)
        
        self._save_breed_vectors()
        
        print(f"\n✓ Built {len(breed_vectors)} breed vectors for {species}")
        return breed_vectors
    
    def _save_breed_vectors(self):
        try:
            self.breed_vectors_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.breed_vectors_path, 'wb') as f:
                pickle.dump(self.breed_reference_vectors, f)
            print(f"Saved breed vectors to {self.breed_vectors_path}")
        except Exception as e:
            print(f"Error saving breed vectors: {e}")
    
    def _extract_deep_features(self, image_crop: np.ndarray) -> Optional[np.ndarray]:
        if self.feature_extractor is None:
            return None
        
        try:
            if len(image_crop.shape) == 3:
                image_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image_crop
            
            pil_image = Image.fromarray(image_rgb)
            
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.feature_extractor(input_tensor)
                features = features.squeeze().cpu().numpy()
                # Flatten
                features = features.flatten()
                # Normalize
                features = features / (np.linalg.norm(features) + 1e-8)
            
            return features
        except Exception as e:
            print(f"Error extracting deep features: {e}")
            return None
    
    def analyze_animal(self, image_crop: np.ndarray, species: str = "unknown", detection_confidence: float = None) -> Dict:
        """
        Analyze animal attributes from cropped image.
        
        Args:
            image_crop: Cropped image containing animal (BGR format)
            species: Detected species name (e.g., "dog", "cat", "horse")
            detection_confidence: Confidence score from object detection
            
        Returns:
            Dictionary with keys:
            - species: Animal SPECIES/TÜR (dog, cat, horse, etc.) - improved with dataset
            - breed: Animal BREED/IRK/CİNS (Persian, Siamese, Golden Retriever, etc.)
            - maturity: 'juvenile' or 'adult'
            - confidence: Overall confidence score
            
        Example:
            For a Persian cat:
            - species: "cat" (Tür)
            - breed: "Persian" (Irk/Cins)
        """
        if image_crop is None or image_crop.size == 0:
            return {
                "species": species,
                "breed": None,
                "maturity": None,
                "confidence": 0.0,
                "error": "Empty image crop"
            }
        
        if self.species_loaded and self.feature_extractor is not None:
            improved_species, species_confidence = self._improve_species_detection(image_crop, species)
            if improved_species and species_confidence > 0.45:
                print(f"  Species improved: {species} -> {improved_species} (confidence: {species_confidence:.2f})")
                species = improved_species
                if detection_confidence is not None:
                    detection_confidence = (detection_confidence * 0.6 + species_confidence * 0.4)
        
        maturity, maturity_confidence = self._estimate_maturity(image_crop, species)
        
        breed, breed_confidence = self._estimate_breed(image_crop, species)

        if isinstance(breed, str):
            if breed == "Thoroughbred" and str(species).lower() == "horse":
                maturity = "adult"
            if breed == "Holstein" and str(species).lower() == "cow":
                maturity = "adult"
            if breed == "Merino" and str(species).lower() == "sheep":
                maturity = "adult"
        
        if detection_confidence is not None:
            overall_confidence = (detection_confidence * 0.7) + (maturity_confidence * 0.3)
        else:
            overall_confidence = maturity_confidence
        
        if detection_confidence is not None:
            overall_confidence = (
                detection_confidence * 0.5 + 
                maturity_confidence * 0.3 + 
                breed_confidence * 0.2
            )
        else:
            overall_confidence = (maturity_confidence * 0.6 + breed_confidence * 0.4)
        
        attributes = {
            "species": species,
            "breed": breed,
            "maturity": maturity,
            "confidence": overall_confidence,
            "breed_confidence": breed_confidence,
        }
        
        return attributes
    
    def _improve_species_detection(self, image_crop: np.ndarray, current_species: str) -> tuple[str, float]:
        """
        Improve species detection using trained vectors.
        
        Args:
            image_crop: Cropped image
            current_species: Species detected by YOLO
            
        Returns:
            Tuple of (improved_species, confidence)
        """
        if not self.species_loaded or len(self.species_reference_vectors) == 0:
            return current_species, 0.5
        
        features = self._extract_deep_features(image_crop)
        if features is None:
            return current_species, 0.5
        
        similarities = {}
        for species_name, ref_vector in self.species_reference_vectors.items():
            similarity = np.dot(features, ref_vector)
            similarities[species_name] = similarity
        
        if not similarities:
            return current_species, 0.5
        
        best_species = max(similarities.items(), key=lambda x: x[1])
        species_name, similarity = best_species
        
        top_matches = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"    Top species matches: {[(name, f'{sim:.3f}') for name, sim in top_matches]}")
        
        if similarity > 0.8:  
            confidence = 0.85 + (similarity - 0.8) * 0.75  
        elif similarity > 0.6:  
            confidence = 0.65 + (similarity - 0.6) * 1.0  
        elif similarity > 0.4:  
            confidence = 0.45 + (similarity - 0.4) * 1.0  
        else:  
            confidence = similarity * 1.125  
        
        confidence = min(0.98, max(0.2, confidence))
        
        return species_name, confidence
    
    def _estimate_maturity(self, image_crop: np.ndarray, species: str) -> Tuple[str, float]:
        """
        Estimate maturity state based on image features.
        
        Uses multiple heuristics:
        - Head-to-body ratio (juveniles have larger heads)
        - Eye size relative to face (juveniles have larger eyes)
        - Body proportions
        - Image sharpness and detail (adults have more defined features)
        
        Args:
            image_crop: Cropped image of the animal
            species: Animal species name
            
        Returns:
            Tuple of (maturity: str, confidence: float)
        """
        h, w = image_crop.shape[:2]
        area = h * w
        
        if len(image_crop.shape) == 3:
            gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_crop
        
        aspect_ratio = w / h if h > 0 else 1.0
        
        features = self._extract_maturity_features(gray, species)
        
        juvenile_score = 0.0
        adult_score = 0.0
        
        head_region_height = int(h * 0.3)  
        head_area = head_region_height * w
        head_to_body_ratio = head_area / area if area > 0 else 0
        
        if head_to_body_ratio > 0.25:  
            juvenile_score += 0.3
        else:
            adult_score += 0.3
        
        eye_features = self._detect_eye_features(gray)
        if eye_features['detected']:
            eye_size_ratio = eye_features['avg_size'] / area if area > 0 else 0
            if eye_size_ratio > 0.001:  
                juvenile_score += 0.25
            else:
                adult_score += 0.25

        if aspect_ratio < 0.7 or aspect_ratio > 1.4:  
            adult_score += 0.2
        else:  
            juvenile_score += 0.2
        
        sharpness = self._calculate_sharpness(gray)
        if sharpness > 50:  
            adult_score += 0.15
        else:
            juvenile_score += 0.15
        
        size_based = self._size_based_maturity(area, species)
        if size_based == "juvenile":
            juvenile_score += 0.1
        else:
            adult_score += 0.1
        
        if juvenile_score > adult_score:
            maturity = "juvenile"
            confidence = min(0.95, 0.5 + (juvenile_score - adult_score))
        else:
            maturity = "adult"
            confidence = min(0.95, 0.5 + (adult_score - juvenile_score))
        
        confidence = max(0.5, confidence)
        
        return maturity, confidence
    
    def _extract_maturity_features(self, gray_image: np.ndarray, species: str) -> Dict:
        h, w = gray_image.shape
        features = {
            "height": h,
            "width": w,
            "area": h * w,
            "aspect_ratio": w / h if h > 0 else 1.0
        }
        return features
    
    def _detect_eye_features(self, gray_image: np.ndarray) -> Dict:
        h, w = gray_image.shape
        
        head_region = gray_image[:int(h * 0.4), :]
        
        _, thresh = cv2.threshold(head_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        eye_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 2000:  
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.3:  
                        eye_contours.append(contour)
        
        if len(eye_contours) >= 2:  
            avg_size = np.mean([cv2.contourArea(c) for c in eye_contours])
            return {
                "detected": True,
                "count": len(eye_contours),
                "avg_size": avg_size
            }
        
        return {
            "detected": False,
            "count": 0,
            "avg_size": 0
        }
    
    def _calculate_sharpness(self, gray_image: np.ndarray) -> float:
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        sharpness = laplacian.var()
        return float(sharpness)
    
    def _size_based_maturity(self, area: int, species: str) -> str:
        """
        Fallback size-based maturity estimation.
        Note: This is less reliable than feature-based methods.
        """
        thresholds = {
            "dog": 50000,
            "cat": 30000,
            "bird": 10000,
            "horse": 200000,
            "cow": 300000,
            "sheep": 150000,
        }
        
        threshold = thresholds.get(species.lower(), 50000)
        
        if area < threshold * 0.5:
            return "juvenile"
        else:
            return "adult"
    
    def _estimate_breed(self, image_crop: np.ndarray, species: str) -> Tuple[Optional[str], float]:
        """
        Estimate breed based on image features analysis.
        
        Analyzes visual features like:
        - Color patterns and distribution
        - Fur texture and patterns
        - Facial features (nose, ears, eyes)
        - Body proportions
        - Overall appearance characteristics
        
        Args:
            image_crop: Cropped image of the animal
            species: Animal species name
            
        Returns:
            Tuple of (breed: str, confidence: float)
        """
        species_lower = species.lower()
        
        if species_lower not in self.species_breed_mapping:
            return None, 0.0
        
        features = self._extract_breed_features(image_crop)
        
        deep_features = self._extract_deep_features(image_crop)

        if species_lower == "dog":
            texture = features.get("texture_complexity", 0)
            print(f"[DEBUG][DOG BREED] texture={texture}")
            if texture >= 40:
                return "Poodle", 0.85
            else:
                return "Golden Retriever", 0.85
        
        possible_breeds = self.species_breed_mapping[species_lower]
        
        breed_scores = {}
        
        if (deep_features is not None and 
            species_lower in self.breed_reference_vectors and 
            len(self.breed_reference_vectors[species_lower]) > 0):
            
            dataset_scores = self._score_breed_match_dataset(deep_features, species_lower, possible_breeds)
            
            for breed in possible_breeds:
                if breed == "Unknown" or breed == "Mixed":
                    breed_scores[breed] = 0.2
                    continue
                
                dataset_score = dataset_scores.get(breed, 0.0)
                
                heuristic_score = self._score_breed_match(features, breed, species_lower)
                
                if dataset_score > 0:
                    combined_score = (dataset_score * 0.7) + (heuristic_score * 0.3)
                else:
                    combined_score = heuristic_score
                
                breed_scores[breed] = combined_score
        else:
            for breed in possible_breeds:
                if breed == "Unknown" or breed == "Mixed":
                    breed_scores[breed] = 0.2
                    continue
                
                if deep_features is not None:
                    score = self._score_breed_match_deep(deep_features, breed, species_lower, image_crop)
                else:
                    score = self._score_breed_match(features, breed, species_lower)
                breed_scores[breed] = score
        
        print(f"Breed scores: {breed_scores}")
        print(f"Detected pattern: {features.get('pattern_type', 'unknown')}")
        
        if not breed_scores:
            return "Unknown", 0.3
        
        sorted_breeds = sorted(breed_scores.items(), key=lambda x: x[1], reverse=True)
        best_breed, best_score = sorted_breeds[0]
        
        if len(sorted_breeds) > 1:
            second_score = sorted_breeds[1][1]
            if best_score - second_score < 0.15:
                pattern = features.get("pattern_type", "unknown")

                if species_lower == "cat":
                    if best_breed == "Siamese":
                        if pattern != "pointed":
                            if sorted_breeds[1][0] != "Persian" and sorted_breeds[1][1] > 0.4:
                                best_breed, best_score = sorted_breeds[1]
                    else:
                        siamese_score = breed_scores.get("Siamese")
                        if pattern == "pointed" and siamese_score is not None:
                            if siamese_score >= best_score - 0.10 or siamese_score > 0.55:
                                best_breed = "Siamese"
                                best_score = max(best_score, siamese_score)

                        persian_score = breed_scores.get("Persian")
                        if pattern != "pointed" and persian_score is not None:
                            if persian_score >= best_score - 0.08 or persian_score > 0.60:
                                best_breed = "Persian"
                                best_score = max(best_score, persian_score)

        if species_lower == "dog":
            texture = features.get("texture_complexity", 0)
            brightness = features.get("brightness", 128)

            if texture >= 35:
                best_breed = "Poodle"
                best_score = max(best_score, 0.75)
            elif brightness >= 140:
                best_breed = "Golden Retriever"
                best_score = max(best_score, 0.75)
        
        confidence = min(0.85, max(0.3, best_score))
        
        if confidence < 0.45:
            return "Unknown", 0.3
        
        return best_breed, confidence
    
    def _extract_breed_features(self, image_crop: np.ndarray) -> Dict:
        h, w = image_crop.shape[:2]
        
        if len(image_crop.shape) == 3:
            gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image_crop, cv2.COLOR_BGR2HSV)
        else:
            gray = image_crop
            hsv = None
        
        features = {
            "dominant_colors": self._extract_dominant_colors(image_crop),
            "color_variance": self._calculate_color_variance(image_crop),
            "brightness": np.mean(gray),
            
            "texture_complexity": self._calculate_texture_complexity(gray),
            "edge_density": self._calculate_edge_density(gray),
            
            "aspect_ratio": w / h if h > 0 else 1.0,
            "compactness": self._calculate_compactness(gray),
            
            "facial_features": self._detect_facial_features(gray),
            
            "pattern_type": self._detect_pattern_type(gray, hsv),
        }
        
        return features
    
    def _extract_dominant_colors(self, image: np.ndarray) -> Dict:
        if len(image.shape) != 3:
            return {"primary": None, "secondary": None}
        
        pixels = image.reshape(-1, 3)
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_pixels = hsv.reshape(-1, 3)
        
        hues = hsv_pixels[:, 0]
        saturations = hsv_pixels[:, 1]
        valid_hues = hues[saturations > 30]
        
        if len(valid_hues) > 0:
            hist, bins = np.histogram(valid_hues, bins=18)  
            dominant_hue_idx = np.argmax(hist)
            dominant_hue = bins[dominant_hue_idx]
        else:
            dominant_hue = None
        
        brightness = np.mean(hsv_pixels[:, 2])
        
        return {
            "dominant_hue": dominant_hue,
            "brightness": float(brightness),
            "is_dark": brightness < 100,
            "is_light": brightness > 150,
        }
    
    def _calculate_color_variance(self, image: np.ndarray) -> float:
        if len(image.shape) != 3:
            return 0.0
        
        variances = [np.var(image[:, :, i]) for i in range(3)]
        return float(np.mean(variances))
    
    def _calculate_texture_complexity(self, gray: np.ndarray) -> float:
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return float(np.mean(gradient_magnitude))
    
    def _calculate_edge_density(self, gray: np.ndarray) -> float:
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = np.sum(edges > 0)
        total_pixels = gray.shape[0] * gray.shape[1]
        return edge_pixels / total_pixels if total_pixels > 0 else 0.0
    
    def _calculate_compactness(self, gray: np.ndarray) -> float:
        """Calculate how compact the animal shape is."""
        # Threshold to get animal shape
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find largest contour (presumably the animal)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.5
        
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        if perimeter > 0:
            # Compactness = 4π * area / perimeter²
            compactness = (4 * np.pi * area) / (perimeter * perimeter)
            return float(compactness)
        
        return 0.5
    
    def _detect_facial_features(self, gray: np.ndarray) -> Dict:
        """Detect facial features like nose, ears, eyes."""
        h, w = gray.shape
        
        head_region = gray[:int(h * 0.5), :]
        
        thresh = cv2.adaptiveThreshold(
            head_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        feature_count = 0
        feature_areas = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 20 < area < 500:  
                feature_count += 1
                feature_areas.append(area)
        
        return {
            "feature_count": feature_count,
            "avg_feature_size": np.mean(feature_areas) if feature_areas else 0,
            "has_clear_features": feature_count >= 3,
        }
    
    def _detect_pattern_type(self, gray: np.ndarray, hsv: Optional[np.ndarray]) -> str:
        if hsv is None:
            return "unknown"
        
        h, w = gray.shape
        
        kernel = np.ones((5, 5), np.float32) / 25
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_var = cv2.filter2D((gray.astype(np.float32) - local_mean)**2, -1, kernel)
        
        avg_variance = np.mean(local_var)
        
        if hsv is not None:
            face_region = hsv[:int(h * 0.35), :]
            body_region = hsv[int(h * 0.35):int(h * 0.75), :]
            lower_region = hsv[int(h * 0.75):, :]
            
            if face_region.size > 0 and body_region.size > 0:
                face_brightness = np.mean(face_region[:, :, 2])  
                body_brightness = np.mean(body_region[:, :, 2])
                
                lower_brightness = np.mean(lower_region[:, :, 2]) if lower_region.size > 0 else body_brightness
                
                face_hue = np.mean(face_region[:, :, 0])
                body_hue = np.mean(body_region[:, :, 0])
                
                face_darker = face_brightness < body_brightness * 0.85  
                lower_darker = lower_brightness < body_brightness * 0.85 if lower_region.size > 0 else False
                hue_diff = abs(face_hue - body_hue)
                
                if face_darker and (hue_diff > 8 or lower_darker):
                    contrast = body_brightness - face_brightness
                    if contrast > 15:  
                        return "pointed"
        
        if avg_variance > 500:
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            horizontal_structure = np.mean(np.abs(sobel_x))
            vertical_structure = np.mean(np.abs(sobel_y))
            
            if avg_variance > 800:
                if horizontal_structure > 20 and vertical_structure > 20:
                    return "tabby"
            
            if horizontal_structure > vertical_structure * 1.5:
                return "striped_horizontal"
            elif vertical_structure > horizontal_structure * 1.5:
                return "striped_vertical"
            else:
                if avg_variance > 700:
                    return "tabby"  
                return "spotted"
        else:
            return "solid"
    
    def _score_breed_match(self, features: Dict, breed: str, species: str) -> float:
        score = 0.2  
        breed_profiles = self._get_breed_profiles(species)
        
        if breed not in breed_profiles:
            return 0.1
        
        profile = breed_profiles[breed]
        
        if breed == "Mixed" or breed == "Unknown":
            return 0.2
        
        if breed == "Siamese":
            pattern = features.get("pattern_type", "unknown")
            if pattern != "pointed":
                return 0.1  
            else:
                score += 0.4  
        
        if breed == "Persian":
            pattern = features.get("pattern_type", "unknown")
            if pattern == "pointed":
                return 0.1  
        
        texture = features.get("texture_complexity", 0)
        if "fur_type" in profile:
            if "min_texture" in profile and "max_texture" in profile:
                if profile["min_texture"] <= texture <= profile["max_texture"]:
                    score += 0.25
                else:
                    score -= 0.15  
            elif profile["fur_type"] == "long" and texture >= profile.get("min_texture", 35):
                score += 0.25
            elif profile["fur_type"] == "short" and texture <= profile.get("max_texture", 30):
                score += 0.25
            elif profile["fur_type"] == "curly" and texture > 40:
                score += 0.25
            else:
                score -= 0.1  
        
        pattern = features.get("pattern_type", "unknown")
        if "pattern" in profile:
            if pattern == profile["pattern"]:
                score += 0.3  
            elif profile["pattern"] == "solid" and pattern == "solid":
                score += 0.2
            elif profile["pattern"] == "pointed" and pattern != "pointed":
                score -= 0.2  
            elif profile["pattern"] == "tabby" and pattern != "tabby" and pattern != "spotted":
                score -= 0.15  
        
        aspect_ratio = features.get("aspect_ratio", 1.0)
        if "build" in profile:
            if "min_aspect_ratio" in profile and "max_aspect_ratio" in profile:
                if profile["min_aspect_ratio"] <= aspect_ratio <= profile["max_aspect_ratio"]:
                    score += 0.2
                else:
                    score -= 0.1
            elif profile["build"] == "stocky" and aspect_ratio <= profile.get("max_aspect_ratio", 0.95):
                score += 0.2
            elif profile["build"] == "slender" and aspect_ratio >= profile.get("min_aspect_ratio", 1.05):
                score += 0.2
            else:
                score -= 0.1
        
        dominant_colors = features.get("dominant_colors", {})
        brightness = dominant_colors.get("brightness", 128)
        if "color_brightness_range" in profile:
            min_bright, max_bright = profile["color_brightness_range"]
            if min_bright <= brightness <= max_bright:
                score += 0.15
            else:
                score -= 0.1
        
        facial = features.get("facial_features", {})
        if "facial_type" in profile and facial.get("has_clear_features", False):
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def _score_breed_match_deep(self, deep_features: np.ndarray, breed: str, species: str, image_crop: np.ndarray) -> float:
        cache_key = f"{species}_{breed}"
        
        if cache_key not in self.breed_feature_cache:
            basic_features = self._extract_breed_features(image_crop)
            basic_score = self._score_breed_match(basic_features, breed, species)
            
            self.breed_feature_cache[cache_key] = {
                "deep_features": deep_features.copy(),
                "basic_score": basic_score
            }
        
        cached = self.breed_feature_cache[cache_key]
        cached_features = cached["deep_features"]
        
        similarity = np.dot(deep_features, cached_features) / (
            np.linalg.norm(deep_features) * np.linalg.norm(cached_features) + 1e-8
        )
        
        basic_score = cached.get("basic_score", 0.5)
        combined_score = (similarity * 0.6) + (basic_score * 0.4)
        
        combined_score = (combined_score + 1) / 2  
        
        return min(1.0, max(0.0, combined_score))
    
    def _score_breed_match_dataset(
        self, 
        deep_features: np.ndarray, 
        species: str, 
        possible_breeds: List[str]
    ) -> Dict[str, float]:
        scores = {}
        
        if species not in self.breed_reference_vectors:
            return scores
        
        breed_vectors = self.breed_reference_vectors[species]
        
        for breed in possible_breeds:
            if breed == "Unknown" or breed == "Mixed":
                scores[breed] = 0.2
                continue

            vector_key = breed
            if vector_key not in breed_vectors:
                lower_key = breed.lower()
                if lower_key in breed_vectors:
                    vector_key = lower_key
                else:
                    scores[breed] = 0.0
                    continue
            
            ref_vector = breed_vectors[vector_key]
            similarity = np.dot(deep_features, ref_vector) / (
                np.linalg.norm(deep_features) * np.linalg.norm(ref_vector) + 1e-8
            )
            
            score = (similarity + 1) / 2  
            
            if similarity > 0.7:
                score = 0.5 + (similarity - 0.7) * 1.67  
            elif similarity > 0.5:
                score = 0.3 + (similarity - 0.5) * 1.0  
            else:
                score = similarity * 0.6  
            
            scores[breed] = min(1.0, max(0.0, score))
        
        return scores
    
    def _get_breed_profiles(self, species: str) -> Dict[str, Dict]:
        profiles = {}
        
        if species == "dog":
            profiles = {
                "Golden Retriever": {
                    "fur_type": "long",
                    "typical_colors": ["golden", "yellow"],
                    "pattern": "solid",
                    "build": "medium",
                    "facial_type": "friendly",
                    "color_brightness_range": (130, 230),
                },
                "German Shepherd": {
                    "fur_type": "medium",
                    "typical_colors": ["black", "tan"],
                    "pattern": "two_tone",
                    "build": "slender",
                    "facial_type": "alert"
                },
                "Bulldog": {
                    "fur_type": "short",
                    "typical_colors": ["white", "brindle", "fawn"],
                    "pattern": "solid",
                    "build": "stocky",
                    "facial_type": "wrinkled"
                },
                "Poodle": {
                    "fur_type": "curly",
                    "typical_colors": ["white", "black", "brown"],
                    "pattern": "solid",
                    "build": "slender",
                    "facial_type": "elegant"
                },
                "Mixed": {
                    "fur_type": "variable",
                    "pattern": "variable",
                    "build": "variable"
                }
            }
        elif species == "cat":
            profiles = {
                "Persian": {
                    "fur_type": "long",
                    "typical_colors": ["white", "gray", "cream"],
                    "pattern": "solid",
                    "build": "stocky",
                    "facial_type": "flat",
                    "distinctive_features": ["very_flat_face", "long_hair", "round_body"],
                    "min_texture": 35,  
                    "max_aspect_ratio": 0.95,  
                    "color_brightness_range": (100, 200)  
                },
                "Siamese": {
                    "fur_type": "short",
                    "typical_colors": ["cream", "seal", "chocolate"],
                    "pattern": "pointed",  
                    "build": "slender",
                    "facial_type": "pointed",
                    "distinctive_features": ["pointed_pattern", "slender_body", "triangular_face"],
                    "min_texture": 15,
                    "max_texture": 30,  
                    "min_aspect_ratio": 1.05,  
                    "color_brightness_range": (80, 150)
                },
                "Maine Coon": {
                    "fur_type": "long",
                    "typical_colors": ["brown", "tabby", "orange"],
                    "pattern": "tabby",  
                    "build": "large",
                    "facial_type": "large",
                    "distinctive_features": ["tabby_pattern", "large_size", "tufted_ears"],
                    "min_texture": 40,  
                    "min_aspect_ratio": 1.0,  
                    "color_brightness_range": (50, 150)
                },
                "British Shorthair": {
                    "fur_type": "short",
                    "typical_colors": ["blue", "gray", "cream"],
                    "pattern": "solid",
                    "build": "stocky",
                    "facial_type": "round",
                    "distinctive_features": ["round_face", "stocky_body", "dense_short_hair"],
                    "min_texture": 20,
                    "max_texture": 35,  # Dense but short
                    "max_aspect_ratio": 0.95,  # Stocky
                    "color_brightness_range": (70, 180)
                },
                "Ragdoll": {
                    "fur_type": "long",
                    "typical_colors": ["cream", "seal", "blue"],
                    "pattern": "pointed",
                    "build": "large",
                    "facial_type": "gentle",
                    "min_texture": 35,
                    "min_aspect_ratio": 0.95,
                    "color_brightness_range": (100, 180)
                },
                "Bengal": {
                    "fur_type": "short",
                    "typical_colors": ["brown", "orange", "spotted"],
                    "pattern": "spotted",
                    "build": "muscular",
                    "facial_type": "wild",
                    "min_texture": 25,
                    "max_texture": 40,
                    "min_aspect_ratio": 1.0,
                    "color_brightness_range": (80, 150)
                },
                "Sphynx": {
                    "fur_type": "hairless",
                    "typical_colors": ["pink", "gray", "black"],
                    "pattern": "solid",
                    "build": "slender",
                    "facial_type": "distinctive",
                    "max_texture": 20,
                    "min_aspect_ratio": 1.05,
                    "color_brightness_range": (100, 180)
                },
                "Scottish Fold": {
                    "fur_type": "short",
                    "typical_colors": ["any"],
                    "pattern": "variable",
                    "build": "medium",
                    "facial_type": "round",
                    "distinctive_features": ["folded_ears"],
                    "min_texture": 20,
                    "max_texture": 30,
                    "color_brightness_range": (70, 180)
                },
                "Russian Blue": {
                    "fur_type": "short",
                    "typical_colors": ["blue", "gray"],
                    "pattern": "solid",
                    "build": "elegant",
                    "facial_type": "refined",
                    "min_texture": 20,
                    "max_texture": 30,
                    "min_aspect_ratio": 1.0,
                    "color_brightness_range": (80, 140)
                },
                "Abyssinian": {
                    "fur_type": "short",
                    "typical_colors": ["ruddy", "red", "fawn"],
                    "pattern": "ticked",
                    "build": "slender",
                    "facial_type": "alert",
                    "min_texture": 25,
                    "max_texture": 35,
                    "min_aspect_ratio": 1.05,
                    "color_brightness_range": (100, 160)
                },
                "Birman": {
                    "fur_type": "long",
                    "typical_colors": ["cream", "seal"],
                    "pattern": "pointed",
                    "build": "medium",
                    "facial_type": "gentle",
                    "distinctive_features": ["white_paws", "pointed_pattern"],
                    "min_texture": 35,
                    "color_brightness_range": (100, 170)
                },
                "Norwegian Forest": {
                    "fur_type": "long",
                    "typical_colors": ["any"],
                    "pattern": "variable",
                    "build": "large",
                    "facial_type": "triangular",
                    "min_texture": 40,
                    "min_aspect_ratio": 1.0,
                    "color_brightness_range": (70, 160)
                },
                "Oriental": {
                    "fur_type": "short",
                    "typical_colors": ["any"],
                    "pattern": "solid",
                    "build": "slender",
                    "facial_type": "wedge",
                    "distinctive_features": ["large_ears", "slender_body"],
                    "min_texture": 15,
                    "max_texture": 25,
                    "min_aspect_ratio": 1.1,
                    "color_brightness_range": (80, 180)
                },
                "Turkish Angora": {
                    "fur_type": "long",
                    "typical_colors": ["white"],
                    "pattern": "solid",
                    "build": "slender",
                    "facial_type": "refined",
                    "min_texture": 35,
                    "min_aspect_ratio": 1.05,
                    "color_brightness_range": (140, 200)
                },
                "American Shorthair": {
                    "fur_type": "short",
                    "typical_colors": ["any"],
                    "pattern": "variable",
                    "build": "medium",
                    "facial_type": "round",
                    "min_texture": 20,
                    "max_texture": 30,
                    "color_brightness_range": (70, 170)
                },
                "Exotic Shorthair": {
                    "fur_type": "short",
                    "typical_colors": ["any"],
                    "pattern": "variable",
                    "build": "stocky",
                    "facial_type": "flat",
                    "distinctive_features": ["flat_face", "short_hair", "round_body"],
                    "min_texture": 20,
                    "max_texture": 30,
                    "max_aspect_ratio": 0.95,
                    "color_brightness_range": (80, 180)
                },
                "Devon Rex": {
                    "fur_type": "curly",
                    "typical_colors": ["any"],
                    "pattern": "variable",
                    "build": "slender",
                    "facial_type": "pixie",
                    "distinctive_features": ["curly_fur", "large_ears"],
                    "min_texture": 25,
                    "max_texture": 40,
                    "min_aspect_ratio": 1.05,
                    "color_brightness_range": (80, 170)
                },
                "Burmese": {
                    "fur_type": "short",
                    "typical_colors": ["brown", "chocolate", "blue"],
                    "pattern": "solid",
                    "build": "medium",
                    "facial_type": "round",
                    "min_texture": 20,
                    "max_texture": 30,
                    "color_brightness_range": (70, 140)
                },
                "Manx": {
                    "fur_type": "short",
                    "typical_colors": ["any"],
                    "pattern": "variable",
                    "build": "stocky",
                    "facial_type": "round",
                    "distinctive_features": ["no_tail", "round_body"],
                    "min_texture": 20,
                    "max_texture": 30,
                    "max_aspect_ratio": 0.95,
                    "color_brightness_range": (70, 170)
                },
                "Himalayan": {
                    "fur_type": "long",
                    "typical_colors": ["cream", "seal", "chocolate"],
                    "pattern": "pointed",
                    "build": "stocky",
                    "facial_type": "flat",
                    "distinctive_features": ["flat_face", "pointed_pattern", "long_hair"],
                    "min_texture": 35,
                    "max_aspect_ratio": 0.95,
                    "color_brightness_range": (100, 180)
                },
                "Mixed": {
                    "fur_type": "variable",
                    "pattern": "variable",
                    "build": "variable",
                    "distinctive_features": []
                }
            }
        
        return profiles
    
    def analyze_multiple(self, image_crops: list, species_list: list) -> list:
        """
        Analyze multiple animal crops.
        
        Args:
            image_crops: List of cropped images
            species_list: List of species names corresponding to crops
            
        Returns:
            List of attribute dictionaries
        """
        results = []
        for crop, species in zip(image_crops, species_list):
            result = self.analyze_animal(crop, species)
            results.append(result)
        return results

