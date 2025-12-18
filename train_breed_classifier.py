"""
Script to build breed reference vectors from a dataset.

Usage:
    python train_breed_classifier.py --dataset_path path/to/dataset --species cat
    python train_breed_classifier.py --dataset_path path/to/dataset --species dog

Dataset structure should be:
    dataset_path/
        breed1/
            image1.jpg
            image2.jpg
            ...
        breed2/
            image1.jpg
            ...
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from models.animal_analysis import AnimalAnalyzer


def main():
    parser = argparse.ArgumentParser(
        description="Build breed reference vectors from dataset"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to dataset directory (should contain breed subdirectories)"
    )
    parser.add_argument(
        "--species",
        type=str,
        required=True,
        choices=["cat", "dog", "bird", "horse", "cow", "sheep"],
        help="Animal species"
    )
    parser.add_argument(
        "--samples_per_breed",
        type=int,
        default=10,
        help="Number of samples to use per breed (default: 10)"
    )
    
    args = parser.parse_args()
    
    print("Initializing AnimalAnalyzer...")
    analyzer = AnimalAnalyzer()
    
    print(f"\nBuilding breed vectors for {args.species}...")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Samples per breed: {args.samples_per_breed}\n")
    
    breed_vectors = analyzer.build_breed_vectors_from_dataset(
        dataset_path=args.dataset_path,
        species=args.species,
        samples_per_breed=args.samples_per_breed
    )
    
    if breed_vectors:
        print(f"\n✓ Successfully built {len(breed_vectors)} breed vectors!")
        print(f"Breed vectors saved to: {analyzer.breed_vectors_path}")
        print("\nBreed vectors are now ready to use for breed classification.")
    else:
        print("\n✗ Failed to build breed vectors. Check dataset path and structure.")


if __name__ == "__main__":
    main()

