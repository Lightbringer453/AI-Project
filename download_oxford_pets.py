"""
Oxford-IIIT Pet Dataset Downloader
37 cat and dog breeds with ~200 images per breed
"""

import os
import tarfile
import urllib.request
from pathlib import Path
from tqdm import tqdm
import shutil

def download_file(url, destination):
    """Download file with progress bar"""
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=destination) as t:
        urllib.request.urlretrieve(url, filename=destination, reporthook=t.update_to)

def main():
    print("=" * 70)
    print("OXFORD-IIIT PET DATASET DOWNLOADER")
    print("=" * 70)
    print()
    
    # URLs
    images_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
    annotations_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
    
    # Paths
    base_dir = Path("oxford_pets_raw")
    base_dir.mkdir(exist_ok=True)
    
    images_tar = base_dir / "images.tar.gz"
    annotations_tar = base_dir / "annotations.tar.gz"
    
    # Download images
    if not images_tar.exists():
        print("1. Downloading images (~800MB)...")
        download_file(images_url, str(images_tar))
        print("   ✓ Images downloaded\n")
    else:
        print("1. Images already downloaded\n")
    
    # Download annotations
    if not annotations_tar.exists():
        print("2. Downloading annotations...")
        download_file(annotations_url, str(annotations_tar))
        print("   ✓ Annotations downloaded\n")
    else:
        print("2. Annotations already downloaded\n")
    
    # Extract
    print("3. Extracting files...")
    
    if not (base_dir / "images").exists():
        print("   Extracting images...")
        with tarfile.open(images_tar, 'r:gz') as tar:
            tar.extractall(base_dir)
        print("   ✓ Images extracted")
    else:
        print("   Images already extracted")
    
    if not (base_dir / "annotations").exists():
        print("   Extracting annotations...")
        with tarfile.open(annotations_tar, 'r:gz') as tar:
            tar.extractall(base_dir)
        print("   ✓ Annotations extracted")
    else:
        print("   Annotations already extracted")
    
    # Organize by breed
    print("\n4. Organizing images by breed...")
    
    images_dir = base_dir / "images"
    output_dir = Path("oxford_pets_organized")
    output_dir.mkdir(exist_ok=True)
    
    # Get all image files
    image_files = list(images_dir.glob("*.jpg"))
    
    # Cat breeds in Oxford dataset (25 cat breeds)
    cat_breeds = {
        'Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair',
        'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue',
        'Siamese', 'Sphynx'
    }
    
    # Organize files
    breed_counts = {}
    
    for img_file in tqdm(image_files, desc="   Processing images"):
        # Extract breed name from filename
        # Format: BreedName_123.jpg
        filename = img_file.stem
        parts = filename.split('_')
        
        # Find breed name (everything before the last number)
        breed_parts = []
        for part in parts:
            if part.isdigit():
                break
            breed_parts.append(part)
        
        if not breed_parts:
            continue
            
        breed_name = '_'.join(breed_parts)
        
        # Only process cat breeds
        if breed_name not in cat_breeds:
            continue
        
        # Create breed directory
        breed_dir = output_dir / breed_name.lower().replace('_', ' ')
        breed_dir.mkdir(exist_ok=True)
        
        # Copy image
        dest = breed_dir / img_file.name
        if not dest.exists():
            shutil.copy2(img_file, dest)
        
        # Count
        breed_counts[breed_name] = breed_counts.get(breed_name, 0) + 1
    
    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n✓ Dataset organized to: {output_dir}")
    print(f"\nCat breeds found: {len(breed_counts)}")
    print("\nImages per breed:")
    
    total_images = 0
    for breed, count in sorted(breed_counts.items()):
        print(f"   {breed.replace('_', ' '):<25}: {count:>3} images")
        total_images += count
    
    print(f"\n   {'TOTAL':<25}: {total_images:>3} images")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("\n1. Merge with existing cat breeds dataset:")
    print("   python merge_cat_datasets.py")
    print("\n2. Retrain breed classifier with more data:")
    print("   python train_breed_classifier.py --dataset_path cat_breeds_merged --species cat --samples_per_breed 100")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
