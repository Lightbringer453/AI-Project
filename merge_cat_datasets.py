"""
Merge cat breed datasets from different sources
"""

import shutil
from pathlib import Path
from tqdm import tqdm

def main():
    print("=" * 70)
    print("CAT BREED DATASETS MERGER")
    print("=" * 70)
    print()
    
    dataset1 = Path("cat_breeds_dataset")
    dataset2 = Path("oxford_pets_organized")
    
    output_dir = Path("cat_breeds_merged")
    output_dir.mkdir(exist_ok=True)
    
    if not dataset1.exists():
        print(f"❌ Error: {dataset1} not found!")
        print("   Run: python download_cat_breed_dataset.py")
        return
    
    if not dataset2.exists():
        print(f"❌ Error: {dataset2} not found!")
        print("   Run: python download_oxford_pets.py")
        return
    
    print("Source datasets found:")
    print(f"   1. {dataset1}")
    print(f"   2. {dataset2}")
    print()
    
    breed_mapping = {
        'abyssinian': 'abyssinian',
        'american shorthair': 'american shorthair',
        'bengal': 'bengal',
        'birman': 'birman',
        'bombay': 'bombay',
        'british shorthair': 'british shorthair',
        'egyptian mau': 'egyptian mau',
        'maine coon': 'maine coon',
        'persian': 'persian',
        'ragdoll': 'ragdoll',
        'russian blue': 'russian blue',
        'siamese': 'siamese',
        'siberian': 'siberian',
        'sphynx': 'sphynx',
    }
    
    print("Merging datasets...")
    print()
    
    breed_counts = {}
    total_images = 0
    
    print("Processing li-yan dataset...")
    for breed_dir in tqdm(list(dataset1.iterdir()), desc="   Dataset 1"):
        if not breed_dir.is_dir():
            continue
        
        breed_name = breed_dir.name.lower()
        if breed_name not in breed_mapping:
            continue
        
        normalized_breed = breed_mapping[breed_name]
        target_dir = output_dir / normalized_breed
        target_dir.mkdir(exist_ok=True)
        
        image_files = list(breed_dir.glob("*.jpg")) + list(breed_dir.glob("*.jpeg")) + list(breed_dir.glob("*.png"))
        for img_file in image_files:
            dest = target_dir / f"li_yan_{img_file.name}"
            if not dest.exists():
                shutil.copy2(img_file, dest)
                breed_counts[normalized_breed] = breed_counts.get(normalized_breed, 0) + 1
                total_images += 1
    
    print(f"   ✓ Processed {len(breed_counts)} breeds from li-yan dataset\n")
    
    print("Processing Oxford-IIIT dataset...")
    oxford_count = 0
    for breed_dir in tqdm(list(dataset2.iterdir()), desc="   Dataset 2"):
        if not breed_dir.is_dir():
            continue
        
        breed_name = breed_dir.name.lower()
        if breed_name not in breed_mapping:
            continue
        
        normalized_breed = breed_mapping[breed_name]
        target_dir = output_dir / normalized_breed
        target_dir.mkdir(exist_ok=True)
        
        image_files = list(breed_dir.glob("*.jpg")) + list(breed_dir.glob("*.jpeg")) + list(breed_dir.glob("*.png"))
        for img_file in image_files:
            dest = target_dir / f"oxford_{img_file.name}"
            if not dest.exists():
                shutil.copy2(img_file, dest)
                breed_counts[normalized_breed] = breed_counts.get(normalized_breed, 0) + 1
                total_images += 1
                oxford_count += 1
    
    print(f"   ✓ Added {oxford_count} images from Oxford dataset\n")
    
    print("=" * 70)
    print("MERGED DATASET SUMMARY")
    print("=" * 70)
    print(f"\n✓ Merged dataset saved to: {output_dir}")
    print(f"\nTotal breeds: {len(breed_counts)}")
    print("\nImages per breed:")
    
    for breed, count in sorted(breed_counts.items()):
        print(f"   {breed:<25}: {count:>3} images")
    
    print(f"\n   {'TOTAL':<25}: {total_images:>3} images")
    
    print("\n" + "=" * 70)
    print("NEXT STEP: RETRAIN BREED CLASSIFIER")
    print("=" * 70)
    print("\nTrain with merged dataset:")
    print(f"   python train_breed_classifier.py --dataset_path {output_dir} --species cat --samples_per_breed 100")
    print("\n💡 With more data (150-200 samples):")
    print(f"   python train_breed_classifier.py --dataset_path {output_dir} --species cat --samples_per_breed 150")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
