"""
Kedi ırkları dataset'ini Hugging Face'ten indirip organize eden script.

Dataset: cat-by-breed-v1
Kaynak: https://huggingface.co/datasets/li-yan/cat-by-breed-v1

Kullanım:
    python download_cat_breed_dataset.py
"""

from datasets import load_dataset
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import os


def download_cat_breed_dataset(output_dir="cat_breeds_dataset", max_images_per_breed=200):
    """
    Cat breed dataset'ini indir ve organize et.
    
    Args:
        output_dir: Çıktı klasörü
        max_images_per_breed: Her ırk için maksimum görsel sayısı
    """
    print("=" * 70)
    print("CAT BREED DATASET İNDİRME VE ORGANİZASYON")
    print("=" * 70)
    
    print("\n1. Dataset yükleniyor (bu biraz zaman alabilir)...")
    try:
        dataset = load_dataset("li-yan/cat-by-breed-v1", split="train")
        print(f"   ✓ Dataset yüklendi: {len(dataset)} görsel")
    except Exception as e:
        print(f"   ✗ Hata: {e}")
        print("\n   Alternatif yöntem deneniyor...")
        try:
            dataset = load_dataset("li-yan/cat-by-breed-v1")
            dataset = dataset["train"]
            print(f"   ✓ Dataset yüklendi: {len(dataset)} görsel")
        except Exception as e2:
            print(f"   ✗ Hata: {e2}")
            return False
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print(f"\n2. Görseller '{output_dir}' klasörüne kaydediliyor...")
    
    if hasattr(dataset.features["label"], "names"):
        breed_names = dataset.features["label"].names
    else:
        breed_names = []
        for item in dataset:
            breed = item.get("breed", "unknown")
            if breed not in breed_names:
                breed_names.append(breed)
    
    print(f"   Bulunan ırklar: {len(breed_names)}")
    for i, breed in enumerate(breed_names[:10], 1):
        print(f"     {i}. {breed}")
    if len(breed_names) > 10:
        print(f"     ... ve {len(breed_names) - 10} tane daha")
    
    for breed_name in breed_names:
        breed_path = output_path / breed_name
        breed_path.mkdir(exist_ok=True)
    
    breed_counts = {name: 0 for name in breed_names}
    
    print("\n3. Görseller işleniyor...")
    for idx, item in enumerate(tqdm(dataset, desc="   Progress", ncols=70)):
        if "label" in item:
            label = item["label"]
            breed_name = breed_names[label] if isinstance(label, int) else label
        elif "breed" in item:
            breed_name = item["breed"]
        else:
            breed_name = "unknown"
        
        if breed_counts.get(breed_name, 0) >= max_images_per_breed:
            continue
        
        image = item["image"]
        
        image_filename = f"{breed_name}_{breed_counts.get(breed_name, 0):04d}.jpg"
        image_path = output_path / breed_name / image_filename
        
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image.save(image_path, 'JPEG', quality=95)
            breed_counts[breed_name] = breed_counts.get(breed_name, 0) + 1
        except Exception as e:
            print(f"\n   ⚠ Görsel kaydedilemedi: {image_filename} - {e}")
            continue
    
    print("\n" + "=" * 70)
    print("SONUÇ")
    print("=" * 70)
    print(f"\n✓ Dataset başarıyla organize edildi!\n")
    print("Irk başına görsel sayısı:")
    total = 0
    for breed_name, count in sorted(breed_counts.items()):
        if count > 0:
            print(f"   {breed_name:30s}: {count:4d} görsel")
            total += count
    print(f"\n   {'TOPLAM':30s}: {total:4d} görsel")
    
    print(f"\n📁 Dataset konumu: {output_path.absolute()}")
    
    return True


def train_with_cat_breeds(dataset_dir="cat_breeds_dataset"):
    """
    Kedi ırkları için eğitim talimatları.
    """
    dataset_path = Path(dataset_dir)
    
    print("\n" + "=" * 70)
    print("SONRAKİ ADIM: EĞİTİM")
    print("=" * 70)
    
    print("\nKedi ırklarını eğitmek için:")
    print(f"\n   python train_breed_classifier.py \\")
    print(f"       --dataset_path \"{dataset_path}\" \\")
    print(f"       --species cat \\")
    print(f"       --samples_per_breed 50")
    
    print("\n💡 İpucu:")
    print("   - samples_per_breed: Her ırktan kaç örnek kullanılacak")
    print("   - Daha fazla örnek = daha iyi sonuç ama daha yavaş eğitim")
    print("   - 50-100 arası genelde yeterli")
    

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Cat breed dataset'ini indir ve organize et")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="cat_breeds_dataset",
        help="Çıktı klasörü (varsayılan: cat_breeds_dataset)"
    )
    parser.add_argument(
        "--max_per_breed",
        type=int,
        default=200,
        help="Her ırk için maksimum görsel sayısı (varsayılan: 200)"
    )
    
    args = parser.parse_args()
    
    print("\n🐱 Cat Breed Dataset İndirme Aracı\n")
    
    success = download_cat_breed_dataset(
        output_dir=args.output_dir,
        max_images_per_breed=args.max_per_breed
    )
    
    if success:
        train_with_cat_breeds(args.output_dir)
        
        print("\n" + "=" * 70)
        print("✓ HAZIR!")
        print("=" * 70)
        print("\nDataset indirildi. Şimdi eğitimi başlatabilirsin!")
    else:
        print("\n✗ Dataset indirme başarısız oldu.")
