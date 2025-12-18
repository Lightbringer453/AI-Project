"""
Köpek ırkları dataset'ini Hugging Face'ten indirip organize eden script.

Dataset: dog-breed-identification veya benzeri
Kaynak: Hugging Face datasets

Kullanım:
    python download_dog_breed_dataset.py
"""

from datasets import load_dataset
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import os


def download_dog_breed_dataset(output_dir="dog_breeds_dataset", max_images_per_breed=200):
    """
    Dog breed dataset'ini indir ve organize et.
    
    Args:
        output_dir: Çıktı klasörü
        max_images_per_breed: Her ırk için maksimum görsel sayısı
    """
    print("=" * 70)
    print("DOG BREED DATASET İNDİRME VE ORGANİZASYON")
    print("=" * 70)
    
    dataset_sources = [
        ("stanford-dogs", "train"),  # Stanford Dogs Dataset
        ("dog-breed-identification", "train"),  # Kaggle'dan
        ("dogs-vs-cats", "train"),  # Alternatif
    ]
    
    dataset = None
    dataset_name = None
    
    print("\n1. Dataset yükleniyor (bu biraz zaman alabilir)...")
    for source_name, split_name in dataset_sources:
        try:
            print(f"   Deneniyor: {source_name}...")
            dataset = load_dataset(source_name, split=split_name)
            dataset_name = source_name
            print(f"   ✓ Dataset yüklendi: {source_name} - {len(dataset)} görsel")
            break
        except Exception as e:
            print(f"   ✗ {source_name} bulunamadı: {e}")
            continue
    
    if dataset is None:
        print("\n   Alternatif yöntem deneniyor...")
        try:
            dataset = load_dataset("imagefolder", data_dir="path/to/dogs")  # Bu örnek, gerçek path gerekli
            print(f"   ✓ Dataset yüklendi: {len(dataset)} görsel")
        except Exception as e2:
            print(f"   ✗ Alternatif yöntem başarısız: {e2}")
            print("\n⚠ UYARI: Otomatik dataset bulunamadı!")
            print("\nManuel olarak köpek breed dataset'i indirmeniz gerekiyor:")
            print("1. Stanford Dogs Dataset: https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset")
            print("2. Veya Hugging Face'te 'dog breed' araması yapın")
            print("3. Dataset'i indirip 'dog_breeds_dataset' klasörüne breed klasörleri halinde yerleştirin")
            print("   Örnek yapı:")
            print("   dog_breeds_dataset/")
            print("     ├── Golden Retriever/")
            print("     ├── Labrador/")
            print("     ├── German Shepherd/")
            print("     └── ...")
            return False
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print(f"\n2. Görseller '{output_dir}' klasörüne kaydediliyor...")
    
    breed_names = []
    if hasattr(dataset.features, "label") and hasattr(dataset.features["label"], "names"):
        breed_names = dataset.features["label"].names
    elif "label" in dataset.features:
        unique_labels = set()
        for item in dataset:
            label = item.get("label", item.get("breed", "unknown"))
            if isinstance(label, int):
                if hasattr(dataset.features["label"], "names"):
                    breed_names = dataset.features["label"].names
                    break
            else:
                unique_labels.add(str(label))
        if not breed_names:
            breed_names = sorted(list(unique_labels))
    else:
        for item in dataset:
            breed = item.get("breed", item.get("label", "unknown"))
            if breed not in breed_names:
                breed_names.append(str(breed))
    
    if not breed_names:
        print("   ⚠ Breed isimleri bulunamadı, 'unknown' olarak işaretlenecek")
        breed_names = ["unknown"]
    
    print(f"   Bulunan ırklar: {len(breed_names)}")
    for i, breed in enumerate(breed_names[:15], 1):
        print(f"     {i}. {breed}")
    if len(breed_names) > 15:
        print(f"     ... ve {len(breed_names) - 15} tane daha")
    
    for breed_name in breed_names:
        clean_name = breed_name.replace("/", "_").replace("\\", "_").strip()
        breed_path = output_path / clean_name
        breed_path.mkdir(exist_ok=True)
    
    breed_counts = {name: 0 for name in breed_names}
    
    print("\n3. Görseller işleniyor...")
    for idx, item in enumerate(tqdm(dataset, desc="   Progress", ncols=70)):
        breed_name = "unknown"
        if "label" in item:
            label = item["label"]
            if isinstance(label, int) and label < len(breed_names):
                breed_name = breed_names[label]
            else:
                breed_name = str(label)
        elif "breed" in item:
            breed_name = str(item["breed"])
        
        clean_name = breed_name.replace("/", "_").replace("\\", "_").strip()
        
        if breed_counts.get(breed_name, 0) >= max_images_per_breed:
            continue
        
        if "image" not in item:
            continue
        image = item["image"]
        
        image_filename = f"{clean_name}_{breed_counts.get(breed_name, 0):04d}.jpg"
        image_path = output_path / clean_name / image_filename
        
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


def train_with_dog_breeds(dataset_dir="dog_breeds_dataset"):
    """
    Köpek ırkları için eğitim talimatları.
    """
    dataset_path = Path(dataset_dir)
    
    print("\n" + "=" * 70)
    print("SONRAKİ ADIM: EĞİTİM")
    print("=" * 70)
    
    print("\nKöpek ırklarını eğitmek için:")
    print(f"\n   python train_breed_classifier.py \\")
    print(f"       --dataset_path \"{dataset_path}\" \\")
    print(f"       --species dog \\")
    print(f"       --samples_per_breed 50")
    
    print("\n💡 İpucu:")
    print("   - samples_per_breed: Her ırktan kaç örnek kullanılacak")
    print("   - Daha fazla örnek = daha iyi sonuç ama daha yavaş eğitim")
    print("   - 50-100 arası genelde yeterli")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Dog breed dataset'ini indir ve organize et")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dog_breeds_dataset",
        help="Çıktı klasörü (varsayılan: dog_breeds_dataset)"
    )
    parser.add_argument(
        "--max_per_breed",
        type=int,
        default=200,
        help="Her ırk için maksimum görsel sayısı (varsayılan: 200)"
    )
    
    args = parser.parse_args()
    
    print("\n🐕 Dog Breed Dataset İndirme Aracı\n")
    
    success = download_dog_breed_dataset(
        output_dir=args.output_dir,
        max_images_per_breed=args.max_per_breed
    )
    
    if success:
        train_with_dog_breeds(args.output_dir)
        
        print("\n" + "=" * 70)
        print("✓ HAZIR!")
        print("=" * 70)
        print("\nDataset indirildi. Şimdi eğitimi başlatabilirsin!")
    else:
        print("\n✗ Dataset indirme başarısız oldu.")
        print("\nManuel indirme için:")
        print("1. Kaggle'dan Stanford Dogs Dataset indir")
        print("2. Veya Hugging Face'te dog breed dataset ara")
        print("3. Dataset'i breed klasörlerine organize et")

