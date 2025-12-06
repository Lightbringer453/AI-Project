"""
Dataset'i Hugging Face'ten indirip organize eden script.

Dataset: animals-10 (26k+ fotoğraf)
Kaynak: https://huggingface.co/datasets/dgrnd4/animals-10

Kullanım:
    python download_animals_dataset.py
"""

from datasets import load_dataset
from pathlib import Path
from PIL import Image
import os
from tqdm import tqdm


def download_and_organize_dataset(output_dir="animal_dataset", max_images_per_class=500):
    """
    Animals-10 dataset'ini indir ve organize et.
    
    Args:
        output_dir: Çıktı klasörü
        max_images_per_class: Her sınıf için maksimum görsel sayısı (RAM tasarrufu için)
    """
    print("=" * 70)
    print("ANIMALS-10 DATASET İNDİRME VE ORGANİZASYON")
    print("=" * 70)
    
    # Dataset'i yükle
    print("\n1. Dataset yükleniyor (bu biraz zaman alabilir)...")
    try:
        # Train split'i yükle (daha fazla veri için)
        dataset = load_dataset("dgrnd4/animals-10", split="train")
        print(f"   ✓ Dataset yüklendi: {len(dataset)} görsel")
    except Exception as e:
        print(f"   ✗ Hata: {e}")
        print("\n   Alternatif yöntem deneniyor...")
        try:
            dataset = load_dataset("dgrnd4/animals-10")
            dataset = dataset["train"]
            print(f"   ✓ Dataset yüklendi: {len(dataset)} görsel")
        except Exception as e2:
            print(f"   ✗ Hata: {e2}")
            return False
    
    # Output klasörünü oluştur
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print(f"\n2. Görseller '{output_dir}' klasörüne kaydediliyor...")
    
    # Sınıf isimlerini al
    if hasattr(dataset.features["label"], "names"):
        class_names = dataset.features["label"].names
    else:
        # Manuel olarak sınıf isimlerini belirle (animals-10 için)
        class_names = [
            "dog", "cat", "horse", "spider", "butterfly",
            "chicken", "sheep", "cow", "squirrel", "elephant"
        ]
    
    print(f"   Sınıflar: {', '.join(class_names)}")
    
    # Her sınıf için klasör oluştur
    for class_name in class_names:
        class_path = output_path / class_name
        class_path.mkdir(exist_ok=True)
    
    # Görselleri organize et
    class_counts = {name: 0 for name in class_names}
    
    print("\n3. Görseller işleniyor...")
    for idx, item in enumerate(tqdm(dataset, desc="   Progress")):
        # Label'ı al
        label = item["label"]
        class_name = class_names[label]
        
        # Maksimum sayıya ulaşıldıysa atla
        if class_counts[class_name] >= max_images_per_class:
            continue
        
        # Görseli al
        image = item["image"]
        
        # Görseli kaydet
        image_filename = f"{class_name}_{class_counts[class_name]:04d}.jpg"
        image_path = output_path / class_name / image_filename
        
        try:
            # RGB'ye çevir (eğer grayscale ise)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Kaydet
            image.save(image_path, 'JPEG', quality=95)
            class_counts[class_name] += 1
        except Exception as e:
            print(f"\n   ⚠ Görsel kaydedilemedi: {image_filename} - {e}")
            continue
    
    # Sonuçları göster
    print("\n" + "=" * 70)
    print("SONUÇ")
    print("=" * 70)
    print(f"\n✓ Dataset başarıyla organize edildi!\n")
    print("Sınıf başına görsel sayısı:")
    total = 0
    for class_name, count in sorted(class_counts.items()):
        print(f"   {class_name:12s}: {count:4d} görsel")
        total += count
    print(f"\n   {'TOPLAM':12s}: {total:4d} görsel")
    
    print(f"\n📁 Dataset konumu: {output_path.absolute()}")
    
    return True


def prepare_for_training(dataset_dir="animal_dataset"):
    """
    Dataset'i eğitim için organize et (türlere göre grupla).
    """
    dataset_path = Path(dataset_dir)
    
    # Evcil hayvanlar ve çiftlik hayvanları için alt klasörler oluştur
    categories = {
        "pets": ["dog", "cat"],
        "farm": ["horse", "sheep", "cow", "chicken"],
        "wild": ["elephant", "squirrel"],
        "others": ["spider", "butterfly"]
    }
    
    print("\n" + "=" * 70)
    print("EĞİTİM İÇİN ÖNERİLER")
    print("=" * 70)
    
    print("\n1. Köpek ırkları için eğitim:")
    print("   python train_breed_classifier.py \\")
    print(f"       --dataset_path \"{dataset_path / 'dog'}\" \\")
    print("       --species dog --samples_per_breed 50")
    
    print("\n2. Kedi ırkları için eğitim:")
    print("   python train_breed_classifier.py \\")
    print(f"       --dataset_path \"{dataset_path / 'cat'}\" \\")
    print("       --species cat --samples_per_breed 50")
    
    print("\n3. Diğer hayvanlar için eğitim:")
    print("   python train_breed_classifier.py \\")
    print(f"       --dataset_path \"{dataset_path / 'horse'}\" \\")
    print("       --species horse --samples_per_breed 30")
    
    print("\n⚠ NOT: animals-10 dataset'i türler (species) içeriyor, ırklar (breeds) değil.")
    print("   Daha iyi ırk tespiti için breed-specific dataset'ler bulman gerekebilir.")
    print("\n   Örnek breed dataset'ler:")
    print("   - Stanford Dogs Dataset (120 köpek ırğı)")
    print("   - Oxford-IIIT Pet Dataset (37 kedi ve köpek ırğı)")
    

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Animals-10 dataset'ini indir ve organize et")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="animal_dataset",
        help="Çıktı klasörü (varsayılan: animal_dataset)"
    )
    parser.add_argument(
        "--max_per_class",
        type=int,
        default=500,
        help="Her sınıf için maksimum görsel sayısı (varsayılan: 500)"
    )
    
    args = parser.parse_args()
    
    print("\n🐾 Animals-10 Dataset İndirme Aracı\n")
    
    # Dataset'i indir ve organize et
    success = download_and_organize_dataset(
        output_dir=args.output_dir,
        max_images_per_class=args.max_per_class
    )
    
    if success:
        # Eğitim önerilerini göster
        prepare_for_training(args.output_dir)
        
        print("\n" + "=" * 70)
        print("BİR SONRAKİ ADIM")
        print("=" * 70)
        print("\n⚠ ÖNEMLİ: Bu dataset TÜRLER içeriyor, IRKLAR değil!")
        print("\nIrk tespiti için iki seçenek:")
        print("\n1. Bu dataset ile tür tespitini geliştir:")
        print("   - Daha doğru köpek/kedi/at tespiti")
        print("   - Mevcut sistemini iyileştir")
        
        print("\n2. Breed-specific dataset bul:")
        print("   - Stanford Dogs: 120 köpek ırğı")
        print("   - Oxford Pets: 37 kedi/köpek ırğı")
        print("   - Kaggle'da birçok breed dataset var")
        
        print("\n" + "=" * 70)
    else:
        print("\n✗ Dataset indirme başarısız oldu.")
        print("\nAlternatif: Manuel indirme")
        print("1. https://huggingface.co/datasets/dgrnd4/animals-10 adresini ziyaret et")
        print("2. Dataset'i manuel olarak indir")
        print("3. 'animal_dataset' klasörüne çıkart")
