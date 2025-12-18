"""
Dataset klasör isimlerini İtalyanca'dan İngilizce'ye çevir.
"""

import os
from pathlib import Path
import shutil

translation = {
    "cane": "dog",
    "cavallo": "horse",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "ragno": "spider",
    "scoiattolo": "squirrel"
}

def rename_folders(dataset_path="animal_dataset"):
    """Klasör isimlerini İngilizce'ye çevir."""
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"Hata: {dataset_path} bulunamadı!")
        return
    
    print("=" * 70)
    print("KLASÖR İSİMLERİNİ ÇEVİRİYOR")
    print("=" * 70)
    
    for italian_name, english_name in translation.items():
        italian_path = dataset_path / italian_name
        english_path = dataset_path / english_name
        
        if italian_path.exists():
            print(f"✓ {italian_name} -> {english_name}")
            italian_path.rename(english_path)
        else:
            print(f"  {italian_name} bulunamadı, atlanıyor...")
    
    print("\n" + "=" * 70)
    print("TAMAMLANDI!")
    print("=" * 70)
    print("\nYeni klasörler:")
    for folder in sorted(dataset_path.iterdir()):
        if folder.is_dir():
            count = len(list(folder.glob("*.jpg")))
            print(f"  {folder.name}: {count} görsel")

if __name__ == "__main__":
    rename_folders()
