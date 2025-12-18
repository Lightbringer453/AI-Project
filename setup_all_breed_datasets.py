"""
Tüm hayvan türleri için breed dataset'lerini indirip eğiten master script.

Bu script:
1. Kedi breed dataset'ini indirir
2. Köpek breed dataset'ini indirir (mümkünse)
3. Oxford Pets dataset'ini indirir (kedi ve köpek)
4. Tüm breed classifier'ları eğitir

Kullanım:
    python setup_all_breed_datasets.py
"""

from pathlib import Path
import subprocess
import sys
import os


def run_command(command, description):
    print("\n" + "=" * 70)
    print(description)
    print("=" * 70)
    print(f"\nÇalıştırılıyor: {command}\n")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=False,
            text=True
        )
        print(f"\n✓ {description} tamamlandı!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} başarısız oldu: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Hata: {e}")
        return False


def main():
    print("=" * 70)
    print("TÜM BREED DATASET'LERİNİ İNDİRME VE EĞİTİM")
    print("=" * 70)
    print("\nBu script şunları yapacak:")
    print("1. Kedi breed dataset'ini indirir")
    print("2. Köpek breed dataset'ini indirir (mümkünse)")
    print("3. Oxford Pets dataset'ini indirir")
    print("4. Tüm breed classifier'ları eğitir")
    print("\n⚠ Bu işlem biraz zaman alabilir ve internet bağlantısı gerektirir.\n")
    
    input("Devam etmek için Enter'a basın...")
    
    success_count = 0
    total_steps = 0
    
    total_steps += 1
    if run_command(
        "python download_cat_breed_dataset.py --max_per_breed 200",
        "1. Kedi Breed Dataset İndirme"
    ):
        success_count += 1
    
    total_steps += 1
    if run_command(
        "python download_dog_breed_dataset.py --max_per_breed 200",
        "2. Köpek Breed Dataset İndirme"
    ):
        success_count += 1
    else:
        print("\n⚠ Köpek dataset indirilemedi, manuel indirme gerekebilir")
    
    total_steps += 1
    if run_command(
        "python download_oxford_pets.py",
        "3. Oxford Pets Dataset İndirme"
    ):
        success_count += 1
    
    cat_dataset_paths = [
        "cat_breeds_dataset",
        "oxford_pets_organized"
    ]
    
    for dataset_path in cat_dataset_paths:
        if Path(dataset_path).exists():
            total_steps += 1
            if run_command(
                f'python train_breed_classifier.py --dataset_path "{dataset_path}" --species cat --samples_per_breed 50',
                f"4. Kedi Breed Classifier Eğitimi ({dataset_path})"
            ):
                success_count += 1
            break
    
    dog_dataset_paths = [
        "dog_breeds_dataset",
        "oxford_pets_organized"
    ]
    
    for dataset_path in dog_dataset_paths:
        if Path(dataset_path).exists():
            total_steps += 1
            if run_command(
                f'python train_breed_classifier.py --dataset_path "{dataset_path}" --species dog --samples_per_breed 50',
                f"5. Köpek Breed Classifier Eğitimi ({dataset_path})"
            ):
                success_count += 1
            break
    
    print("\n" + "=" * 70)
    print("ÖZET")
    print("=" * 70)
    print(f"\nTamamlanan: {success_count}/{total_steps} adım")
    
    if success_count == total_steps:
        print("\n✓ Tüm işlemler başarıyla tamamlandı!")
        print("\nArtık breed classifier'ınız hazır. Test edebilirsiniz:")
        print("   python test_simple.py test_image.jpg")
        print("   streamlit run app/interface.py")
    else:
        print(f"\n⚠ Bazı adımlar başarısız oldu ({total_steps - success_count} adım)")
        print("\nManuel olarak eksik dataset'leri indirip eğitebilirsiniz:")
        print("\nKediler için:")
        print("   python download_cat_breed_dataset.py")
        print("   python train_breed_classifier.py --dataset_path cat_breeds_dataset --species cat --samples_per_breed 50")
        print("\nKöpekler için:")
        print("   python download_dog_breed_dataset.py")
        print("   python train_breed_classifier.py --dataset_path dog_breeds_dataset --species dog --samples_per_breed 50")


if __name__ == "__main__":
    main()

