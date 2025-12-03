"""
Test görseli indirme scripti
"""

import urllib.request
from pathlib import Path

def download_test_image():
    """Örnek test görseli indir"""
    
    # Ücretsiz test görseli URL'leri (Unsplash)
    test_images = {
        "1": {
            "url": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=800",
            "name": "test_person.jpg",
            "description": "İnsan içeren görsel"
        },
        "2": {
            "url": "https://images.unsplash.com/photo-1517849845537-4d257902454a?w=800",
            "name": "test_dog.jpg",
            "description": "Köpek içeren görsel"
        },
        "3": {
            "url": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=800",
            "name": "test_cat.jpg",
            "description": "Kedi içeren görsel"
        }
    }
    
    print("=" * 60)
    print("TEST GÖRSELİ İNDİR")
    print("=" * 60)
    print("\nHangi görseli indirmek istersin?")
    print("1. İnsan içeren görsel")
    print("2. Köpek içeren görsel")
    print("3. Kedi içeren görsel")
    
    choice = input("\nSeçim (1-3): ").strip()
    
    if choice not in test_images:
        print("Geçersiz seçim!")
        return
    
    selected = test_images[choice]
    
    try:
        print(f"\n{selected['description']} indiriliyor...")
        print(f"URL: {selected['url']}")
        
        urllib.request.urlretrieve(selected['url'], selected['name'])
        
        print(f"\n✓ Başarılı! Görsel '{selected['name']}' olarak kaydedildi.")
        print(f"\nŞimdi test edebilirsin:")
        print(f"  python test_simple.py {selected['name']}")
        
    except Exception as e:
        print(f"\n❌ Hata: {e}")
        print("\nAlternatif:")
        print("1. Kendi görselini kullan")
        print("2. İnternetten manuel olarak indir")
        print("3. Unsplash.com veya Pexels.com'dan ücretsiz görsel indir")


if __name__ == "__main__":
    download_test_image()

