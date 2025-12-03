"""
Basit test scripti - Hızlı test için
"""

from app.main import ImageAnalysisPipeline
from utils.image_loader import load_image
import cv2

def test_with_image(image_path):
    """Görsel ile test et"""
    print("=" * 60)
    print("GÖRÜNTÜ ANALİZİ TESTİ")
    print("=" * 60)
    
    try:
        # Pipeline'ı başlat
        print("\n1. Pipeline başlatılıyor...")
        pipeline = ImageAnalysisPipeline(detector_type="yolo")
        print("   ✓ Pipeline hazır!")
        
        # Görseli işle
        print(f"\n2. Görsel işleniyor: {image_path}")
        result = pipeline.process_image(image_path, save_output=True)
        print("   ✓ İşlem tamamlandı!")
        
        # Sonuçları göster
        print("\n" + "=" * 60)
        print("SONUÇLAR")
        print("=" * 60)
        print(f"Toplam tespit: {result['summary']['total_detections']}")
        print(f"İnsan sayısı: {result['summary']['humans']}")
        print(f"Hayvan sayısı: {result['summary']['animals']}")
        
        # Detaylı sonuçlar
        if result['detections']:
            print("\n" + "-" * 60)
            print("DETAYLI SONUÇLAR")
            print("-" * 60)
            for i, detection in enumerate(result['detections'], 1):
                print(f"\n[{i}] {detection['class_type'].upper()}")
                print(f"    Sınıf: {detection.get('class_name', 'N/A')}")
                print(f"    Güven: {detection.get('confidence', 0.0):.2%}")
                
                attrs = detection.get('attributes', {})
                if detection['class_type'] == 'human':
                    age = attrs.get('age')
                    if age is not None:
                        print(f"    Yaş: {age:.0f}")
                    else:
                        print(f"    Yaş: N/A")
                    print(f"    Cinsiyet: {attrs.get('gender', 'N/A')}")
                    print(f"    Duygu: {attrs.get('emotion', 'N/A')}")
                elif detection['class_type'] == 'animal':
                    print(f"    Tür: {attrs.get('species', 'N/A')}")
                    print(f"    Irk: {attrs.get('breed', 'N/A')}")
                    print(f"    Olgunluk: {attrs.get('maturity', 'N/A')}")
        
        print("\n" + "=" * 60)
        print("✓ Test başarılı! Sonuçlar 'outputs' klasörüne kaydedildi.")
        print("=" * 60)
        
    except FileNotFoundError:
        print(f"\n❌ HATA: '{image_path}' dosyası bulunamadı!")
        print("\nÇözüm:")
        print("  1. Görsel dosyasının yolunu kontrol et")
        print("  2. Veya yeni bir test görseli indir")
    except Exception as e:
        print(f"\n❌ HATA: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Varsayılan test görseli yolu
        image_path = "test_image.jpg"
        print(f"Görsel yolu belirtilmedi, varsayılan kullanılıyor: {image_path}")
        print("Özel görsel için: python test_simple.py <görsel_yolu>")
        print()
    
    test_with_image(image_path)

