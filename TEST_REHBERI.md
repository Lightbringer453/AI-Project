# Test Rehberi - Türkçe

## 🚀 Hızlı Başlangıç

### Yöntem 1: Streamlit Web Arayüzü (En Kolay)

```bash
streamlit run app/interface.py
```

Tarayıcıda açılan sayfadan görsel yükle ve sonuçları gör!

### Yöntem 2: Komut Satırı ile Test

```bash
# Basit test scripti ile
python test_simple.py path/to/görsel.jpg

# Veya ana script ile
python app/main.py path/to/görsel.jpg
```

### Yöntem 3: Python Script ile Test

```bash
python example_usage.py
```

(Önce `example_usage.py` dosyasındaki `image_path` değişkenini güncelle)

### Yöntem 4: Jupyter Notebook

```bash
jupyter notebook notebooks/demo.ipynb
```

## 📸 Test Görseli Nasıl Bulunur?

### Seçenek 1: Kendi Görselini Kullan
- İnsan veya hayvan içeren herhangi bir fotoğraf
- JPG, PNG, JPEG formatında olmalı

### Seçenek 2: İnternetten İndir
Örnek görseller için:
- Unsplash.com (ücretsiz fotoğraflar)
- Pexels.com (ücretsiz fotoğraflar)
- Google Images (kullanım haklarına dikkat!)

### Seçenek 3: Test Görseli Oluştur
Proje klasörüne `test_image.jpg` adında bir görsel koy.

## ✅ Test Adımları

1. **Görsel hazırla**
   - İnsan veya hayvan içeren bir fotoğraf seç
   - Proje klasörüne kopyala

2. **Test çalıştır**
   ```bash
   python test_simple.py test_image.jpg
   ```

3. **Sonuçları kontrol et**
   - Konsolda sonuçları gör
   - `outputs/` klasöründe işaretlenmiş görseli kontrol et
   - `outputs/` klasöründe JSON dosyasını kontrol et

## 🔍 Ne Test Edilmeli?

### İnsan Tespiti
- ✅ Yaş tahmini çalışıyor mu?
- ✅ Cinsiyet tespiti doğru mu?
- ✅ Duygu tespiti var mı?

### Hayvan Tespiti
- ✅ Hayvan türü doğru mu?
- ✅ Irk tahmini yapılıyor mu?
- ✅ Olgunluk durumu belirleniyor mu?

## ⚠️ Olası Sorunlar

### "Model bulunamadı" hatası
- İnternet bağlantını kontrol et (ilk kullanımda modeller indirilir)
- YOLOv8 modeli otomatik indirilecek

### "DeepFace hatası"
- TensorFlow kurulu mu kontrol et: `pip install tensorflow`
- DeepFace modelleri ilk kullanımda indirilir (biraz zaman alabilir)

### "Görsel bulunamadı" hatası
- Görsel yolunu kontrol et
- Görsel dosyasının var olduğundan emin ol

## 📊 Beklenen Çıktı

Başarılı bir test şunları göstermeli:
- Tespit edilen nesne sayısı
- Her nesne için:
  - Bounding box koordinatları
  - Güven skoru
  - Özellikler (yaş, cinsiyet, tür, vb.)
- `outputs/` klasöründe işaretlenmiş görsel
- JSON formatında sonuç dosyası

## 🎯 Örnek Test Komutları

```bash
# Basit test
python test_simple.py foto.jpg

# Ana pipeline ile
python app/main.py foto.jpg

# Web arayüzü
streamlit run app/interface.py
```

Başarılar! 🎉

