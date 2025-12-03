# Breed Classifier Dataset Entegrasyonu Kılavuzu

Bu kılavuz, animal breed datasetlerini kullanarak daha doğru breed tahminleri yapmak için gereken adımları açıklar.

## 🎯 Nasıl Çalışıyor?

1. **Dataset'ten Öğrenme**: Dataset'teki her breed için birkaç örnek görüntü alınır
2. **Feature Extraction**: Pre-trained ResNet18 modeli kullanılarak her görüntüden feature vector çıkarılır
3. **Reference Vector Oluşturma**: Her breed için feature vector'lerin ortalaması alınarak bir "reference vector" oluşturulur
4. **Tahmin**: Yeni bir görüntü geldiğinde, feature vector'ü çıkarılır ve cosine similarity ile en yakın breed bulunur

## 📁 Dataset Yapısı

Dataset'iniz şu şekilde organize edilmiş olmalı:

```
dataset/
├── Persian/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Siamese/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Maine Coon/
│   └── ...
└── British Shorthair/
    └── ...
```

**Önemli**: 
- Her breed için ayrı bir klasör olmalı
- Klasör adı breed adı olmalı (büyük/küçük harf duyarlı değil)
- Desteklenen formatlar: `.jpg`, `.jpeg`, `.png`, `.bmp`

## 🚀 Kullanım

### 1. Dataset'ten Breed Vectors Oluşturma

```bash
python train_breed_classifier.py --dataset_path path/to/your/dataset --species cat --samples_per_breed 10
```

**Parametreler:**
- `--dataset_path`: Dataset klasörünün yolu
- `--species`: Hayvan türü (`cat`, `dog`, `bird`, `horse`, `cow`, `sheep`)
- `--samples_per_breed`: Her breed için kullanılacak örnek sayısı (varsayılan: 10)

**Örnek:**
```bash
# Kediler için
python train_breed_classifier.py --dataset_path ./datasets/cats --species cat --samples_per_breed 15

# Köpekler için
python train_breed_classifier.py --dataset_path ./datasets/dogs --species dog --samples_per_breed 20
```

### 2. Otomatik Kullanım

Breed vectors oluşturulduktan sonra, sistem otomatik olarak bunları kullanır. Herhangi bir ek ayar gerekmez!

- Breed vectors `models/breed_vectors.pkl` dosyasına kaydedilir
- Sistem başlatıldığında otomatik olarak yüklenir
- Yeni görüntüler için dataset-based matching kullanılır

## 📊 Performans İyileştirmesi

### Daha İyi Sonuçlar İçin:

1. **Daha Fazla Örnek**: `--samples_per_breed` değerini artırın (20-30 önerilir)
2. **Kaliteli Görüntüler**: Dataset'inizde net, iyi aydınlatılmış görüntüler kullanın
3. **Çeşitlilik**: Her breed için farklı açılardan, farklı yaşlardan örnekler ekleyin
4. **Temiz Dataset**: Yanlış etiketlenmiş görüntüleri kaldırın

### Önerilen Dataset Boyutları:

- **Minimum**: Her breed için 5-10 görüntü
- **Önerilen**: Her breed için 15-25 görüntü
- **Optimal**: Her breed için 30+ görüntü

## 🔧 Gelişmiş Kullanım

### Python Script ile Kullanım

```python
from models.animal_analysis import AnimalAnalyzer

# Initialize analyzer
analyzer = AnimalAnalyzer()

# Build breed vectors from dataset
breed_vectors = analyzer.build_breed_vectors_from_dataset(
    dataset_path="./datasets/cats",
    species="cat",
    samples_per_breed=20
)

# Now the analyzer will automatically use these vectors
# for breed classification
```

### Birden Fazla Tür İçin

```bash
# Kediler için
python train_breed_classifier.py --dataset_path ./datasets/cats --species cat

# Köpekler için
python train_breed_classifier.py --dataset_path ./datasets/dogs --species dog
```

Her tür için ayrı ayrı çalıştırın. Sistem tüm türlerin breed vectors'lerini saklar.

## 📝 Notlar

- **İlk Çalıştırma**: İlk çalıştırmada ResNet18 modeli otomatik olarak indirilir (yaklaşık 45MB)
- **GPU Desteği**: CUDA varsa otomatik olarak kullanılır, yoksa CPU kullanılır
- **Bellek**: Her breed için yaklaşık 512 boyutlu feature vector saklanır (çok küçük)
- **Hız**: Feature extraction her görüntü için ~50-100ms (GPU'da daha hızlı)

## 🐛 Sorun Giderme

### "No pre-computed breed vectors found" Hatası

Bu normaldir! İlk kullanımda breed vectors oluşturmanız gerekir:
```bash
python train_breed_classifier.py --dataset_path your/dataset/path --species cat
```

### "Feature extractor not initialized" Hatası

PyTorch ve torchvision yüklü olduğundan emin olun:
```bash
pip install torch torchvision
```

### Düşük Doğruluk

1. Dataset'inizi kontrol edin (yanlış etiketlenmiş görüntüler var mı?)
2. Daha fazla örnek kullanın (`--samples_per_breed` değerini artırın)
3. Daha kaliteli görüntüler ekleyin

## 📚 Dataset Kaynakları

İnternetten bulabileceğiniz bazı dataset örnekleri:

- **Kaggle**: Pet breed classification datasets
- **Stanford Dogs Dataset**: Köpek breed'leri için
- **Cat vs Dog Datasets**: Kedi ve köpek görüntüleri
- **Oxford-IIIT Pet Dataset**: Kedi ve köpek breed'leri

## 💡 İpuçları

1. **Küçük Başlayın**: Önce 3-4 breed ile test edin
2. **Yavaşça Genişletin**: İyi çalıştığından emin olduktan sonra daha fazla breed ekleyin
3. **Düzenli Güncelleyin**: Yeni görüntüler ekledikçe breed vectors'leri yeniden oluşturun
4. **Backup Alın**: `models/breed_vectors.pkl` dosyasını yedekleyin

## 🎉 Başarılar!

Dataset entegrasyonu tamamlandıktan sonra, breed tahminleri çok daha doğru olacak. Sistem otomatik olarak dataset'ten öğrenilen bilgileri kullanacak!

