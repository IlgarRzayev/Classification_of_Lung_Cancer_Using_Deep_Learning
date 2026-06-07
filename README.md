
# Derin Öğrenme ile Akciğer Kanseri Sınıflandırması (Chest CT-Scan)

Bu proje, Bilgisayarlı Tomografi (CT-Scan) göğüs görüntülerini kullanarak akciğer kanseri türlerini yüksek doğrulukla sınıflandırmak amacıyla geliştirilmiş bir uçtan uca derin öğrenme projesidir. Projede, transfer öğrenmesi (transfer learning) yöntemiyle **EfficientNetB3** mimarisi kullanılmış, veri setindeki kontrast sorunları ve sınıf dengesizlikleri ileri düzey veri işleme teknikleriyle çözülmüştür.

## 📊 Veri Seti Hakkında
Projede Kaggle üzerinde yer alan [Chest CT-Scan Görüntüleri](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images) kullanılmıştır. Veri seti toplam **1000 adet** BT görüntüsünden oluşmakta ve 4 farklı sınıfı barındırmaktadır:
1. **Adenokarsinom (Adenocarcinoma)**
2. **Büyük Hücreli Karsinom (Large Cell Carcinoma)**
3. **Yassı Hücreli Karsinom (Squamous Cell Carcinoma)**
4. **Normal (Sağlıklı)**

## 🛠️ Uygulanan Gelişmiş Teknikler

* **CLAHE Kontrast Artırma:** Medikal görüntülerin (özellikle CT görüntülerinin) doku detaylarını ve tümör sınırlarını belirginleştirmek için **LAB renk uzayında L (parlaklık) kanalına** CLAHE (Contrast Limited Adaptive Histogram Equalization) algoritması uygulanmıştır.
* **Sınıf Dengesizliği (Class Imbalance) Çözümü:** Veri setindeki sınıfların homojen dağılmamasından kaynaklanan yanlılığı engellemek adına `scikit-learn` kullanılarak **Sınıf Ağırlıkları (Class Weights)** hesaplanmış ve eğitim bu ağırlıklarla optimize edilmiştir.
* **İki Aşamalı Eğitim (Fine-Tuning):** 1. *Aşama 1:* EfficientNetB3'ün dondurulmuş (frozen) ana ağırlıkları korunarak sadece sınıflandırma başlığı (Dense katmanları) eğitilmiştir.
    2. *Aşama 2:* Modelin belirli bir derinlikten sonraki katmanları çözülerek düşük bir öğrenme oranı (`1e-4`) ile ince ayar (Fine-Tuning) yapılmıştır.
* **Dinamik Öğrenme Oranı & Erken Durdurma:** Aşırı öğrenmeyi (overfitting) engellemek için `EarlyStopping` ve validation kaybı durakladığında öğrenme oranını otomatik düşüren `ReduceLROnPlateau` callback'leri entegre edilmiştir.

## 📐 Model Mimarisi

* **Base Model:** EfficientNetB3 (ImageNet ağırlıkları ile başlatılmıştır)
* **Global Average Pooling 2D**
* **Batch Normalization & Dropout (0.4)** -> Regülasyon için
* **Dense Katmanı (256 Nöron, ReLU)**
* **Batch Normalization & Dropout (0.3)**
* **Çıkış Katmanı (4 Nöron, Softmax)** -> Çok sınıflı sınıflandırma için

## 🚀 Kurulum ve Çalıştırma

### Gereksinimler
Projeyi çalıştırmak için aşağıdaki kütüphanelerin yüklenmesi gerekmektedir:
```bash
pip install tensorflow keras efficientnet opencv-python-headless numpy pandas matplotlib seaborn scikit-learn
```

### Kaggle Veri Setini İndirme

Google Colab veya yerel ortamınızda veri setini doğrudan çekebilmek için `kaggle.json` API anahtarınızın tanımlı olması gerekir. Kod içerisindeki ilgili hücre API entegrasyonunu otomatik olarak yapmaktadır.


### Çalıştırma

Proje bir Jupyter Notebook (`.ipynb`) olarak tasarlanmıştır. Adım adım tüm hücreleri çalıştırarak veri görselleştirmeden model testine kadar olan süreci izleyebilirsiniz:

`jupyter notebook Derin_Öğrenme_ile_Akciğer_Kanseri_Sınıflandırması.ipynb`

## 📈 Model Performansı ve Metrikler

Model, test veri seti üzerinde yapılan değerlendirmelerde **%79.05 Genel Doğruluk (Accuracy)** oranına ulaşmıştır. Sınıf bazlı kesinlik, duyarlılık ve F1-skor değerlerini içeren detaylı performans raporu aşağıda listelenmiştir:

| Test Edilen Sınıf / Hücre Tipi | Kesinlik (Precision) | Duyarlılık (Recall) | F1-Skor | Destek (Support) |
| :--- | :---: | :---: | :---: | :---: |
| **Adenokarsinom** | 0.7410 | 0.8583 | 0.7954 | 120 |
| **Büyük Hücreli Karsinom** | 0.7619 | 0.6275 | 0.6882 | 51 |
| **Yassı Hücreli Karsinom** | 0.8140 | 0.7778 | 0.7955 | 90 |
| **Normal (Sağlıklı)** | 0.8846 | 0.8519 | 0.8679 | 54 |
| 📊 **Genel Doğruluk (Accuracy)** | | | **0.7905** | **315** |


### Çıkarımlar:

-   Model, **Normal (Kanserli Olmayan)** akciğer dokularını **%88 Precision** ve **%86 F1-Skor** ile çok yüksek başarıyla ayırt edebilmektedir.
    
-   En zor ayırt edilen kanser türü olan _Büyük Hücreli Karsinom_ için bile kabul edilebilir bir performans yakalanmıştır.
