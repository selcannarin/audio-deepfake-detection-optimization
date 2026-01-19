Proje Özeti
Bu çalışma, Deepfake (sahte) seslerin tespitinde kullanılan Derin Öğrenme modellerinin (1D-CNN) başarısını Optimizasyon Teorisi perspektifinden incelemektedir. Çalışma iki ana modülden oluşmaktadır:

Eğitim ve Analiz Modülü (Colab): SGD, Adam, AdamW ve RMSProp algoritmalarının kayıp yüzeyindeki davranışları, gradyan normları ve yakınsama hızları analiz edilmiştir.

Gerçek Zamanlı Tespit Modülü (Local/Streamlit): Eğitilen modellerin yerel bilgisayarda çalıştırılmasını sağlayan, "Alan Kayması" (Domain Shift) düzeltmelerini içeren etkileşimli arayüzdür.

Proje Klasör Yapısı
Proje dosyaları, eğitim ve uygulama süreçlerini ayıracak şekilde düzenlenmiştir:


Deepfake_Optimization_Project/
│
├── README.md                       # Proje dokümantasyonu (Bu dosya)
├── requirements.txt                # Gerekli Python kütüphaneleri
│
├── 1_Training_Analysis_Colab/      # MODÜL 1: Eğitim ve Analiz Kodları
│   ├── DeepfakeAudioDetector.py    # Ana eğitim ve analiz scripti
│   ├── audioDeepfake.txt           # Eğitim logları ve ham çıktılar
│   └── reports/                    # JSON formatında detaylı metrikler
│
├── 2_Streamlit_App_Local/          # MODÜL 2: Yerel Arayüz Uygulaması
│   ├── app.py                      # Streamlit ana uygulama dosyası
│   └── utils.py                    # Yardımcı fonksiyonlar (opsiyonel)
│
├── models/                         # Eğitilmiş Modeller (Her iki modül kullanır)
│   ├── cnn_adamw_model.pth         
│   ├── cnn_adam_model.pth
│   ├── cnn_sgd_model.pth
│   ├── cnn_rmsprop_model.pth
│   └── scaler.pkl                  # Öznitelik ölçeklendirici
│
└── figures/                        # Analiz Grafikleri
    ├── optimizer_comparison.png    # Optimizer karşılaştırma grafiği
    ├── gradient_explainability.png # Saliency map ve FGSM
    └── ensemble_roc.png            # Ensemble performans eğrisi