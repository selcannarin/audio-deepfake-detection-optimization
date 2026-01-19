import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import librosa
import librosa.display
import joblib
import os
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

# Sayfa Ayarları
st.set_page_config(page_title="Deepfake Ses Tespiti", layout="wide")

# ==============================================================================
# 1. KONFİGÜRASYON
# ==============================================================================
CONFIG = {
    'sr': 16000,
    'duration': 2.0,
    'n_mfcc': 20,
    'n_lfcc': 20,
    'n_lps': 20,
    'n_contrast_bands': 6,
    'mpe_scales': [1, 2, 4, 8, 16],
    'mpe_orders': [3, 4, 5, 6],
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# ==============================================================================
# 2. MODEL SINIFI
# ==============================================================================
class ImprovedCNN(nn.Module):
    def __init__(self, input_dim=154):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ==============================================================================
# 3. YARDIMCI FONKSİYONLAR
# ==============================================================================
def preprocess_audio(audio, sr_orig, target_sr=16000, target_samples=32000):
    if sr_orig != target_sr:
        audio = librosa.resample(audio, orig_sr=sr_orig, target_sr=target_sr)
    if len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)), mode='constant')
    else:
        audio = audio[:target_samples]
    return audio

def extract_lps(audio, n_lps=20):
    S = np.abs(librosa.stft(audio, n_fft=512, hop_length=160))
    P = S ** 2
    LPS = np.log(P + 1e-10)
    lps_mean = LPS.mean(axis=1)
    lps_std = LPS.std(axis=1)
    
    def fix_dim(arr, target):
        if len(arr) > target: return np.mean(arr[:len(arr)//target*target].reshape(-1, len(arr)//target), axis=1)[:target]
        if len(arr) < target: return np.pad(arr, (0, target - len(arr)))
        return arr

    return np.concatenate([fix_dim(lps_mean, n_lps), fix_dim(lps_std, n_lps)])

def permutation_entropy(signal, order=3, delay=1):
    n = len(signal)
    if n < delay * (order - 1) + 1: return 0.0
    permutations = []
    for i in range(n - delay * (order - 1)):
        indices = [i + j * delay for j in range(order)]
        permutations.append(tuple(np.argsort(signal[indices])))
    if not permutations: return 0.0
    probs = np.array(list(Counter(permutations).values())) / len(permutations)
    return -np.sum(probs * np.log2(probs + 1e-10))

def extract_mpe(audio, scales, orders):
    mpe_feats = []
    for order in orders:
        for scale in scales:
            if scale == 1: sig = audio
            else:
                n = len(audio) // scale
                if n < order: 
                    mpe_feats.append(0.0)
                    continue
                sig = audio[:n*scale].reshape(n, scale).mean(axis=1)
            mpe_feats.append(permutation_entropy(sig, order=order))
    return np.array(mpe_feats)

def extract_features(audio, sr):
    lps = extract_lps(audio, CONFIG['n_lps'])
    S = np.abs(librosa.stft(audio))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    lfcc = librosa.feature.mfcc(S=S_db, n_mfcc=CONFIG['n_lfcc'])
    lfcc_feat = np.concatenate([lfcc.mean(axis=1), lfcc.std(axis=1)])
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_bands=CONFIG['n_contrast_bands'])
    cont_feat = np.concatenate([contrast.mean(axis=1), contrast.std(axis=1)])
    mpe = extract_mpe(audio, CONFIG['mpe_scales'], CONFIG['mpe_orders'])
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=CONFIG['n_mfcc'])
    mfcc_feat = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
    return np.concatenate([lps, lfcc_feat, cont_feat, mpe, mfcc_feat]).astype(np.float32)

# --- GÜNCELLENMİŞ GRAFİK FONKSİYONU ---
def plot_comprehensive_analysis(y, sr, features_scaled):
    """
    1. Waveform
    2. Spectrogram
    3. 154-D Feature Vector (Input Layer Visualization)
    """
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.5], hspace=0.4)
    
    # 1. Waveform
    ax0 = fig.add_subplot(gs[0])
    librosa.display.waveshow(y, sr=sr, ax=ax0, color='#3498db', alpha=0.8)
    ax0.set_title("1. Zaman Domeni (Waveform)", fontweight="bold")
    ax0.set_ylabel("Genlik")
    
    # 2. Mel-Spectrogram
    ax1 = fig.add_subplot(gs[1])
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_db = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax1, cmap='magma')
    ax1.set_title("2. Frekans Analizi (Mel-Spectrogram)", fontweight="bold")
    fig.colorbar(img, ax=ax1, format='%+2.0f dB')
    
    # 3. 154-D Feature Vector Bar Chart
    ax2 = fig.add_subplot(gs[2])
    
    # Feature grupları ve renkleri (Analysis raporuna uygun)
    # LPS: 0-40 (40), LFCC: 40-80 (40), Contrast: 80-94 (14), MPE: 94-114 (20), MFCC: 114-154 (40)
    features = features_scaled.flatten()
    
    indices = np.arange(len(features))
    
    # Grupları çiz
    ax2.bar(indices[:40], features[:40], color='purple', label='LPS (Log Power Spec)', alpha=0.7, width=1.0)
    ax2.bar(indices[40:80], features[40:80], color='green', label='LFCC', alpha=0.7, width=1.0)
    ax2.bar(indices[80:94], features[80:94], color='orange', label='Spectral Contrast', alpha=0.7, width=1.0)
    ax2.bar(indices[94:114], features[94:114], color='red', label='MPE (Entropy)', alpha=0.7, width=1.0)
    ax2.bar(indices[114:], features[114:], color='blue', label='MFCC', alpha=0.7, width=1.0)
    
    ax2.set_title("3. CNN Modeline Giren Vektör (154 Öznitelik - Input Layer)", fontweight="bold")
    ax2.set_xlabel("Öznitelik İndeksi")
    ax2.set_ylabel("Normalize Değer (Scaled)")
    ax2.legend(loc='upper right', fontsize='small', frameon=True)
    ax2.grid(True, alpha=0.2)
    
    # Ayrım çizgileri
    for boundary in [40, 80, 94, 114]:
        ax2.axvline(boundary, color='black', linestyle='--', alpha=0.5)

    return fig

# ==============================================================================
# 4. YÜKLEME
# ==============================================================================
@st.cache_resource
def load_resources():
    models = {}
    optimizers = ['sgd', 'adam', 'adamw', 'rmsprop']
    for opt in optimizers:
        path = f"models/cnn_{opt}_model.pth"
        if os.path.exists(path):
            model = ImprovedCNN().to(CONFIG['device'])
            model.load_state_dict(torch.load(path, map_location=CONFIG['device']))
            model.eval()
            models[opt] = model
    
    scaler = None
    if os.path.exists("models/scaler.pkl"):
        scaler = joblib.load("models/scaler.pkl")
    return models, scaler

models, scaler = load_resources()

# ==============================================================================
# 5. ARAYÜZ
# ==============================================================================

st.title("Optimizasyon Teorisi: Deepfake Ses Tespiti")
st.markdown("Farklı optimizerların (SGD, Adam, AdamW, RMSProp) performans analizi.")

# --- YAN MENÜ ---
st.sidebar.header("⚙️ Ayarlar")

scenario = st.sidebar.radio(
    "1. Test Senaryosu:",
    ("Laboratuvar Ortamı (ASVspoof)", "Gerçek Dünya (InTheWild)")
)

model_mode = st.sidebar.radio(
    "2. Karar Mekanizması:",
    ("Ensemble (Tüm Modeller)", "Sadece AdamW (En İyi Single)")
)

if scenario == "Laboratuvar Ortamı (ASVspoof)":
    current_threshold = 0.17
    inverted_correction = False 
    st.sidebar.success(f"Eşik: {current_threshold} (Lab verisi için optimize)")
else:
    current_threshold = 0.45
    inverted_correction = True
    st.sidebar.warning(f"Eşik: {current_threshold} (Gürültülü veri için optimize)")


tabs = st.tabs(["Canlı Analiz", "Grafikler", "Hakkında"])

with tabs[0]:
    uploaded_file = st.file_uploader("Ses dosyası yükle (.wav, .mp3)", type=['wav', 'mp3', 'flac'])
    
    if uploaded_file and st.button("Analiz Et"):
        if not models or not scaler:
            st.error("Model dosyaları eksik!")
        else:
            with open("temp.wav", "wb") as f: f.write(uploaded_file.getbuffer())
            
            with st.spinner("Sinyal işleniyor ve modeller tahmin yürütüyor..."):
                # İşlemler
                y, sr = librosa.load("temp.wav", sr=None)
                y_proc = preprocess_audio(y, sr) # Model için
                
                feat = extract_features(y_proc, CONFIG['sr'])
                feat_scaled = scaler.transform(feat.reshape(1, -1))
                feat_t = torch.FloatTensor(feat_scaled).to(CONFIG['device'])

                results = {}
                weights = {'sgd': 0.28, 'adam': 0.34, 'adamw': 0.33, 'rmsprop': 0.32}
                models_to_invert = ['sgd', 'adamw'] if inverted_correction else []

                weighted_sum = 0
                total_weight = 0
                
                for name, model in models.items():
                    with torch.no_grad():
                        prob = torch.sigmoid(model(feat_t)).item()
                    
                    if name in models_to_invert:
                        prob = 1 - prob
                        
                    results[name] = prob
                    
                    w = weights.get(name, 0.25)
                    weighted_sum += prob * w
                    total_weight += w

                if model_mode == "Sadece AdamW (En İyi Single)":
                    final_score = results['adamw']
                    used_method_name = "AdamW"
                else:
                    final_score = weighted_sum / total_weight
                    used_method_name = "Ensemble"

                # --- SONUÇ BÖLÜMÜ ---
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.subheader("Karar")
                    is_fake = final_score > current_threshold
                    lbl = "SAHTE (SPOOF)" if is_fake else "GERÇEK (BONAFIDE)"
                    clr = "red" if is_fake else "green"
                    
                    st.markdown(f"<h2 style='color: {clr};'>{lbl}</h2>", unsafe_allow_html=True)
                    st.metric(f"Skor ({used_method_name})", f"%{final_score*100:.2f}")
                    st.caption(f"Eşik Değeri: %{current_threshold*100:.0f}")

                with c2:
                    st.subheader("Model Olasılıkları")
                    df = pd.DataFrame(list(results.items()), columns=['Model', 'Olasılık'])
                    df['Olasılık'] *= 100
                    
                    fig, ax = plt.subplots(figsize=(6, 2.5))
                    clrs = ['#e74c3c' if x > current_threshold*100 else '#2ecc71' for x in df['Olasılık']]
                    bars = ax.barh(df['Model'].str.upper(), df['Olasılık'], color=clrs)
                    ax.axvline(current_threshold*100, color='black', ls='--')
                    ax.set_xlim(0, 100)
                    for bar in bars:
                        width = bar.get_width()
                        ax.text(width+1, bar.get_y()+0.25, f'%{width:.1f}', fontsize=9)
                    st.pyplot(fig)
                
                # --- SİNYAL GRAFİKLERİ (GÜNCELLENMİŞ) ---
                st.divider()
                st.subheader("🔍 Sinyal ve Öznitelik Analizi")
                st.markdown("CNN modeline girmeden önce ses sinyalinin dönüştürüldüğü matematiksel formlar:")
                with st.expander("Grafikleri Göster", expanded=True):
                    # Scaler'dan geçmiş (Normalize edilmiş) özellikleri çiziyoruz
                    # Çünkü modelin gördüğü asıl veri bu.
                    fig_signal = plot_comprehensive_analysis(y_proc, CONFIG['sr'], feat_scaled)
                    st.pyplot(fig_signal)
                    
                    st.info("""
                    **3. Grafik Açıklaması (154 Boyutlu Vektör):**
                    * **LPS (Mor):** Kodek bozulmalarını yakalar.
                    * **LFCC (Yeşil):** İnsan kulağının duymadığı frekanslardaki detayları yakalar.
                    * **Spectral Contrast (Turuncu):** Sesin "dokusunu" ayırt eder.
                    * **MPE (Kırmızı):** Sinyalin karmaşıklığını (Entropy) ölçer.
                    * **MFCC (Mavi):** Standart konuşma özelliklerini taşır.
                    """)

# --- DİĞER TABLAR ---
with tabs[1]:
    st.header("Grafikler")
    c1, c2 = st.columns(2)
    if os.path.exists("figures/optimizer_comparison_analysis.png"):
        c1.image("figures/optimizer_comparison_analysis.png", caption="Optimizer Karşılaştırması")
    if os.path.exists("figures/gradient_explainability_analysis.png"):
        c2.image("figures/gradient_explainability_analysis.png", caption="Açıklanabilirlik Analizi")

with tabs[2]:
    st.markdown("""
    **Proje Hakkında:**
    Optimizasyon Teorisi dersi kapsamında geliştirilmiştir.
    SGD, Adam, AdamW ve RMSProp algoritmalarının Deepfake tespitindeki performansları incelenmiştir.
    """)