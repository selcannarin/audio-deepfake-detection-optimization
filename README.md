# Audio Deepfake Detection – Optimization-Aware Deep Learning System

## Overview
This project addresses the problem of **audio deepfake detection** from an **optimization theory perspective**, rather than treating it solely as a model architecture problem. While most existing studies focus on network design or feature engineering, this work systematically analyzes how different **gradient-based optimization algorithms** affect convergence behavior, generalization, robustness, and real-world performance.

The study is implemented as an **end-to-end system** consisting of both:
1. A **training & analysis module** for controlled experiments and optimization analysis.
2. A **real-time web application** that demonstrates deployment-ready inference under domain shift.

---

## Key Contributions
- Systematic optimizer comparison (**SGD, Adam, AdamW, RMSProp**) under fixed architecture and data.
- **154-dimensional robust feature representation** combining spectral, cepstral, and temporal-complexity features.
- Gradient dynamics analysis via gradient norm evolution and loss landscape behavior.
- Identification and quantitative analysis of **domain shift** and **inverted learning**.
- **Weighted ensemble strategy** to mitigate generalization collapse in real-world data.
- **Gradient-based explainability** and **adversarial robustness analysis (FGSM)**.
- Deployment of trained models in an **interactive Streamlit web application**.

---

## System Architecture
The project consists of two main modules:

### 1. Training & Analysis Module (Google Colab)
This module focuses on controlled experimentation and optimization analysis.

- Improved 1D-CNN architecture for audio classification.
- Optimizer comparison: **SGD, Adam, AdamW, RMSProp**.
- Fixed random seed (`seed=42`) for reproducibility.
- Gradient norm tracking and convergence analysis.
- Threshold optimization and probability calibration.
- Domain shift evaluation between laboratory and real-world data.
- Ensemble learning across optimizer-specific models.

### 2. Real-Time Detection Web Application (Streamlit)
This module demonstrates how trained models can be deployed in practice.

- Local inference with pretrained models.
- Interactive audio upload and prediction.
- Optimizer-aware model comparison.
- Domain shift correction through ensemble weighting.
- Designed to simulate **real-world usage conditions**.

---

## Datasets
Due to size constraints, datasets are **not included** in this repository.

- **ASVspoof 2019 LA** – laboratory-controlled benchmark.
- **In-the-Wild Audio Deepfake Dataset** – real-world distribution shift evaluation.

**Dataset sources:**
- [ASVspoof 2019 LA (HuggingFace)](https://huggingface.co/datasets/Bisher/ASVspoof_2019_LA)
- [In-the-Wild Dataset (HuggingFace)](https://huggingface.co/datasets/UncovAI/InTheWild)

---

## Feature Engineering
A robust **154-dimensional hybrid feature vector** is constructed:

- **Log Power Spectrum (LPS)** – 40 features
- **Linear Frequency Cepstral Coefficients (LFCC)** – 40 features
- **Mel-Frequency Cepstral Coefficients (MFCC)** – 40 features
- **Spectral Contrast** – 14 features
- **Multi-scale Permutation Entropy (MPE)** – 20 features

MPE is integrated to capture **temporal complexity** information that classical cepstral features fail to represent.

---

## Model Architecture
- **ImprovedCNN (1D-CNN)** architecture.
- Architecture held constant to isolate optimizer behavior.
- Preliminary experiments with ResNet1D and SAM optimizer were conducted but excluded due to computational constraints.

---

## Experimental Results

### Laboratory Performance (ASVspoof 2019 LA)
- **AdamW** achieved the most stable convergence.
- Mean gradient norm: **0.0167**.
- Convergence around **epoch 19**.
- **ROC-AUC: 99.56%**.
- Accuracy improved from **66.50% → 95.33%** via threshold optimization.

### Real-World Performance & Domain Shift (In-the-Wild)
- AdamW exhibited severe generalization collapse (**42% accuracy**).
- Clear evidence of **domain shift** and **inverted learning**.
- Optimizer-specific minima demonstrated different robustness properties.

### Ensemble Learning
- Weighted ensemble of optimizer-specific models.
- Improved real-world accuracy to **65%**.
- Demonstrated that combining diverse optimization minima improves robustness.

---

## Explainability & Robustness

### Gradient-Based Explainability
- Saliency maps and gradient-based attribution.
- Feature-level decision analysis consistent with optimization behavior.

### Adversarial Robustness (FGSM)
- Significant accuracy degradation under small perturbations.
- AdamW solutions identified as **sharp minima**.
- Results motivate adversarial training and sharpness-aware optimization.

---

## Repository Structure

```text
audio-deepfake-detection-optimization/
│
├── README.md
├── requirements.txt
│
├── 1_Training_Analysis_Colab/
│   ├── DeepfakeAudioDetector.py
│   ├── audioDeepfake.txt
│   └── reports/
│
├── 2_Streamlit_App_Local/
│   ├── app.py
│   └── utils.py
│
├── models/
│   ├── cnn_adamw_model.pth
│   ├── cnn_adam_model.pth
│   ├── cnn_sgd_model.pth
│   ├── cnn_rmsprop_model.pth
│   └── scaler.pkl
│
└── figures/
    ├── optimizer_comparison.png
    ├── gradient_explainability.png
    └── ensemble_roc.png
```

---

## How to Run the Web Application
```bash
pip install -r requirements.txt
streamlit run 2_Streamlit_App_Local/app.py
```

---

## Project Report
A detailed academic report covering methodology, optimization theory, experiments, and analysis is available in the report/ directory.

---

## Notes
This project was developed as part of M.Sc. coursework in Computer Engineering (Optimization Theory).
The goal was not only to achieve high accuracy, but to understand and explain why models fail under real-world conditions.
