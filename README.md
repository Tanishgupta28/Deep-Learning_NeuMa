<div align="center">

<img src="https://img.shields.io/badge/Institution-TIET%20Patiala-cc0000?style=for-the-badge&logoColor=white" />
<img src="https://img.shields.io/badge/Domain-Neuromarketing%20%7C%20Deep%20Learning-0057b7?style=for-the-badge" />
<img src="https://img.shields.io/badge/Status-Completed-28a745?style=for-the-badge" />
<img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />

<br/><br/>

#  User Purchase Prediction
### Using EEG, Eye-Tracking Signals & Numerical Data Analysis

*A Deep Learning + Machine Learning project at the intersection of Neuroscience and AI*

**Thapar Institute of Engineering and Technology, Patiala**  
*BE Third Year - Computer Science & Engineering | May 2026*

---

</div>

##  Team

| Name | Roll No. |
|------|----------|
| Tanish Gupta | 102316041 |
| Kashish Rana | 102316021 |
| Swastik | 102316020 |
| Achin Agarwal | 102316011 |

**Submitted To:** Dr. Stuti Chugh

---

##  Overview

Traditional purchase prediction models rely on clickstream data and demographic profiles — they record the *what* but miss the *why*. This project bridges that gap by tapping directly into consumers' **neurological and physiological responses** during product evaluation.

We solve a **binary classification problem**: *Will a user purchase a product?* — using two independent yet complementary pipelines:

1. **Deep Learning Pipeline** — Bidirectional GRU with Attention mechanism on raw EEG + Eye-Tracking signals
2. **Machine Learning Pipeline** — XGBoost, CatBoost, and AdaBoost on structured tabular/numerical features

> **Dataset:** [NeuMa Neuromarketing Dataset](https://doi.org/10.3389/fncom.2024.1516440) (Georgiadis et al., 2023) — 42 participants browsing FMCG supermarket brochures with simultaneous EEG and eye-tracking recording.

---

##  Key Results

<div align="center">

### Deep Learning Model (BiGRU + Attention)

| Metric | Value |
|--------|-------|
| **Accuracy** | **82.6%** |
| **ROC-AUC** | **0.79** |
| **F1 Score** | 0.48 (minority class) |
| **Input Modality** | EEG (19 ch) + Eye Tracking (6 ch) |

### Best ML Model (XGBoost)

| Metric | Value |
|--------|-------|
| **Accuracy** | **77.03%** |
| **Precision** | **0.8333** |
| **Recall** | 0.7558 |
| **F1 Score** | **0.7927** |
| **ROC-AUC** | **0.8791** |

</div>

>  The BiGRU + Attention model achieves the **highest accuracy among all DL baselines** tested (outperforming LSTM, Vanilla GRU, CNN+LSTM, EEGNet, Transformer Encoder). The lower F1 reflects inherent class imbalance in physiological datasets, addressed using weighted BCE loss.

---

## Project Structure

```
 purchase-prediction-eeg
├──  deep_learning/
│   ├── bigru_attention_model.py     # BiGRU + Attention architecture
│   ├── data_loader.py               # EEG + ET data loading & preprocessing
│   ├── train.py                     # Training loop with early stopping
│   └── evaluate.py                  # Threshold optimization & metrics
│
├──  machine_learning/
│   ├── preprocess_numerical.py      # Tabular data preprocessing & SMOTE
│   ├── xgboost_model.py             # XGBoost with regularization
│   ├── catboost_model.py            # CatBoost classifier
│   ├── adaboost_model.py            # AdaBoost baseline
│   └── evaluate_ml.py               # Cross-validation & metrics
│
├──  visualizations/
│   ├── loss_curves.py               # Training vs. Validation loss plots
│   ├── confusion_matrices.py        # Confusion matrix plots
│   ├── roc_curves.py                # ROC-AUC curves
│   ├── feature_importance.py        # SHAP / gain feature plots
│   └── smote_visualization.py       # PCA-projected SMOTE plots
│
├──  data/                         # NeuMa dataset (.mat files) — not included
├── requirements.txt
└── README.md
```

---

##  Methodology

### Part 1 — Deep Learning: BiGRU with Attention

```
Input: (sequence_length × 25 features)
  ├── 19 EEG channels (300 Hz)
  └── 6 Eye-Tracking channels (gaze, pupil dilation, saccades)

Architecture:
  Input → BiGRU Layer 1 (hidden=32, bidir → 64) 
        → BiGRU Layer 2 
        → Temporal Attention (softmax over timesteps)
        → Context Vector (64-dim)
        → FC (64→32) → ReLU → Dropout(0.4)
        → FC (32→1) → Sigmoid
        → Purchase Probability
```

**Key Design Choices:**
-  **Bidirectional GRU** — captures both temporal lead-up and aftermath of a fixation event
-  **Attention Mechanism** — focuses on critical fixation moments tied to purchase intent
-  **BCEWithLogitsLoss + pos_weight** — directly counteracts ~6:1 class imbalance
-  **Early Stopping** (patience=10) + Gradient Clipping (max norm=1.0)
-  **Subject-Disjoint Splits** — no participant appears in both train and test sets

---

### Part 2 — Machine Learning: Gradient Boosting Ensemble

**Dataset:** 737 samples × 11 features | Binary target: `bought ∈ {0, 1}`

| Model | Description |
|-------|-------------|
| **XGBoost** | L1+L2 regularized gradient boosting; column subsampling |
| **CatBoost** | Ordered boosting with oblivious trees; native categorical handling |
| **AdaBoost** | Iterative re-weighting of misclassified samples (baseline) |

**Anti-Overfitting Strategy:**
- ✅ SMOTE oversampling: 429/308 → 343/343 (balanced)
- ✅ Stratified 5-fold cross-validation
- ✅ Early stopping on validation performance
- ✅ L1 + L2 regularization + shallow tree depth constraints
- ✅ Row/column subsampling per tree

---

##  Full Model Comparison

### Deep Learning Baselines (EEG/Physiological Stream)

| Model | Input | Accuracy (%) | F1 | ROC-AUC |
|-------|-------|--------------|----|---------|
| LSTM | EEG | 74.3 | 0.71 | 0.76 |
| Vanilla GRU | EEG | 76.1 | 0.73 | 0.77 |
| CNN + LSTM | EEG | 78.0 | 0.74 | 0.78 |
| EEGNet | EEG | 75.5 | 0.72 | 0.75 |
| Transformer Encoder | EEG | 80.2 | 0.76 | 0.80 |
| CNN + Eye-Tracking | EEG + ET | 79.0 | 0.75 | 0.78 |
| Bi-LSTM | EEG | 77.8 | 0.74 | 0.77 |
| **Bi-GRU + Attention** ⭐ | **EEG + ET** | **82.6** | **0.48*** | **0.79** |

*\*Lower F1 reflects class imbalance in physiological data; addressed via weighted BCE loss.*

### Machine Learning Baselines (Tabular Stream)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression | 0.6950 | 0.7210 | 0.7100 | 0.7154 | 0.7800 |
| Decision Tree | 0.6720 | 0.6900 | 0.6800 | 0.6850 | 0.7100 |
| Random Forest | 0.7400 | 0.7800 | 0.7300 | 0.7542 | 0.8400 |
| SVM (RBF) | 0.7100 | 0.7500 | 0.7200 | 0.7347 | 0.8100 |
| Gradient Boosting | 0.7500 | 0.8000 | 0.7400 | 0.7689 | 0.8600 |
| LightGBM | 0.7550 | 0.8100 | 0.7450 | 0.7762 | 0.8650 |
| **XGBoost** ⭐ | **0.7703** | **0.8333** | **0.7558** | **0.7927** | **0.8791** |
| CatBoost | 0.7600 | — | — | 0.7673 | **0.8794** |
| AdaBoost | 0.7550 | — | — | 0.7673 | 0.8490 |

---

##  Installation & Setup

### Prerequisites

```bash
Python 3.8+
CUDA-enabled GPU (recommended for deep learning)
```

### Install Dependencies

```bash
git clone https://github.com/your-username/purchase-prediction-eeg.git
cd purchase-prediction-eeg
pip install -r requirements.txt
```

### Requirements

```txt
torch>=2.0.0
scikit-learn>=1.3.0
xgboost>=2.0.0
catboost>=1.2.0
imbalanced-learn>=0.11.0
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

---

##  Usage

### Deep Learning Pipeline

```python
# 1. Preprocess EEG + Eye Tracking data
python deep_learning/data_loader.py --data_dir ./data/NeuMa/

# 2. Train BiGRU + Attention model
python deep_learning/train.py \
    --epochs 50 \
    --batch_size 64 \
    --lr 0.0005 \
    --hidden_size 32 \
    --dropout 0.4 \
    --patience 10

# 3. Evaluate with threshold optimization
python deep_learning/evaluate.py --threshold_search 0.30 0.75
```

### Machine Learning Pipeline

```python
# 1. Preprocess numerical data + apply SMOTE
python machine_learning/preprocess_numerical.py

# 2. Train and evaluate all models
python machine_learning/xgboost_model.py
python machine_learning/catboost_model.py
python machine_learning/adaboost_model.py
```

### Generate Visualizations

```python
python visualizations/loss_curves.py
python visualizations/roc_curves.py
python visualizations/feature_importance.py
python visualizations/smote_visualization.py
```

---

##  Hyperparameters

| Parameter | BiGRU (Deep Learning) | XGBoost / CatBoost |
|-----------|----------------------|-------------------|
| Learning Rate | 0.0005 (Adam) | 0.05–0.1 |
| Batch Size | 64 | N/A |
| Epochs / Estimators | 50 (early stopping) | 300–500 trees |
| Hidden Size | 32 × 2 (bidir) = 64 | N/A |
| Dropout | 0.4 | N/A |
| Regularization | Gradient clipping (1.0) | L1 + L2 (α, λ) |
| Cross-Validation | — | 5-fold stratified |
| Class Imbalance | BCELoss pos_weight | SMOTE + scale_pos_weight |
| Early Stopping | patience = 10 (val loss) | patience = 20 rounds |

---

##  Key Features of This Work

- **Neuromarketing meets AI** — bypass self-reported surveys; use raw neural signals
- **Multimodal fusion** — EEG (temporal brain signals) + Eye Tracking (visual attention) as a unified 25-channel input
- **Temporal attention** — model learns *which moments* during product fixation are most predictive
- **Subject-disjoint evaluation** — ensures cross-subject generalization, not subject memorization
- **Class imbalance handling** — weighted BCE loss (DL) + SMOTE (ML)
- **Dual-stream comparison** — first systematic comparison of DL (physiological) vs ML (tabular) on NeuMa dataset

---

## 📉 Limitations & Future Work

**Current Limitations:**
- Small dataset (42 participants) limits population-level generalization
- EEG signals are highly subject-specific → cross-subject generalization is fundamentally hard
- Real-time EEG acquisition has high hardware cost for practical deployment

**Future Directions:**
- 🔗 **Multimodal end-to-end fusion** — combine EEG, Eye Tracking, and numerical features in one unified model
- 🤖 **Transformer for EEG** — multi-head self-attention for global temporal context
- 🔓 **Self-supervised pre-training** on unlabeled EEG to improve minority-class representation
- 👤 **Personalized fine-tuning** — subject-adaptive models to handle inter-individual neural variability
- 📦 **Larger datasets** — more participants + product categories for better generalization

---

## References

1. Usman et al., *"Multimodal consumer choice prediction using EEG signals and eye tracking"*, Frontiers in Computational Neuroscience, 2025.
2. Afshar & Azimi, *"EEG-Based Consumer Behaviour Prediction: Classical ML to GNNs"*, arXiv, 2025.
3. Mashrur et al., *"Intelligent neuromarketing framework for consumers' preference prediction"*, Journal of Consumer Behaviour, 2024.
4. *Hybrid EEG + gaze neuromarketing approach*, Brain Informatics, 2025.
5. Fang et al., *"MB-MSTFNet: Multi-Band Spatio-Temporal Attention Network for EEG"*, Sensors, 2025.
6. Baig et al., *"Hybrid CNN-GRU Models for EEG Motor Imagery Classification"*, Sensors, 2025.
7. Abdelhamid & Desai, *"Comprehensive Benchmark of ML/DL Across Tabular Datasets"*, arXiv, 2024.
8. *Implementation and Performance Comparison of Gradient Boosting Algorithms*, Springer LNNS, 2024.
9. Abdelhamid & Desai, *"Balancing the Scales: Class Imbalance in Binary Classification"*, arXiv, 2024.
10. Elreedy et al., *"Theoretical distribution analysis of SMOTE for imbalanced learning"*, Machine Learning, 2024.

---

<div align="center">


*Department of Computer Science & Engineering, Patiala, Punjab, India*

</div>
