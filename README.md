# PRODIGY_ML_03 – Cats vs Dogs Image Classification using SVM

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/patelramlala414/PRODIGY_ML_03_Cats_vs_Dogs_SVM/blob/main/PRODIGY_ML_03_Cats_vs_Dogs_SVM.ipynb)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning project implementing a **Support Vector Machine (SVM)** to classify images of cats and dogs using the Kaggle **Dogs vs. Cats** competition dataset.

The pipeline incorporates computer vision preprocessing with **OpenCV**, feature extraction via **Histogram of Oriented Gradients (HOG)**, feature scaling with **StandardScaler**, dimensionality reduction via **Principal Component Analysis (PCA)**, and hyperparameter tuning with **GridSearchCV**.

---

## 📌 Table of Contents
1. [Project Overview](#-project-overview)
2. [Dataset Details](#-dataset-details)
3. [Methodology & Architecture](#-methodology--architecture)
4. [Tech Stack](#-tech-stack)
5. [Installation & Setup](#-installation--setup)
6. [How to Run](#-how-to-run)
7. [Results & Benchmark](#-results--benchmark)
8. [Project Structure](#-project-structure)
9. [Future Enhancements](#-future-enhancements)

---

## 🔍 Project Overview

Support Vector Machines (SVMs) are effective high-margin discriminative classifiers in high-dimensional feature spaces. However, raw pixel arrays suffer from variance to lighting and pose. This project demonstrates how pairing **HOG feature descriptors** with an **RBF-kernel SVM** achieves **84.25% test accuracy**, overcoming raw pixel limitations without requiring deep neural networks.

### Key Highlights
- **Balanced Subsetting**: Configurable `SAMPLE_SIZE = 4,000` (2,000 cats and 2,000 dogs) to prevent computational bottlenecks ($\mathcal{O}(N^2)$ to $\mathcal{O}(N^3)$).
- **Dual Feature Extraction**: Toggle between **HOG (Histogram of Oriented Gradients)** (84.25% accuracy) and **Flattened Raw Pixels** (62.40% accuracy).
- **PCA Dimensionality Reduction**: Retains 95% cumulative explained variance while compressing the feature vector from 1,764 to 142 components (~12x acceleration).
- **Kernel Comparison**: Benchmark across **Linear**, **RBF**, and **Polynomial** kernels.
- **Visual Validation**: 10-image test grid with color-coded classification feedback.

---

## 📊 Dataset Details

- **Source**: [Kaggle Dogs vs. Cats Dataset](https://www.kaggle.com/c/dogs-vs-cats/data)
- **Total Images**: 25,000 RGB images (12,500 cats, 12,500 dogs)
- **Naming Convention**: `cat.0.jpg` (Label = 0), `dog.0.jpg` (Label = 1)
- **Working Sample**: 4,000 balanced images (80% train / 20% test split, stratified)

---

## ⚙️ Methodology & Architecture

```text
[Raw Image] 
     │
     ▼
[OpenCV Preprocessing] ──► Grayscale Conversion & Resize (64x64)
     │
     ▼
[Feature Extraction]   ──► HOG (9 bins, 8x8 cell, 2x2 block) -> 1,764 features
     │
     ▼
[StandardScaler]       ──► Zero Mean & Unit Variance Normalization
     │
     ▼
[PCA (95% Variance)]   ──► 1,764 dims ──► 142 Principal Components
     │
     ▼
[SVM (RBF Kernel)]     ──► Hyperplane Optimization (C=1.0, gamma='scale')
     │
     ▼
[Prediction]           ──► Cat (0) or Dog (1) with Decision Margin
```

---

## 🛠 Tech Stack

- **Core Language**: Python 3.8+
- **Computer Vision**: OpenCV (`cv2`), Scikit-Image (`skimage.feature.hog`)
- **Machine Learning**: Scikit-Learn (`SVC`, `GridSearchCV`, `StandardScaler`, `PCA`)
- **Data & Numerical**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Utilities**: `tqdm` (progress tracking), `joblib` (model serialization)

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/patelramlala414/PRODIGY_ML_03_Cats_vs_Dogs_SVM.git
cd PRODIGY_ML_03_Cats_vs_Dogs_SVM
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download Dataset from Kaggle
Place your `kaggle.json` API token in `~/.kaggle/` and run:
```bash
kaggle competitions download -c dogs-vs-cats
unzip -q dogs-vs-cats.zip
unzip -q train.zip
```

---

## 💻 How to Run

### Run in Google Colab (1-Click)
Click the badge above or navigate to:
`https://colab.research.google.com/github/patelramlala414/PRODIGY_ML_03_Cats_vs_Dogs_SVM/blob/main/PRODIGY_ML_03_Cats_vs_Dogs_SVM.ipynb`

### Run the Standalone Script
```bash
python prodigy_ml_03_svm.py
```

### Run Locally with Jupyter Lab
```bash
jupyter lab PRODIGY_ML_03_Cats_vs_Dogs_SVM.ipynb
```

### Inference on a New Image
```python
import joblib
from prodigy_ml_03_svm import predict_image

# Load artifacts
model = joblib.load('artifacts/svm_cats_dogs_model.joblib')
scaler = joblib.load('artifacts/scaler.joblib')
pca = joblib.load('artifacts/pca.joblib')

# Predict
predict_image('path_to_my_pet.jpg', model, scaler, pca)
```

---

## 📈 Results & Benchmark

| Method | Feature Dimension | Kernel | Test Accuracy | F1-Score | Training Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Raw Pixels** | 4,096 | Linear | 62.40% | 0.621 | 18.2s |
| **HOG (No PCA)** | 1,764 | Linear | 78.12% | 0.779 | 5.1s |
| **HOG + PCA** | 142 (95% var) | Poly (d=3) | 80.40% | 0.802 | 14.8s |
| **HOG + PCA** | **142 (95% var)** | **RBF (Tuned)** | **84.25%** | **0.844** | **8.4s** |

---

## 📁 Project Structure

```text
PRODIGY_ML_03-Cats-vs-Dogs-SVM/
├── PRODIGY_ML_03_Cats_vs_Dogs_SVM.ipynb   # Complete Colab / Jupyter Notebook
├── prodigy_ml_03_svm.py                   # Standalone Python script
├── README.md                              # Project documentation
├── requirements.txt                       # Python dependencies
├── kaggle.json.example                    # Kaggle API credential template
└── artifacts/
    ├── svm_cats_dogs_model.joblib         # Trained SVM model
    ├── scaler.joblib                      # Fitted StandardScaler
    └── pca.joblib                         # Fitted PCA transformer
```

---

## 🔮 Future Enhancements

1. **Deep Feature Extraction**: Using pre-trained CNNs (e.g. ResNet50 or MobileNetV2) as feature extractors before SVM classification.
2. **Data Augmentation**: Introducing random horizontal flips and rotations during training to improve invariance.
3. **Linear SVM Approximation**: Using `LinearSVC` with Nystroem kernel approximations to scale to the full 25,000 images in seconds.
