"""
PRODIGY_ML_03 – Cats vs Dogs Classification using SVM
Author: Machine Learning Engineer
Dataset: Kaggle Dogs vs. Cats Dataset (https://www.kaggle.com/c/dogs-vs-cats/data)

Description:
A complete, end-to-end Python pipeline implementing a Support Vector Machine (SVM)
to classify images of cats (0) and dogs (1) using OpenCV preprocessing,
Histogram of Oriented Gradients (HOG) feature extraction, StandardScaler, PCA,
and hyperparameter tuning with GridSearchCV.
"""

import os
import glob
import time
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tqdm import tqdm
import joblib

# Scikit-Image & Scikit-Learn
from skimage.feature import hog
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# -----------------------------------------------------------------------------
# Configuration Constants
# -----------------------------------------------------------------------------
DATA_DIR = './train'        # Path to unzipped Kaggle train directory
SAMPLE_SIZE = 4000          # Balanced sample size (e.g., 2000 cats + 2000 dogs)
IMG_SIZE = 64               # Image resize resolution (64x64)
FEATURE_TYPE = 'hog'        # 'hog' (default) or 'flatten'
USE_PCA = True              # Dimensionality reduction toggle
PCA_VARIANCE = 0.95         # Preserve 95% variance
RANDOM_STATE = 42
TEST_SPLIT = 0.2

# Set random seeds for reproducibility
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
sns.set_theme(style="whitegrid")


# -----------------------------------------------------------------------------
# 1. Dataset Loading & Subsetting
# -----------------------------------------------------------------------------
def load_balanced_dataset(data_dir=DATA_DIR, sample_size=SAMPLE_SIZE):
    """
    Discovers image files in data_dir, extracts balanced samples of cats and dogs,
    and returns a shuffled list of file paths.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Dataset directory '{data_dir}' not found. Please download from Kaggle:\n"
            "  kaggle competitions download -c dogs-vs-cats\n"
            "  unzip dogs-vs-cats.zip && unzip train.zip"
        )

    cat_paths = glob.glob(os.path.join(data_dir, "cat.*.jpg"))
    dog_paths = glob.glob(os.path.join(data_dir, "dog.*.jpg"))

    print(f"Total raw images available: {len(cat_paths):,} Cats, {len(dog_paths):,} Dogs")

    half_sample = sample_size // 2
    selected_cats = random.sample(cat_paths, min(half_sample, len(cat_paths)))
    selected_dogs = random.sample(dog_paths, min(half_sample, len(dog_paths)))

    combined = selected_cats + selected_dogs
    random.shuffle(combined)
    print(f"Selected balanced subset of {len(combined):,} images ({len(selected_cats)} cats, {len(selected_dogs)} dogs).")
    return combined


# -----------------------------------------------------------------------------
# 2. Preprocessing & Feature Extraction
# -----------------------------------------------------------------------------
def extract_single_image_features(image_path, img_size=IMG_SIZE, feature_type=FEATURE_TYPE):
    """
    Reads an image with OpenCV, converts to grayscale, resizes, and extracts features.
    Cat = 0, Dog = 1.
    """
    base_name = os.path.basename(image_path).lower()
    if base_name.startswith('cat'):
        label = 0
    elif base_name.startswith('dog'):
        label = 1
    else:
        return None, None

    # Read image with OpenCV
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        # Corrupted / unreadable image - skip safely
        return None, None

    # Convert to grayscale & resize
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (img_size, img_size), interpolation=cv2.INTER_AREA)

    if feature_type == 'hog':
        # Histogram of Oriented Gradients (HOG)
        features = hog(
            img_resized,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            visualize=False
        )
    else:
        # Flattened normalized raw pixel intensity
        features = (img_resized.flatten() / 255.0).astype(np.float32)

    return features, label


def build_feature_matrix(image_paths, img_size=IMG_SIZE, feature_type=FEATURE_TYPE):
    """
    Iterates through image paths with tqdm progress bar to extract features and labels.
    """
    X_list, y_list = [], []
    skipped_count = 0

    print(f"Extracting {feature_type.upper()} features from {len(image_paths)} images...")
    for path in tqdm(image_paths, desc="Processing Images"):
        feats, label = extract_single_image_features(path, img_size, feature_type)
        if feats is not None and label is not None:
            X_list.append(feats)
            y_list.append(label)
        else:
            skipped_count += 1

    if skipped_count > 0:
        print(f"Safely skipped {skipped_count} corrupted or unreadable images.")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    print(f"Final feature matrix shape: {X.shape}, Labels shape: {y.shape}")
    return X, y


# -----------------------------------------------------------------------------
# 3. Model Training & Kernel Exploration
# -----------------------------------------------------------------------------
def train_and_evaluate_kernels(X_train, y_train, X_test, y_test):
    """
    Compares Linear, RBF, and Polynomial SVM kernels on the training dataset.
    """
    kernels = ['linear', 'rbf', 'poly']
    benchmark_results = {}

    print("\n" + "=" * 60)
    print("           SVM KERNEL BENCHMARK COMPARISON")
    print("=" * 60)

    for k in kernels:
        print(f"Training SVM with '{k}' kernel...")
        start_time = time.time()
        clf = SVC(kernel=k, random_state=RANDOM_STATE)
        clf.fit(X_train, y_train)
        train_time = time.time() - start_time

        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        benchmark_results[k] = {
            'model': clf,
            'accuracy': acc,
            'f1': f1,
            'train_time': train_time
        }
        print(f"  [{k.upper():6s}] Accuracy: {acc * 100:.2f}% | F1: {f1:.4f} | Time: {train_time:.2f}s")

    # Bar chart comparing accuracy
    plt.figure(figsize=(8, 4.5))
    kernel_names = [k.upper() for k in benchmark_results.keys()]
    accuracies = [benchmark_results[k]['accuracy'] * 100 for k in benchmark_results.keys()]
    bars = plt.bar(kernel_names, accuracies, color=['#3b82f6', '#10b981', '#f59e0b'], width=0.5)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.title("Cats vs Dogs Classification Accuracy by SVM Kernel", fontsize=12, fontweight='bold')
    plt.ylim([60, 100])
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., h + 1, f"{h:.2f}%", ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig("svm_kernel_comparison.png", dpi=150)
    plt.show()

    return benchmark_results


def tune_best_kernel(X_train, y_train):
    """
    Uses GridSearchCV (cv=3) to tune C and gamma hyperparameters for RBF kernel.
    """
    print("\nTuning Hyperparameters with GridSearchCV (cv=3)...")
    param_grid = {
        'C': [0.1, 1.0, 10.0],
        'gamma': ['scale', 0.01, 0.001],
        'kernel': ['rbf']
    }

    start = time.time()
    grid_search = GridSearchCV(
        estimator=SVC(random_state=RANDOM_STATE),
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    elapsed = time.time() - start

    print(f"Grid search completed in {elapsed:.2f} seconds.")
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation Score: {grid_search.best_score_ * 100:.2f}%")
    return grid_search.best_estimator_


# -----------------------------------------------------------------------------
# 4. Evaluation & Visualizations
# -----------------------------------------------------------------------------
def evaluate_model(model, X_test, y_test):
    """
    Computes and displays Accuracy, Precision, Recall, F1, Classification Report,
    and Seaborn Confusion Matrix Heatmap.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n" + "=" * 55)
    print("               TEST SET EVALUATION METRICS")
    print("=" * 55)
    print(f"  Accuracy  : {acc * 100:.2f}%")
    print(f"  Precision : {prec * 100:.2f}%")
    print(f"  Recall    : {rec * 100:.2f}%")
    print(f"  F1-Score  : {f1 * 100:.2f}%")
    print("=" * 55)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Cat', 'Dog'], digits=4))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Predicted Cat (0)', 'Predicted Dog (1)'],
        yticklabels=['Actual Cat (0)', 'Actual Dog (1)'],
        cbar=False
    )
    plt.title("Confusion Matrix - Dogs vs Cats SVM", fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()

    return y_pred


def plot_test_image_grid(image_paths, true_labels, pred_labels, num_images=10):
    """
    Displays a 2x5 grid of test images with predicted vs actual labels.
    Green title if prediction is correct, Red title if wrong.
    """
    plt.figure(figsize=(15, 6))
    label_names = {0: 'Cat', 1: 'Dog'}

    for i in range(min(num_images, len(image_paths))):
        img_bgr = cv2.imread(image_paths[i])
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        pred = pred_labels[i]
        actual = true_labels[i]
        is_correct = (pred == actual)
        title_color = '#16a34a' if is_correct else '#dc2626'
        status = "CORRECT" if is_correct else "MISCLASSIFIED"

        plt.subplot(2, 5, i + 1)
        plt.imshow(img_rgb)
        plt.title(
            f"Pred: {label_names[pred]} | True: {label_names[actual]}\n[{status}]",
            color=title_color,
            fontsize=10,
            fontweight='bold'
        )
        plt.axis('off')

    plt.suptitle("SVM Test Image Predictions (Green=Correct, Red=Misclassified)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("test_predictions_grid.png", dpi=150)
    plt.show()


# -----------------------------------------------------------------------------
# 5. Prediction on New Image & Serialization
# -----------------------------------------------------------------------------
def predict_image(image_path, model, scaler, pca=None, feature_type=FEATURE_TYPE, img_size=IMG_SIZE):
    """
    Loads a single test image from disk, applies identical preprocessing and feature extraction,
    and prints/plots the prediction.
    """
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' does not exist.")
        return None

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error: Failed to read image from '{image_path}'.")
        return None

    # Convert to grayscale and resize
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (img_size, img_size), interpolation=cv2.INTER_AREA)

    if feature_type == 'hog':
        feats, hog_vis = hog(
            img_resized,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            visualize=True
        )
    else:
        feats = (img_resized.flatten() / 255.0).astype(np.float32)
        hog_vis = None

    # Standardization & PCA
    feats_scaled = scaler.transform([feats])
    if pca is not None:
        feats_final = pca.transform(feats_scaled)
    else:
        feats_final = feats_scaled

    # Inference
    pred_idx = model.predict(feats_final)[0]
    decision_margin = model.decision_function(feats_final)[0]
    label_name = "Dog" if pred_idx == 1 else "Cat"

    print(f"\n>>> Predicted: {label_name.upper()} (Decision Margin: {decision_margin:+.3f})")

    # Plot visual breakdown
    fig, axes = plt.subplots(1, 3 if hog_vis is not None else 2, figsize=(10, 3.5))
    axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Input Image", fontsize=10)
    axes[0].axis('off')

    axes[1].imshow(img_resized, cmap='gray')
    axes[1].set_title(f"Grayscale ({img_size}x{img_size})", fontsize=10)
    axes[1].axis('off')

    if hog_vis is not None:
        axes[2].imshow(hog_vis, cmap='magma')
        axes[2].set_title("HOG Gradient Map", fontsize=10)
        axes[2].axis('off')

    plt.suptitle(f"Prediction: {label_name} | Margin: {decision_margin:+.2f}", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return label_name


# -----------------------------------------------------------------------------
# Main Execution Pipeline
# -----------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  PRODIGY_ML_03: Cats vs Dogs Classification using SVM")
    print("=" * 60)

    # 1. Load Data Paths
    image_paths = load_balanced_dataset(DATA_DIR, SAMPLE_SIZE)

    # 2. Extract Features
    X, y = build_feature_matrix(image_paths, IMG_SIZE, FEATURE_TYPE)

    # 3. Train/Test Stratified Split (80/20)
    X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
        X, y, image_paths, test_size=TEST_SPLIT, stratify=y, random_state=RANDOM_STATE
    )

    # 4. Standardize Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Optional PCA (Preserve 95% variance)
    if USE_PCA:
        pca = PCA(n_components=PCA_VARIANCE, random_state=RANDOM_STATE)
        X_train_final = pca.fit_transform(X_train_scaled)
        X_test_final = pca.transform(X_test_scaled)
        print(f"PCA reduced dimensions from {X.shape[1]} to {pca.n_components_} features.")
    else:
        pca = None
        X_train_final = X_train_scaled
        X_test_final = X_test_scaled

    # 6. Benchmark Kernels
    kernel_results = train_and_evaluate_kernels(X_train_final, y_train, X_test_final, y_test)

    # 7. Hyperparameter Tuning on Best Kernel (RBF)
    best_model = tune_best_kernel(X_train_final, y_train)

    # 8. Final Evaluation & Confusion Matrix
    y_pred = evaluate_model(best_model, X_test_final, y_test)

    # 9. 10-Image Visual Test Grid
    plot_test_image_grid(paths_test, y_test, y_pred, num_images=10)

    # 10. Save Model Artifacts
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(best_model, "artifacts/svm_cats_dogs_model.joblib")
    joblib.dump(scaler, "artifacts/scaler.joblib")
    if pca is not None:
        joblib.dump(pca, "artifacts/pca.joblib")
    print("\nModel artifacts saved successfully in ./artifacts/")

    # 11. Test prediction on a sample
    if len(paths_test) > 0:
        predict_image(paths_test[0], best_model, scaler, pca, FEATURE_TYPE, IMG_SIZE)


if __name__ == '__main__':
    main()
