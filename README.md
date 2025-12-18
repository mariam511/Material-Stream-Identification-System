# Material Stream Identification System

An automated system for identifying material streams using feature extraction, data augmentation, and machine learning classifiers (SVM and k-NN).

---

## 🚀 Project Pipeline

The project follows a structured machine learning workflow, divided into a core data preparation phase followed by algorithm-specific implementations.

### 🛠 Phase 1: Data Preparation
1.  **Load & Clean:** Initial data ingestion and removal of noise or corrupted files.
2.  **Preprocessing:** Data normalization and formatting for model readiness.
3.  **Train-Test Split:** Partitioning the dataset into training and evaluation sets.
4.  **Data Augmentation:** Expanding the dataset to improve model generalization.

---

## 🧠 Model Workflows



### 🔹 k-Nearest Neighbors (k-NN)
Workflow designed for distance-based classification:
* **Step 1:** `feature_extraction_KNN` — Extracting relevant spatial or color features.
* **Step 2:** `scaleData_KNN` — Standardizing data to ensure equal feature weighting.
* **Step 3:** `knn_train` — Training the k-NN classifier.
* **Step 4:** `camera_knn` — Real-time identification via live camera feed.

### 🔸 Support Vector Machine (SVM)
Workflow optimized for high-dimensional boundary classification:
* **Step 1:** `feature_extraction_SVM` — Extracting features tailored for hyperplane separation.
* **Step 2:** `scale_data_SVM` — Feature scaling for optimal SVM convergence.
* **Step 3:** `svm_train` — Training the SVM model.
* **Step 4:** `camera_svm` — Real-time identification via live camera feed.

---

## 📊 Summary Table

| Stage | k-NN Path | SVM Path |
| :--- | :--- | :--- |
| **Features** | `feature_extraction_KNN` | `feature_extraction_SVM` |
| **Scaling** | `scaleData_KNN` | `scale_data_SVM` |
| **Training** | `knn_train` | `svm_train` |
| **Inference** | `camera_knn` | `camera_svm` |

> **Note:** The preprocessing and augmentation steps are shared across both models to ensure a fair comparison of performance.
