import numpy as np
import pickle
import os
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA

models = "artifacts"

# Load the KNN-specific raw features
X_TRAIN_PATH = f"{models}/knn_X_train.npy"
X_TEST_PATH  = f"{models}/knn_X_test.npy"

# Save processed features
X_TRAIN_PROCESSED_PATH = f"{models}/knn_X_train_processed.npy"
X_TEST_PROCESSED_PATH  = f"{models}/knn_X_test_processed.npy"

# Save pipeline objects
SCALER_PATH = f"{models}/knn_scaler.pkl"
PCA_PATH    = f"{models}/knn_pca.pkl"

print("Loading KNN raw features...")
X_train = np.load(X_TRAIN_PATH)
X_test  = np.load(X_TEST_PATH)

# 1. L2 NORMALIZATION (Better for VGG+KNN than StandardScaler)
print("Applying L2 Normalization...")
scaler = Normalizer(norm='l2')
X_train_norm = scaler.fit_transform(X_train)
X_test_norm  = scaler.transform(X_test)

# 2. PCA (Reduces noise and dimensions, e.g., 512 -> ~80)
print("Applying PCA...")
pca = PCA(n_components=0.95) # Keep 95% variance
X_train_final = pca.fit_transform(X_train_norm)
X_test_final  = pca.transform(X_test_norm)

print(f"Original shape: {X_train.shape}")
print(f"Processed shape: {X_train_final.shape}")

# Save Everything
np.save(X_TRAIN_PROCESSED_PATH, X_train_final)
np.save(X_TEST_PROCESSED_PATH, X_test_final)

with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)
with open(PCA_PATH, "wb") as f:
    pickle.dump(pca, f)
