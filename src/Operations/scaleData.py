import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler

ARTIFACTS_DIR = "artifacts"

X_TRAIN_PATH = f"{ARTIFACTS_DIR}/X_train.npy"
Y_TRAIN_PATH = f"{ARTIFACTS_DIR}/y_train.npy"
X_TEST_PATH  = f"{ARTIFACTS_DIR}/X_test.npy"
Y_TEST_PATH  = f"{ARTIFACTS_DIR}/y_test.npy"
SCALER_PATH  = f"{ARTIFACTS_DIR}/scaler.pkl"
X_TRAIN_SCALED_PATH = f"{ARTIFACTS_DIR}/X_train_scaled.npy"
X_TEST_SCALED_PATH  = f"{ARTIFACTS_DIR}/X_test_scaled.npy"

# ================== LOAD FEATURES ==================
X_train = np.load(X_TRAIN_PATH)
y_train = np.load(Y_TRAIN_PATH)
X_test  = np.load(X_TEST_PATH)
y_test  = np.load(Y_TEST_PATH)

print("Loaded cached features:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test :", X_test.shape)
print("y_test :", y_test.shape)

# ================== LOAD OR FIT SCALER ==================
if os.path.exists(SCALER_PATH) and os.path.exists(X_TRAIN_SCALED_PATH) and os.path.exists(X_TEST_SCALED_PATH):
    print("Cached scaled features found. Loading...")
    X_train_scaled = np.load(X_TRAIN_SCALED_PATH)
    X_test_scaled  = np.load(X_TEST_SCALED_PATH)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
else:
    print("No cached scaled features. Scaling now...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Save scaler and scaled features
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    np.save(X_TRAIN_SCALED_PATH, X_train_scaled)
    np.save(X_TEST_SCALED_PATH, X_test_scaled)
    print("Scaled features cached successfully!")

# ================== FINAL CHECK ==================
print("\nFinal data ready for ML:")
print("X_train_scaled:", X_train_scaled.shape)
print("X_test_scaled :", X_test_scaled.shape)