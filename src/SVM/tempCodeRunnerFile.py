import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score , classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import os

print("Training SVM model...")
ARTIFACTS_DIR = "artifacts"
X_TRAIN_PATH = f"{ARTIFACTS_DIR}/X_train_scaled.npy"
Y_TRAIN_PATH = f"{ARTIFACTS_DIR}/y_train.npy"
X_TEST_PATH  = f"{ARTIFACTS_DIR}/X_test_scaled.npy"
Y_TEST_PATH  = f"{ARTIFACTS_DIR}/y_test.npy"

LE_PATH = f"{ARTIFACTS_DIR}/label_encoder.pkl"
SVM_MODEL_PATH = f"{ARTIFACTS_DIR}/svm_model.pkl"
print("Loading data...")
# Load data
X_train_scaled = np.load(X_TRAIN_PATH)
y_train = np.load(Y_TRAIN_PATH)
X_test_scaled  = np.load(X_TEST_PATH)
y_test  = np.load(Y_TEST_PATH)

# Load label encoder
with open(LE_PATH, "rb") as f:
    le = pickle.load(f)
print("Data loaded.")
# Train SVM
print("Fitting SVM model...")
svm_model = SVC(C=100,gamma='scale'   kernel='rbf', probability=True ,class_weight='balanced')
svm_model.fit(X_train_scaled, y_train)

# Evaluate with unknown class handling
CONF_THRESH = 0.44

y_pred_prob = svm_model.predict_proba(X_test_scaled)
y_pred = [np.argmax(p) if np.max(p) >= CONF_THRESH else len(le.classes_) for p in y_pred_prob]

accuracy = accuracy_score(y_test, y_pred)
n_unknown = y_pred.count(len(le.classes_))
rejection_rate = (n_unknown / len(y_test)) * 100

print(f"SVM Test Accuracy: {accuracy:.4f}")
print(f"SVM Rejection Rate: {rejection_rate:.2f}%")

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


# Save model
with open(SVM_MODEL_PATH, "wb") as f:
    pickle.dump(svm_model, f)

print("SVM model saved!")
