import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import os

ARTIFACTS_DIR = "artifacts"
X_TRAIN_SCALED_PATH = f"{ARTIFACTS_DIR}/X_train_scaled.npy"
Y_TRAIN_PATH = f"{ARTIFACTS_DIR}/y_train.npy"
X_TEST_SCALED_PATH  = f"{ARTIFACTS_DIR}/X_test_scaled.npy"
Y_TEST_PATH  = f"{ARTIFACTS_DIR}/y_test.npy"
# 
LE_PATH = f"{ARTIFACTS_DIR}/label_encoder.pkl"
KNN_MODEL_PATH = f"{ARTIFACTS_DIR}/knn_model.pkl"

# ================== LOAD DATA ==================
X_train_scaled = np.load(X_TRAIN_SCALED_PATH)
y_train = np.load(Y_TRAIN_PATH)
X_test_scaled = np.load(X_TEST_SCALED_PATH)
y_test = np.load(Y_TEST_PATH)

# ================== GRID SEARCH FOR BEST HYPERPARAMS ==================
# Load label encoder
with open(LE_PATH, "rb") as f:
    le = pickle.load(f)


# ================== TRAIN KNN WITH BEST PARAMS ==================
print("Training KNN model...")
best_knn = KNeighborsClassifier(n_neighbors=4, metric='manhattan', weights='distance')
best_knn.fit(X_train_scaled, y_train)

# ================== EVALUATE ==================
# y_pred = best_knn.predict(X_test_scaled)
CONF_THRESH = 0.44

y_pred_prob = best_knn.predict_proba(X_test_scaled)

y_pred = [np.argmax(p) if np.max(p) >= CONF_THRESH else len(le.classes_) for p in y_pred_prob]

accuracy = accuracy_score(y_test, y_pred)
n_unknown = y_pred.count(len(le.classes_))
rejection_rate = (n_unknown / len(y_test)) * 100
print(f"KNN Test Accuracy: {accuracy:.4f}")
print(f"KNN Rejection Rate: {rejection_rate:.2f}%")

print("Test Accuracy :", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


# Save model
with open(KNN_MODEL_PATH, "wb") as f:
    pickle.dump(best_knn, f)

print("KNN model saved!")