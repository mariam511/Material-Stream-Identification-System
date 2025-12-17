import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

models = "artifacts"
# Load the PROCESSED KNN data
x_train_path = f"{models}/knn_X_train_processed.npy"
y_tain_path = f"{models}/knn_y_train.npy"
X_TEST_PATH  = f"{models}/knn_X_test_processed.npy"
Y_TEST_PATH  = f"{models}/knn_y_test.npy"
LE_PATH      = f"{models}/knn_label_encoder.pkl"
KNN_MODEL_PATH = f"{models}/knn_model.pkl"

X_train = np.load(x_train_path)
y_train = np.load(y_tain_path)
X_test  = np.load(X_TEST_PATH)
y_test  = np.load(Y_TEST_PATH)

with open(LE_PATH, "rb") as f:
    le = pickle.load(f)

# train the model
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean', weights='distance')
knn.fit(X_train, y_train)


#confidence threshold
CONF_THRESH = 0.60 

y_pred_prob = knn.predict_proba(X_test)
y_pred_raw = knn.predict(X_test)

# Apply Threshold
y_pred_thresholded = []
for p in y_pred_prob:
    if np.max(p) >= CONF_THRESH:
        y_pred_thresholded.append(np.argmax(p))
    else:
        y_pred_thresholded.append(-1) # Unknown

# Calculate Metrics
acc_raw = accuracy_score(y_test, y_pred_raw)
print(f"\n       KNN Results              ")
print(f"Standard Accuracy (No Rejection): {acc_raw:.4f}")

accepted_indices = [i for i, x in enumerate(y_pred_thresholded) if x != -1]
if accepted_indices:
    acc_accepted = accuracy_score(y_test[accepted_indices], [y_pred_thresholded[i] for i in accepted_indices])
    print(f"Accuracy on Accepted Samples: {acc_accepted:.4f}")
else:
    print("All samples rejected.")

rejection_rate = (y_pred_thresholded.count(-1) / len(y_test)) * 100
print(f"Rejection Rate: {rejection_rate:.2f}%")

print("\nClassification Report (Standard):")
print(classification_report(y_test, y_pred_raw, target_names=le.classes_))


# save the model
with open(KNN_MODEL_PATH, "wb") as f:
    pickle.dump(knn, f)
