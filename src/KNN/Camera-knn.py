import cv2
import pickle
from featureEtraction import extract_features_from_image

ARTIFACTS_DIR = "artifacts"
SCALER_PATH = f"{ARTIFACTS_DIR}/scaler.pkl"
LE_PATH = f"{ARTIFACTS_DIR}/label_encoder.pkl"
KNN_MODEL_PATH = f"{ARTIFACTS_DIR}/knn_model.pkl"

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)
with open(LE_PATH, "rb") as f:
    le = pickle.load(f)
with open(KNN_MODEL_PATH, "rb") as f:
    model = pickle.load(f)

CONFIDENCE_THRESHOLD = 0.75

UNKNOWN_CLASS = "Unknown"

# # Load model, scaler, label encoder
# model = joblib.load("artifacts/knn_model.pkl")
# scaler = joblib.load("artifacts/scaler.pkl")
# le = joblib.load("artifacts/label_encoder.pkl")


cap = cv2.VideoCapture(0)
print("Starting camera...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    feat = extract_features_from_image(frame)
    feat_scaled = scaler.transform([feat])

    probs = model.predict_proba(feat_scaled)[0]
    max_prob = max(probs)

    if max_prob < CONFIDENCE_THRESHOLD:
        label = UNKNOWN_CLASS
    else:
        label = le.inverse_transform([probs.argmax()])[0]

    cv2.putText(frame, f"{label} ({max_prob:.2f})", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Material Classification KNN", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
