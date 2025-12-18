import cv2
import numpy as np
import pickle
from featureEtraction import extract_features_from_image


models = "artifacts"
SCALER_PATH = f"{models}/scaler.pkl"
LE_PATH = f"{models}/label_encoder.pkl"
SVM_MODEL_PATH = f"{models}/svm_model.pkl"



with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)
with open(LE_PATH, "rb") as f:
    le = pickle.load(f)
with open(SVM_MODEL_PATH, "rb") as f:
    model = pickle.load(f)
    
CONF_THRESH = 0.5
UNKNOWN_CLASS = "Unknown"

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    feat = extract_features_from_image(frame)
    feat_scaled = scaler.transform([feat])
    probs = model.predict_proba(feat_scaled)[0]
    max_prob = max(probs)

    if max_prob < CONF_THRESH:
        label = UNKNOWN_CLASS
    else:
        label = le.inverse_transform([probs.argmax()])[0]

    cv2.putText(frame, f"{label} ({max_prob:.2f})", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("SVM Material Classification ", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
