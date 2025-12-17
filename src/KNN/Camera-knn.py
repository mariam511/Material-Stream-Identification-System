import cv2
import numpy as np
import pickle
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

# Setup paths
ARTIFACTS_DIR = "artifacts"
SCALER_PATH = f"{ARTIFACTS_DIR}/knn_scaler.pkl"
PCA_PATH    = f"{ARTIFACTS_DIR}/knn_pca.pkl"
LE_PATH     = f"{ARTIFACTS_DIR}/knn_label_encoder.pkl"
KNN_MODEL_PATH = f"{ARTIFACTS_DIR}/knn_model.pkl"

# Load Pipeline
print("Loading KNN Pipeline...")
with open(SCALER_PATH, "rb") as f: scaler = pickle.load(f)
with open(PCA_PATH, "rb") as f: pca = pickle.load(f)
with open(LE_PATH, "rb") as f: le = pickle.load(f)
with open(KNN_MODEL_PATH, "rb") as f: model = pickle.load(f)

# Load VGG for feature extraction (must match training extraction)
vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')

IMG_SIZE = (224, 224)
CONF_THRESH = 0.60
UNKNOWN_CLASS = "Unknown"

def get_embedding(img):
    img_resized = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    x = np.expand_dims(img_resized, axis=0)
    x = preprocess_input(x)
    return vgg.predict(x, verbose=0)[0]

cap = cv2.VideoCapture(0)
print("Camera started...")

while True:
    ret, frame = cap.read()
    if not ret: break

    # extract raw features
    raw_feat = get_embedding(frame)
    
    # scale
    feat_norm = scaler.transform([raw_feat])
    
    # pca
    feat_pca = pca.transform(feat_norm)

    # predict
    probs = model.predict_proba(feat_pca)[0]
    max_prob = max(probs)
    idx = np.argmax(probs)

    if max_prob < CONF_THRESH:
        label = UNKNOWN_CLASS
        color = (0, 0, 255) 
    else:
        label = le.inverse_transform([idx])[0]
        color = (0, 255, 0) 

    text = f"{label} ({max_prob:.2f})"
    cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("KNN Classification", frame)

    if cv2.waitKey(1) & 0xFF == 27: # ESC
        break

cap.release()
cv2.destroyAllWindows()