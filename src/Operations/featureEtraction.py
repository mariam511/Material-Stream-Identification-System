import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import pickle

ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

X_TRAIN_PATH = f"{ARTIFACTS_DIR}/X_train.npy"
Y_TRAIN_PATH = f"{ARTIFACTS_DIR}/y_train.npy"
X_TEST_PATH  = f"{ARTIFACTS_DIR}/X_test.npy"
Y_TEST_PATH  = f"{ARTIFACTS_DIR}/y_test.npy"
ENCODER_PATH = f"{ARTIFACTS_DIR}/label_encoder.pkl"

# VGG16 requires exactly 224x224
IMG_SIZE = (224, 224)

# ==================== LOAD CNN MODEL ====================
print("Loading VGG16 Model...")
# include_top=False: Removes the final "classification" layer (Cat vs Dog)
# pooling='avg': Averages the features into a 1D vector of size 512
model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')
print("VGG16 Model Loaded successfully!")


# ==================== FEATURE EXTRACTION FUNCTIONS ====================

def extract_features_from_dir(base_dir):
    X, y = [], []
    classes = sorted([d for d in os.listdir(base_dir)
                      if os.path.isdir(os.path.join(base_dir, d))])

    print(f"Extracting Deep Features from {base_dir}")
    print("Classes:", classes)

    for cls in classes:
        folder = os.path.join(base_dir, cls)
        files = os.listdir(folder)
        print(f"  → {cls}: {len(files)} images")

        for file in files:
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path)
            if img is None:
                continue

            # 1. Resize to 224x224 for VGG16
            img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)

            # 2. Preprocess for VGG16
            # Expand dims: (224, 224, 3) -> (1, 224, 224, 3)
            x = np.expand_dims(img, axis=0)
            # VGG specific preprocessing (Mean subtraction, BGR handling)
            x = preprocess_input(x)

            # 3. Extract Features
            # verbose=0 hides the progress bar for every single image
            features = model.predict(x, verbose=0)
            
            # features shape is (1, 512), we want (512,)
            X.append(features[0])
            y.append(cls)

    return np.array(X), np.array(y)


# ==================== MAIN ====================

# 🔁 If features already exist → LOAD
if all(os.path.exists(p) for p in
       [X_TRAIN_PATH, Y_TRAIN_PATH, X_TEST_PATH, Y_TEST_PATH, ENCODER_PATH]):

    print("Cached features found. Loading...")

    X_train = np.load(X_TRAIN_PATH)
    y_train = np.load(Y_TRAIN_PATH)
    X_test  = np.load(X_TEST_PATH)
    y_test  = np.load(Y_TEST_PATH)

    with open(ENCODER_PATH, "rb") as f:
        le = pickle.load(f)

else:
    print("No cache found. Extracting features using CNN...")

    # -------- TRAIN --------
    X_train, y_train = extract_features_from_dir("data/augmented/train")

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)

    # -------- TEST --------
    X_test, y_test = extract_features_from_dir("data/test")
    y_test = le.transform(y_test)

    # -------- SAVE --------
    np.save(X_TRAIN_PATH, X_train)
    np.save(Y_TRAIN_PATH, y_train)
    np.save(X_TEST_PATH, X_test)
    np.save(Y_TEST_PATH, y_test)

    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)

    print("Features cached successfully!")

print("DONE")
print("Train:", X_train.shape, y_train.shape)
print("Test :", X_test.shape, y_test.shape)
print("Classes:", dict(zip(le.classes_, le.transform(le.classes_))))


# ==================== CAMERA FEATURE EXTRACTION FUNCTION ====================
def extract_features_from_image(img):
    """
    Extract features from a single image using VGG16.
    Returns a 1D feature vector (Size 512).
    """
    # 1. Resize
    img_resized = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    
    # 2. Preprocess
    x = np.expand_dims(img_resized, axis=0)
    x = preprocess_input(x)
    
    # 3. Predict
    features = model.predict(x, verbose=0)
    
    # Return flattened 1D array
    return features[0]