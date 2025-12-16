import os
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.preprocessing import LabelEncoder
import pickle

ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

X_TRAIN_PATH = f"{ARTIFACTS_DIR}/X_train.npy"
Y_TRAIN_PATH = f"{ARTIFACTS_DIR}/y_train.npy"
X_TEST_PATH  = f"{ARTIFACTS_DIR}/X_test.npy"
Y_TEST_PATH  = f"{ARTIFACTS_DIR}/y_test.npy"
ENCODER_PATH = f"{ARTIFACTS_DIR}/label_encoder.pkl"

IMG_SIZE = (64, 64)

LBP_RADIUS = 3
LBP_N_POINTS = 8 * LBP_RADIUS
LBP_METHOD = 'uniform'

HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_ORIENTATIONS = 9

COLOR_BINS = 8

# ==================== FEATURE EXTRACTION FUNCTIONS FOR CAMERA ====================
# Camera feature extraction

def extract_features_from_dir(base_dir):
    X, y = [], []
    classes = sorted([d for d in os.listdir(base_dir)
                      if os.path.isdir(os.path.join(base_dir, d))])

    print(f"Extracting from {base_dir}")
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

            img = cv2.resize(img, IMG_SIZE,interpolation= cv2.INTER_AREA)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            hog_feat = hog(
                gray,
                orientations=HOG_ORIENTATIONS,
                pixels_per_cell=HOG_PIXELS_PER_CELL,
                cells_per_block=HOG_CELLS_PER_BLOCK,
                block_norm='L2-Hys',
                transform_sqrt=True,
                feature_vector=True
            )

            lbp = local_binary_pattern(gray, LBP_N_POINTS, LBP_RADIUS, LBP_METHOD)
            lbp_hist, _ = np.histogram(
                lbp.ravel(),
                bins=np.arange(0, LBP_N_POINTS + 3),
                range=(0, LBP_N_POINTS + 2)
            )
            lbp_hist = lbp_hist.astype("float")
            lbp_hist /= (lbp_hist.sum() + 1e-6)

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
            # h = cv2.calcHist([hsv], [0], None, [COLOR_BINS], [0, 256])
            # s = cv2.calcHist([hsv], [1], None, [COLOR_BINS], [0, 256])
            # v = cv2.calcHist([hsv], [2], None, [COLOR_BINS], [0, 256])
            # color_hist = np.concatenate([h, s, v]).ravel()
            # color_hist /= (color_hist.sum() + 1e-6)
            hist_3d = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            cv2.normalize(hist_3d, hist_3d)
            color_hist = hist_3d.flatten()

            feature_vector = np.concatenate([hog_feat, lbp_hist, color_hist])

            X.append(feature_vector)
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
    print("No cache found. Extracting features...")

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
    #    """
    #Extract features from a single image (numpy array).
    #Returns a 1D feature vector (HOG + LBP + Color Histogram).
    #"""
    # reuse all training parameters
    img = cv2.resize(img, IMG_SIZE,interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # HOG
    hog_feat = hog(gray, orientations=HOG_ORIENTATIONS,
                pixels_per_cell=HOG_PIXELS_PER_CELL,
                cells_per_block=HOG_CELLS_PER_BLOCK,
                block_norm='L2-Hys',
                transform_sqrt=True, feature_vector=True)
    # LBP
    lbp = local_binary_pattern(gray, LBP_N_POINTS, LBP_RADIUS, LBP_METHOD)
    lbp_hist, _ = np.histogram(lbp.ravel(),
                            bins=np.arange(0, LBP_N_POINTS + 3),
                            range=(0, LBP_N_POINTS + 2))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)
    
    # Color histogram
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    # h = cv2.calcHist([hsv], [0], None, [COLOR_BINS], [0, 256])
    # s = cv2.calcHist([hsv], [1], None, [COLOR_BINS], [0, 256])
    # v = cv2.calcHist([hsv], [2], None, [COLOR_BINS], [0, 256])
    # color_hist = np.concatenate([h, s, v]).ravel()
    # color_hist /= (color_hist.sum() + 1e-6)

    hist_3d = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

    # 2. Normalize the result (Standardization)
    cv2.normalize(hist_3d, hist_3d)

    # 3. Flatten the 3D cube into a 1D array
    color_hist = hist_3d.flatten()

    feature_vector = np.concatenate([hog_feat, lbp_hist, color_hist])
    return feature_vector
