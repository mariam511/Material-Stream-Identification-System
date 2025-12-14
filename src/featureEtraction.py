
import os
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.preprocessing import LabelEncoder

augmentedDir = "data\augmented"  
classes = [d for d in os.listdir(augmentedDir) if os.path.isdir(os.path.join(augmentedDir,d))]
print("Classes found:", classes)

IMG_SIZE = (128, 128)
LBP_RADIUS = 1
LBP_N_POINTS = 8 * LBP_RADIUS
LBP_METHOD = 'uniform'
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_ORIENTATIONS = 9
COLOR_BINS = 16  

#  Feature Extraction
X = []
y = []


for cls in classes:
    folder = os.path.join(augmentedDir, cls)
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    print(f"Processing class '{cls}' with {len(files)} images...")
    
    for file in files:
        img_path = os.path.join(folder, file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Resize
        img = cv2.resize(img, IMG_SIZE)
        
        # Convert to grayscale for HOG and LBP
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # HOG feature
        hog_feat = hog(
            gray,
            orientations=HOG_ORIENTATIONS,
            pixels_per_cell=HOG_PIXELS_PER_CELL,
            cells_per_block=HOG_CELLS_PER_BLOCK,
            block_norm='L2-Hys',
            transform_sqrt=True,
            feature_vector=True
        )
        
        # LBP feature
        lbp = local_binary_pattern(gray, LBP_N_POINTS, LBP_RADIUS, LBP_METHOD)
        (lbp_hist, _) = np.histogram(lbp.ravel(),
                                     bins=np.arange(0, LBP_N_POINTS + 3),
                                     range=(0, LBP_N_POINTS + 2))
        lbp_hist = lbp_hist.astype("float")
        lbp_hist /= (lbp_hist.sum() + 1e-6)
        
        # Color histogram (HSV)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [COLOR_BINS], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [COLOR_BINS], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [COLOR_BINS], [0, 256])
        color_hist = np.concatenate([h_hist, s_hist, v_hist]).ravel()
        color_hist = color_hist / (color_hist.sum() + 1e-6)

        # merge all features
        feature_vector = np.concatenate([hog_feat, lbp_hist, color_hist])
        
        X.append(feature_vector)
        y.append(cls)

X = np.array(X)
y = np.array(y)


le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("Feature extraction done!")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("y_encoded shape:", y_encoded.shape)
print("Class mapping:", dict(zip(le.classes_, le.transform(le.classes_))))
