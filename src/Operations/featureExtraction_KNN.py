
#--------------------------- Feature Extraction for KNN model----------------------#


import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import pickle

models = "artifacts"
os.makedirs(models, exist_ok=True)

X_TRAIN_PATH = f"{models}/knn_X_train.npy"  
Y_TRAIN_PATH = f"{models}/knn_y_train.npy"
X_TEST_PATH  = f"{models}/knn_X_test.npy"
Y_TEST_PATH  = f"{models}/knn_y_test.npy"
ENCODER_PATH = f"{models}/knn_label_encoder.pkl"

sizeOfImg = (224, 224)
 # load cnn model
model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')

def extract_features_from_dir(base_dir):
    X, y = [], []
    classes = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    
    print(f"Extracting features from {base_dir} for KNN...")
    
    for cls in classes:
        folder = os.path.join(base_dir, cls)
        files = os.listdir(folder)
        print(f"  -> {cls}: {len(files)} images")
        
        for file in files:
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path)
            if img is None: continue
            
            img = cv2.resize(img, sizeOfImg, interpolation=cv2.INTER_AREA)
            x = np.expand_dims(img, axis=0)
            x = preprocess_input(x)
            features = model.predict(x, verbose=0)
            
            X.append(features[0])
            y.append(cls)
            
    return np.array(X), np.array(y)

if __name__ == "__main__":
    
    X_train, y_train = extract_features_from_dir("data/augmented/train")
    X_test, y_test = extract_features_from_dir("data/test")

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    np.save(X_TRAIN_PATH, X_train)
    np.save(Y_TRAIN_PATH, y_train)
    np.save(X_TEST_PATH, X_test)
    np.save(Y_TEST_PATH, y_test)

    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)

    print("Greeeeeeaaaaaaaat !!!!!!!!!!!!!!!!!!!!!!!!!         Stored successfully ")
    print("Train:", X_train.shape)
    print("Test:", X_test.shape)