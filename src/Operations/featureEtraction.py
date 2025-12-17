

#--------------------------- Feature Extraction for SVM model----------------------#


import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import pickle


models = "artifacts"
os.makedirs(models, exist_ok=True)

X_train_path = f"{models}/X_train.npy"
y_train_path = f"{models}/y_train.npy"
x_test_path  = f"{models}/X_test.npy"
y_test_path  = f"{models}/y_test.npy"
encoderLocation = f"{models}/label_encoder.pkl"

# high image size to be suitable for CNN
IMG_SIZE = (224, 224)

# Use CNN for feature extraction to improve results
model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')



def extract_features_from_dir(base_dir):
    X, y = [], []
    classes = sorted([d for d in os.listdir(base_dir)
                      if os.path.isdir(os.path.join(base_dir, d))])

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

            # Resize to 224x224 
            img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)

            x = np.expand_dims(img, axis=0)
            x = preprocess_input(x)

            # 3. Extract Features
            features = model.predict(x, verbose=0)
            
            # reshape
            X.append(features[0])
            y.append(cls)

    return np.array(X), np.array(y)


# load if already features exist
if all(os.path.exists(p) for p in
       [X_train_path, y_train_path, x_test_path, y_test_path, encoderLocation]):

    X_train = np.load(X_train_path)
    y_train = np.load(y_train_path)
    X_test  = np.load(x_test_path)
    y_test  = np.load(y_test_path)

    with open(encoderLocation, "rb") as f:
        le = pickle.load(f)

else:

    # path of train data
    X_train, y_train = extract_features_from_dir("data/augmented/train")
    #encoding
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)

    # path for test data
    X_test, y_test = extract_features_from_dir("data/test")
    y_test = le.transform(y_test)

    # save train & test data 
    np.save(X_train_path, X_train)
    np.save(y_train_path, y_train)
    np.save(x_test_path, X_test)
    np.save(y_test_path, y_test)

    with open(encoderLocation, "wb") as f:
        pickle.dump(le, f)



print("Greeeeeeaaaaaaaat !!!!!!!!!!!!!!!!!!!!!!!!!         Stored successfully ")
print("Train size:", X_train.shape, y_train.shape)
print("Test :", X_test.shape, y_test.shape)
print("Classes:", dict(zip(le.classes_, le.transform(le.classes_))))


def extract_features_from_image(img):

    img_resized = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    x = np.expand_dims(img_resized, axis=0)
    x = preprocess_input(x)

    features = model.predict(x, verbose=0)

    return features[0]