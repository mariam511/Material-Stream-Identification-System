import os
import cv2
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input


def predict(dataFilePath, bestModelPath,modeType ="KNN"):
    # you will take only test data no training data 
    models = "artifacts"
    

    if(modeType.upper()=="SVM"): # change paths for files
        scalerLoc = f"{models}/scaler.pkl"
        labelLoc = f"{models}/label_encoder.pkl"
        CONFIDENCE_THRESHOLD = 0.44
        

    else:
        scalerLoc = f"{models}/knn_scaler.pkl"
        pcaLoc    = f"{models}/knn_pca.pkl" # should be used only if model used is knn althoough set it by scaler
        labelLoc     = f"{models}/knn_label_encoder.pkl"
        CONFIDENCE_THRESHOLD = 0.6



    with open(scalerLoc, "rb") as f: scaler = pickle.load(f)
    if(modeType.upper()=="KNN"):
        with open(pcaLoc, "rb") as f: pca = pickle.load(f)
    with open(labelLoc, "rb") as f: le = pickle.load(f)
    with open(bestModelPath, "rb") as f: model = pickle.load(f)

    CNN = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    # load test data 
    X = []
    y_true = []
    SizeOfImg = (224, 224)

    for cls_folder in sorted(os.listdir(dataFilePath)):
        full_folder_path = os.path.join(dataFilePath, cls_folder)
        if os.path.isdir(full_folder_path):
            for item in os.listdir(full_folder_path):
                img = cv2.imread(os.path.join(full_folder_path, item))
                if img is not None:
                    img = cv2.resize(img, SizeOfImg, interpolation=cv2.INTER_AREA)
                    x = preprocess_input(np.expand_dims(img, axis=0))
                    features = CNN.predict(x, verbose=0)
                    X.append(features[0])
                    y_true.append(cls_folder)

    X_features = np.array(X)
    X_scaled = scaler.transform(X_features)
    if modeType.upper() == "KNN":
        X_final = pca.transform(X_scaled)
    else:
        X_final = X_scaled

    # Prediction
    y_pred_prob = model.predict_proba(X_final)
    unknown_idx = len(le.classes_)
    
    # Unknown class handling
    final_classes = [np.argmax(prob) if np.max(prob) >= CONFIDENCE_THRESHOLD else unknown_idx for prob in y_pred_prob]

    # Accurcy
    y_true_enc = le.transform(y_true)
    print(f"\n--- {modeType.upper()} Evaluation Report ---")
    print(f"Accuracy: {accuracy_score(y_true_enc, final_classes):.4f}")
    
    return final_classes


if __name__ =="__main__":
    test_data_path = "data/test" 
    svm_model_path = "artifacts/svm_model.pkl"
    knn_model_path = "artifacts/knn_model.pkl"

    # Test SVM
    print("SVM Testing")
    svm_results = predict(test_data_path, svm_model_path, modeType="SVM")

    # test knn
    print("\nKNN Testing ")
    knn_results = predict(test_data_path, knn_model_path, modeType="KNN")



    
