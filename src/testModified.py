import os
import cv2
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import pandas as pd


def predict_unlabeled(dataFilePath, bestModelPath,modeType ="KNN"):
    models = "D:/SHaHD/4th_first term/Machine Learning/Assignments/Clean_ML_Project/artifacts"
    
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
    filenames = []

    for file in os.listdir(dataFilePath):
        img_path = os.path.join(dataFilePath, file)
        if not os.path.isfile(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (224, 224),interpolation=cv2.INTER_AREA)
        x = preprocess_input(np.expand_dims(img, axis=0))
        feat = CNN.predict(x, verbose=0)

        X.append(feat[0])
        filenames.append(file)

    # X = np.array(X)

    # Same preprocessing as training
    X_features = np.array(X)
    X_scaled = scaler.transform(X_features)
    if modeType.upper() == "KNN":
        X_final = pca.transform(X_scaled)
    else:
        X_final = X_scaled
    # X_scaled = scaler.transform(X)
    # X_final = pca.transform(X_scaled)

    # Predict
    probs = model.predict_proba(X_final)
    CONF_THRESH = CONFIDENCE_THRESHOLD

    results = {}
    for fname, p in zip(filenames, probs):
        if np.max(p) >= CONF_THRESH:
            label = le.inverse_transform([np.argmax(p)])[0]
        else:
            label = "UNKNOWN"
        results[fname] = label

    return results

if __name__ == "__main__":
    test_data_path = r"D:/SHaHD/4th_first term/Machine Learning/Assignments/Clean_ML_Project/test_data/test_data"
    knn_model_path = "artifacts/knn_model.pkl"  
    svm_model_path = "artifacts/svm_model.pkl"

    # # Test SVM
    print("SVM Testing")
    svm_results = predict_unlabeled(test_data_path, svm_model_path, modeType="SVM")
    results_SVM = []
    for img, pred in svm_results.items():
        results_SVM.append((img, pred))
        print(f"{img}  -->  {pred}")
    df_SVM = pd.DataFrame(results_SVM, columns=["Image Name", "Prediction_SVM"])
        # Save to Excel
    df_SVM.to_excel("Prediction_SVM.xlsx", index=False)
    print("\nExcel file saved as Prediction_SVM.xlsx")

    # test knn
    print("\nKNN Testing ")
    results_KNN = []
    knn_results = predict_unlabeled(test_data_path, knn_model_path, modeType="KNN")
    for img, pred in knn_results.items():
        results_KNN.append((img, pred))
        # print("Results: ",results)
        print(f"{img}  -->  {pred}")
    df_KNN = pd.DataFrame(results_KNN, columns=["Image Name", "Prediction_KNN"])
        # Save to Excel
    df_KNN.to_excel("Prediction_KNN.xlsx", index=False)
    print("\nExcel file saved as Prediction_KNN.xlsx")
    
    # Save combined results
    df_total = pd.merge(df_SVM, df_KNN, on="Image Name", how="outer")
    df_total.to_excel("Prediction_Total.xlsx", index=False)
    print("\nExcel file saved as Prediction_Total.xlsx")

