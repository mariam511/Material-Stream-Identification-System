
#--------------------------- Data Scaling for SVM model----------------------#



import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler

models = "artifacts"

x_trainLoc = f"{models}/X_train.npy" 
y_trainLoc = f"{models}/y_train.npy"
X_TEST_PATH  = f"{models}/X_test.npy"
Y_TEST_PATH  = f"{models}/y_test.npy"
scalerLoc  = f"{models}/scaler.pkl"
x_train_scaledLoc = f"{models}/X_train_scaled.npy"
x_test_scaledLoc  = f"{models}/X_test_scaled.npy"

# load original data
X_train = np.load(x_trainLoc)
y_train = np.load(y_trainLoc)
X_test  = np.load(X_TEST_PATH) 
y_test  = np.load(Y_TEST_PATH)


print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test :", X_test.shape)
print("y_test :", y_test.shape)

# load and scale data 

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Save scaler and scaled data 
with open(scalerLoc, "wb") as f:
    pickle.dump(scaler, f)
np.save(x_train_scaledLoc
, X_train_scaled)
np.save(x_test_scaledLoc, X_test_scaled)

print("X_train_scaled:", X_train_scaled.shape)
print("X_test_scaled :", X_test_scaled.shape)
