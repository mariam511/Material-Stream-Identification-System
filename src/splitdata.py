from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np


print("Original X shape:", X.shape)
print("Original y shape:", y.shape)


le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Class mapping:", class_mapping)
print("y_encoded shape:", y_encoded.shape)

# split data 
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("Before scaling:")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# Standardization (Mean=0, Std=1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("After scaling:")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)