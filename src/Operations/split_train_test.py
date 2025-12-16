import os
import shutil
from sklearn.model_selection import train_test_split

processedDir = "data/processed"
trainDir = "data/train"
testDir = "data/test"
test_ratio = 0.2  # 20% for testing

os.makedirs(trainDir, exist_ok=True)
os.makedirs(testDir, exist_ok=True)

classes = [cls for cls in os.listdir(processedDir) if os.path.isdir(os.path.join(processedDir, cls))]

for cls in classes:
    cls_path = os.path.join(processedDir, cls)
    images = [f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))]

    train_imgs, test_imgs = train_test_split(images, test_size=test_ratio, random_state=42)

    os.makedirs(os.path.join(trainDir, cls), exist_ok=True)
    os.makedirs(os.path.join(testDir, cls), exist_ok=True)

    for img in train_imgs:
        shutil.copy(os.path.join(cls_path, img), os.path.join(trainDir, cls, img))
    for img in test_imgs:
        shutil.copy(os.path.join(cls_path, img), os.path.join(testDir, cls, img))

print("Train/Test split completed!")