import os
import cv2
import numpy as np
import random

processedDir = "data/processed"
augmentedDir = "data/augmented"
os.makedirs(augmentedDir,exist_ok=True)

classes=[]
for item in os.listdir(processedDir):
    path=os.path.join(processedDir,item)
    if(os.path.isdir(path)):
        classes.append(item)
for cls in classes:
    srcFolder = os.path.join(processedDir, cls)
    distFolder = os.path.join(augmentedDir, cls)
    os.makedirs(distFolder, exist_ok=True)
    for file in os.listdir(srcFolder):
        filePath=os.path.join(srcFolder,file)
        img=cv2.imread(filePath)
        if (img is not None):
            # save original 
            cv2.imwrite(os.path.join(distFolder, file), img)
            # flip horizontally
            flip_img=cv2.flip(img,1)
            cv2.imwrite(os.path.join(distFolder,"flip_"+file),flip_img)
            #rotate 90 clockwise
            rot_img=cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(os.path.join(distFolder,"rot90_"+file),rot_img)
            #random brightness
            #new_pixel=alpha×pixel+beta alpha->contrast beta->brightness
            bright_img=cv2.convertScaleAbs(img, alpha=1.0, beta=random.randint(-50,50))
            cv2.imwrite(os.path.join(distFolder, "bright_"+file), bright_img)
        else:
            print("cannot read:", filePath)


#checking number of images before and after augmentation for each class:")

print("checking number of images before and after augmentation for each class:")

for cls in classes:
    count = 0
    for file in os.listdir(os.path.join(processedDir, cls)):
        if os.path.isfile(os.path.join(processedDir, cls, file)):
            count += 1
    
    augmented_count = 0
    for file in os.listdir(os.path.join(augmentedDir, cls)):
        if os.path.isfile(os.path.join(augmentedDir, cls, file)):
            augmented_count += 1

    required_count = int(count * 1.3)

    if augmented_count >= required_count:
        status = "oK"
    else:
        status = "needs more augmentation"

    print(f"{cls}: original={count}, augmented={augmented_count}, required={required_count} → {status}")



