import os
import cv2
import numpy as np

processedDir = "data/processed"
os.makedirs(processedDir, exist_ok=True)

cleanedDir="data/cleaned"
os.makedirs(cleanedDir,exist_ok=True)

#read cleaned Data and make the classes in processing dir
classes=[]
for cls in os.listdir(cleanedDir):
    path=os.path.join(cleanedDir,cls)
    if(os.path.isdir(path)):
        classes.append(cls)
        classFolderPath=os.path.join(cleanedDir,cls)
        os.makedirs(classFolderPath,exist_ok=True)

IMG_SIZE=(128,128)       

#Resizing
for cls in classes:
    srcFolderPath=os.path.join(cleanedDir,cls)
    distFolderPath=os.path.join(processedDir,cls)
    os.makedirs(distFolderPath, exist_ok=True)


    for file in os.listdir(srcFolderPath):
        srcFilePath=os.path.join(srcFolderPath,file)
        distFilePath=os.path.join(distFolderPath,file)

        img=cv2.imread(srcFilePath)
        if(img is not None):
            img_resized=cv2.resize(img,IMG_SIZE)
            cv2.imwrite(distFilePath,img_resized)
        else:
            print("can not read:",srcFilePath)





