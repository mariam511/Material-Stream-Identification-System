import os
import cv2
import pandas as pd

rawDir="data/raw"

#read raw Data
classes=[]
for item in os.listdir(rawDir):
    path=os.path.join(rawDir,item)
    if(os.path.isdir(path)):
        classes.append(item)

for cls in classes:
    folder_path=os.path.join(rawDir,cls)
    cnt=0
    for file in os.listdir(folder_path):
        file_path=os.path.join(folder_path,file)
        if(os.path.isfile(file_path)):
            cnt+=1
    print(cls,":",cnt,"files")
   
#filter raw Data
cleanedDir="data/cleaned"
os.makedirs(cleanedDir,exist_ok=True)

for cls in classes:
    print("class",cls)
    srcFolderPath=os.path.join(rawDir,cls)
    distFolderPath=os.path.join(cleanedDir,cls)
    os.makedirs(distFolderPath,exist_ok=True)
    cnt=0
    for file in os.listdir(srcFolderPath):
        srcFilePath=os.path.join(srcFolderPath,file)
        distFilePath=os.path.join(distFolderPath,file)

        img=cv2.imread(srcFilePath)
        if(img is not None):
            cv2.imwrite(distFilePath,img)
            cnt+=1
        else:
            print("can not read:",srcFilePath)

#cleaned Data
print("cleaned Data")
for cls in classes:
    folder_path=os.path.join(cleanedDir,cls)
    cnt=0
    for file in os.listdir(folder_path):
        file_path=os.path.join(folder_path,file)
        if(os.path.isfile(file_path)):
            cnt+=1
    print(cls,":",cnt,"files")

