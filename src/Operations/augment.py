import os
import cv2
import numpy as np
import random

processedDir = "data/train"  # augment training data only
augmentedDir = "data/augmented/train"
os.makedirs(augmentedDir, exist_ok=True)

classes = []
for item in os.listdir(processedDir):
    path = os.path.join(processedDir, item)
    if os.path.isdir(path):
        classes.append(item)

for cls in classes:
    srcFolder = os.path.join(processedDir, cls)
    distFolder = os.path.join(augmentedDir, cls)
    os.makedirs(distFolder, exist_ok=True)
    for file in os.listdir(srcFolder):
        filePath = os.path.join(srcFolder, file)
        img = cv2.imread(filePath)
        if img is not None:
            cv2.imwrite(os.path.join(distFolder, file), img)
            # flip horizontally
            flip_img = cv2.flip(img, 1)
            cv2.imwrite(os.path.join(distFolder, "flip_" + file), flip_img)
            # rotate 90 clockwise
            rot_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(os.path.join(distFolder, "rot90_" + file), rot_img)
            # random brightness
            bright_img = cv2.convertScaleAbs(img, alpha=1.0, beta=random.randint(-50, 50))
            cv2.imwrite(os.path.join(distFolder, "bright_" + file), bright_img)


for cls in classes:
    count = 0
    for file in os.listdir(os.path.join(processedDir, cls)):
        if os.path.isfile(os.path.join(processedDir, cls, file)):
            count += 1

    augmented_count = 0
    for file in os.listdir(os.path.join(augmentedDir, cls)):
        if os.path.isfile(os.path.join(augmentedDir, cls, file)):
            augmented_count += 1

# Balancing classes
class_counts = {}
max_count = 0
for cls in classes:
    folder = os.path.join(augmentedDir, cls)
    images = []
    for f in os.listdir(folder):
        file_path = os.path.join(folder, f)
        if os.path.isfile(file_path):
            images.append(f)
    count = len(images)
    class_counts[cls] = count
    if count > max_count:
        max_count = count

for cls in classes:
    folder = os.path.join(augmentedDir, cls)
    images = []
    for f in os.listdir(folder):
        file_path = os.path.join(folder, f)
        if os.path.isfile(file_path):
            images.append(f)

    curr_count = len(images)

    needed = max_count - curr_count

    # Extra flips
    extra_index = 0
    while curr_count < max_count and extra_index < len(images):
        src_file = images[extra_index]
        src_path = os.path.join(folder, src_file)
        img = cv2.imread(src_path)
        if img is not None:
            flipped_img = cv2.flip(img, 1)
            name = "extra_flip_" + str(extra_index) + ".jpg"
            cv2.imwrite(os.path.join(folder, name), flipped_img)
            curr_count += 1
        extra_index += 1

    # Extra rotations
    extra_index = 0
    while curr_count < max_count and extra_index < len(images):
        src_file = images[extra_index]
        src_path = os.path.join(folder, src_file)
        img = cv2.imread(src_path)
        if img is not None:
            rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            name = "extra_rot_" + str(extra_index) + ".jpg"
            cv2.imwrite(os.path.join(folder, name), rotated_img)
            curr_count += 1
        extra_index += 1

    # Extra brightness
    extra_index = 0
    while curr_count < max_count and extra_index < len(images):
        src_file = images[extra_index]
        src_path = os.path.join(folder, src_file)
        img = cv2.imread(src_path)
        if img is not None:
            bright_img = cv2.convertScaleAbs(img, alpha=1.0, beta=30)
            name = "extra_bright_" + str(extra_index) + ".jpg"
            cv2.imwrite(os.path.join(folder, name), bright_img)
            curr_count += 1
        extra_index += 1
