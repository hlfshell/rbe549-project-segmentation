# This script is made to read a random dataset from the Lyft Udacity Challenge.
# The challenge was given to Udacity students to try and create semantic
# segmentation networks for a given CARLA dataset. The dataset is split into
# 5 sub datasets, but each dataset is just an offset fraction fo a second from
# one-another. Since we added delays to our dataset to try and prevent too many
# similar images, we really only need one of these datasets. This still adds
# a thousand results for us, which is fine.
# Dataset is downloaded from:
# https://www.kaggle.com/datasets/kumaresanmanickavelu/lyft-udacity-challenge?resource=download

# from carla_controller.transform_dataset import convert_semantic_image

import cv2
import numpy as np
import os
import shutil
import sys

from pathlib import Path


SEMANTIC_CATEGORIES = {
    0: "None",
    1: "Buildings",
    2: "Fences",
    3: "Other",
    4: "Pedestrians",
    5: "Poles",
    6: "RoadLines",
    7: "Roads",
    8: "Sidewalks",
    9: "Vegetation",
    10: "Vehicles",
    11: "Walls",
    12: "TrafficSigns"
}

SEMANTIC_COLORS = {
    "None": (0, 0, 0),
    "Buildings": [70, 70, 70],
    "Fences": [190, 153, 153],
    "Other": [72, 0, 90],
    "Pedestrians": [220, 20, 60],
    "Poles": [153, 153, 153],
    "RoadLines": [157, 234, 50],
    "Roads": [128, 64, 128],
    "Sidewalks": [244, 35, 232],
    "Vegetation": [107, 142, 35],
    "Vehicles": [0, 0, 255],
    "Walls": [102, 102, 156],
    "TrafficSigns": [220, 220, 0],
}

def convert_semantic_image(image_path : str):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    for key in SEMANTIC_CATEGORIES.keys():
        r_channel = (0, 0, key)
        category = SEMANTIC_CATEGORIES[key]
        color = SEMANTIC_COLORS[category]

        img[np.all(img == r_channel, axis=-1)] = color

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


LYFT_DATASET_DIR = "./lyft"
DATASET_DIR = "./tmp"
CHOSEN_DATASET = "dataA"

rgb_prefix = f"{LYFT_DATASET_DIR}/{CHOSEN_DATASET}/{CHOSEN_DATASET}/CameraRGB"
semantic_prefix = f"{LYFT_DATASET_DIR}/{CHOSEN_DATASET}/{CHOSEN_DATASET}/CameraSem"

Path(f"{DATASET_DIR}/rgb").mkdir(parents=True, exist_ok=True)
Path(f"{DATASET_DIR}/semantic").mkdir(parents=True, exist_ok=True)
Path(f"{DATASET_DIR}/semantic_rgb").mkdir(parents=True, exist_ok=True)

for file in os.listdir(f"{rgb_prefix}/"):
    
    rgb_source_path = f"{rgb_prefix}/{file}"
    sem_source_path = f"{rgb_prefix}/{file}"

    rgb_destination_path = f"{DATASET_DIR}/rgb/{file}"
    sem_destination_path = f"{DATASET_DIR}/semantic/{file}"
    sem_rgb_destination_path = f"{DATASET_DIR}/semantic_rgb/{file}"

    img = convert_semantic_image(sem_source_path)
    shutil.copyfile(rgb_source_path, rgb_destination_path)
    shutil.copy(sem_source_path, sem_destination_path)
    cv2.imwrite(sem_rgb_destination_path, img)