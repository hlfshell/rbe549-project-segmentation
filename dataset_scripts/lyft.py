# This script is made to read a random dataset from the Lyft Udacity Challenge.
# The challenge was given to Udacity students to try and create semantic
# segmentation networks for a given CARLA dataset. The dataset is split into
# 5 sub datasets, but each dataset is just an offset fraction fo a second from
# one-another. Since we added delays to our dataset to try and prevent too many
# similar images, we really only need one of these datasets. This still adds
# a thousand results for us, which is fine.
# Dataset is downloaded from:
# https://www.kaggle.com/datasets/kumaresanmanickavelu/lyft-udacity-challenge?resource=download

from semantic.carla_controller.transform_dataset import convert_semantic_image

import cv2
import os
import shutil


from pathlib import Path

LYFT_DATASET_DIR = "./lyft"
DATASET_DIR = "./dataset"
CHOSEN_DATASETS = ["dataA"]
LYFT_PREFIX = "lyft"

for dataset in CHOSEN_DATASETS:
    rgb_prefix = f"{LYFT_DATASET_DIR}/{dataset}/{dataset}/CameraRGB"
    semantic_prefix = f"{LYFT_DATASET_DIR}/{dataset}/{dataset}/CameraSeg"

    Path(f"{DATASET_DIR}/rgb").mkdir(parents=True, exist_ok=True)
    Path(f"{DATASET_DIR}/semantic").mkdir(parents=True, exist_ok=True)
    Path(f"{DATASET_DIR}/semantic_rgb").mkdir(parents=True, exist_ok=True)

    for file in os.listdir(f"{rgb_prefix}/"):
        rgb_source_path = f"{rgb_prefix}/{file}"
        sem_source_path = f"{semantic_prefix}/{file}"

        rgb_destination_path = f"{DATASET_DIR}/rgb/{LYFT_PREFIX}_{file}"
        sem_destination_path = f"{DATASET_DIR}/semantic/{LYFT_PREFIX}_{file}"
        sem_rgb_destination_path = f"{DATASET_DIR}/semantic_rgb/{LYFT_PREFIX}_{file}"

        img = convert_semantic_image(sem_source_path)
        shutil.copyfile(rgb_source_path, rgb_destination_path)
        shutil.copy(sem_source_path, sem_destination_path)
        cv2.imwrite(sem_rgb_destination_path, img)