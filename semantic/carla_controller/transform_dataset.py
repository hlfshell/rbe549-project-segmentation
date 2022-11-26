import carla
import cv2
import os

import numpy as np
from pathlib import Path
from random import randint
from shutil import copy
from typing import Dict


SOURCE_DIRECTORY = "./output"
DESTINATION_DIRECTORY = "./dataset"


def load_dataset_stats():
    # file_stats is an object 
    file_stats = {}

    directories = os.listdir(SOURCE_DIRECTORY)
    for dir in directories:
        files = os.listdir(f"{SOURCE_DIRECTORY}/{dir}/semantic")

        for file in files:
            pass


# def read_semantic_image(image_path : str) -> Dict:
#     img = cv2.imread(image_path, cv2.IMREAD_COLOR)

#     stats = {}


def random_with_N_digits(n):
    range_start = 10**(n-1)
    range_end = (10**n)-1
    return randint(range_start, range_end)


def convert_semantic_image(image_path : str):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    for key in SEMANTIC_CATEGORIES.keys():
        r_channel = (0, 0, key)
        category = SEMANTIC_CATEGORIES[key]
        color = SEMANTIC_COLORS[category]

        img[np.all(img == r_channel, axis=-1)] = color

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def transform_dataset():
    """
    Converts CARLAs semantic segmented images into a "human readable" image for easy observation and comparison.
    """
    Path(f"{DESTINATION_DIRECTORY}/rgb").mkdir(parents=True, exist_ok=True)
    Path(f"{DESTINATION_DIRECTORY}/semantic").mkdir(parents=True, exist_ok=True)
    Path(f"{DESTINATION_DIRECTORY}/semantic_rgb").mkdir(parents=True, exist_ok=True)

    directories = os.listdir(SOURCE_DIRECTORY)

    for dir in directories:
        files = os.listdir(f"{SOURCE_DIRECTORY}/{dir}/rgb")
        for file in files:
            id = random_with_N_digits(6)

            rgb_source_path = f"{SOURCE_DIRECTORY}/{dir}/rgb/{file}"
            semantic_source_path = f"{SOURCE_DIRECTORY}/{dir}/semantic/{file}"

            rgb_destination_path = f"{DESTINATION_DIRECTORY}/rgb/{id}_{file}"
            semantic_destination_path = f"{DESTINATION_DIRECTORY}/semantic/{id}_{file}"
            semantic_rgb_destination_path = f"{DESTINATION_DIRECTORY}/semantic_rgb/{id}_{file}"

            copy(rgb_source_path, rgb_destination_path)
            copy(semantic_source_path, semantic_destination_path)

            rgb_semantic = convert_semantic_image(semantic_source_path)
            cv2.imwrite(semantic_rgb_destination_path, rgb_semantic)


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


if __name__ == "__main__":
    transform_dataset()