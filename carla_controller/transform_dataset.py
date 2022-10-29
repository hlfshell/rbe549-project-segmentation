from typing import Dict
import cv2
import os

import numpy as np
from pathlib import Path
from shutil import copy


SOURCE_DIRECTORY = "./output"
DESTINATION_DIRECTORY = "./datasets"


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


def convert_semantic_image(image_path : str):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    for key in SEMANTIC_CATEGORIES.keys():
        r_channel = (0, 0, key)
        category = SEMANTIC_CATEGORIES[key]
        color = SEMANTIC_COLORS[category]

        img[np.all(img == r_channel, axis=-1)] = color
    
    return img


def transform_dataset():
    Path(f"{DESTINATION_DIRECTORY}/rgb").mkdir(parents=True, exist_ok=True)
    Path(f"{DESTINATION_DIRECTORY}/semantic").mkdir(parents=True, exist_ok=True)
    Path(f"{DESTINATION_DIRECTORY}/semantic_rgb").mkdir(parents=True, exist_ok=True)

    directories = os.listdir(SOURCE_DIRECTORY)

    for dir in directories:
        files = os.listdir(f"{SOURCE_DIRECTORY}/{dir}/rgb")
        for file in files:
            rgb_source_path = f"{SOURCE_DIRECTORY}/{dir}/rgb/{file}"
            semantic_source_path = f"{SOURCE_DIRECTORY}/{dir}/semantic/{file}"
            semantic_rgb_source_path = f"{SOURCE_DIRECTORY}/{dir}/semantic_rgb/{file}"

            rgb_destination_path = f"{DESTINATION_DIRECTORY}/rgb/{file}"
            semantic_destination_path = f"{DESTINATION_DIRECTORY}/semantic/{file}"
            semantic_rgb_destination_path = f"{DESTINATION_DIRECTORY}/semantic_rgb/{file}"

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
    "Buildings": (97, 85, 44),
    "Fences": (200, 119, 46),
    "Other": (78, 76, 43),
    "Pedestrians": (71, 183, 73),
    "Poles": (30, 72, 105),
    "RoadLines": (240, 227, 103),
    "Roads": (38, 54, 112),
    "Sidewalks": (242, 201, 209),
    "Vegetation": (54, 110, 79),
    "Vehicles": (32, 155, 199),
    "Walls": (93, 60, 33),
    "TrafficSigns": (245, 135, 113),
}


if __name__ == "__main__":
    transform_dataset()