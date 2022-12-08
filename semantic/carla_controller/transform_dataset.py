from semantic.carla_controller.labels import CARLA_SEMANTIC_COLORS

import cv2
import os

import numpy as np
from pathlib import Path
from random import randint
from shutil import copy


SOURCE_DIRECTORY = "./output"
DESTINATION_DIRECTORY = "./dataset"


def random_with_N_digits(n):
    range_start = 10**(n-1)
    range_end = (10**n)-1
    return randint(range_start, range_end)


def convert_semantic_image(image_path : str):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    sem_img = np.zeros_like(img)

    for key in CARLA_SEMANTIC_COLORS.keys():
        r_channel = (0, 0, key)
        color = CARLA_SEMANTIC_COLORS[key]

        sem_img[np.all(img == r_channel, axis=-1)] = color

    sem_img = cv2.cvtColor(sem_img, cv2.COLOR_RGB2BGR)

    return sem_img


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


if __name__ == "__main__":
    transform_dataset()