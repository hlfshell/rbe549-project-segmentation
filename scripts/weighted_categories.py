from semantic.carla_controller.labels import CARLA_SEMANTIC_CATEGORY_MAPPINGS, SEMANTIC_CATEGORIES

import numpy as np
import os

from PIL import Image

DATASET_DIR = "./dataset"
LABELS_DIR = f"{DATASET_DIR}/semantic"

labels_count = {}
total_pixels = 0
total_imgs = 0

for label in SEMANTIC_CATEGORIES.keys():
    labels_count[label] = 0

for filename in os.listdir(f"{LABELS_DIR}/"):
    img = Image.open(f"{LABELS_DIR}/{filename}").convert("RGB")
    r, g, b = img.split()
    img = np.asarray(r)

    total_imgs += 1
    width, height = img.shape
    total_pixels += width*height

    for label in CARLA_SEMANTIC_CATEGORY_MAPPINGS.keys():
        our_label = CARLA_SEMANTIC_CATEGORY_MAPPINGS[label]

        label_pixel_count = np.count_nonzero(img == label)
        labels_count[our_label] += label_pixel_count


print(f"Total Pixels: {total_pixels}")

print("Counts:")
print(labels_count)

print("Percentages:")
for label in labels_count.keys():
    print(f"{label} - {SEMANTIC_CATEGORIES[label]} - {labels_count[label]/total_pixels}")

print("Averages")
avgs = {}
for label in labels_count.keys():
    avgs[label] = labels_count[label] / total_imgs
per_avg = sum(avgs.values()) / len(SEMANTIC_CATEGORIES)
print(f"Average pixels per category - {per_avg}")
for label in avgs.keys():
    print(f"{label} - {SEMANTIC_CATEGORIES[label]} - {avgs[label]} - {per_avg/avgs[label]}")
