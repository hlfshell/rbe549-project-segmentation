from semantic.carla_controller.labels import SEMANTIC_CATEGORIES
from semantic.unet.utils import (
    stitch_high_res_overlays_together,
    infer,
    labels_to_image,
    overlay_labels_img_on_input
)

import os
import numpy as np
import skvideo.io

from focal_loss import SparseCategoricalFocalLoss
from keras.models import load_model
from PIL import Image

# SOURCE_DIRECTORY = "./output/movie/"
SOURCE_DIRECTORY = "./output/movie2/"
OUTPUT_DIRECTORY = "./"
FPS = 12

def make_semantic_overlay():
    model = load_model("./unet_model_512x512_focal_loss_with_weights")

    # Load up a list of all files in order
    files = os.listdir(f"{SOURCE_DIRECTORY}/")
    import natsort
    files =  natsort.natsorted([file for file in files if file.endswith(".png") and not file.startswith(".")])
    # files = sorted([file for file in files if file.endswith(".png") and not file.startswith(".")])

    # Load a single image to get its size
    # initial_image = Image.open(f"{SOURCE_DIRECTORY}/{files[0]}")
    
    # Prepare the ffmpeg writer
    writer = skvideo.io.FFmpegWriter(
        f"{OUTPUT_DIRECTORY}/output_semantic.mp4",
        inputdict={'-r': str(FPS)}
    )

    # For each image in the folder, write to teh writer
    for file in files:
        img = Image.open(f"{SOURCE_DIRECTORY}/{file}")

        labels_img = stitch_high_res_overlays_together(model, img)
        results = overlay_labels_img_on_input(img, labels_img)

        # labels = infer(model, img)
        # labels_img = labels_to_image(labels, img.size)
        # results = overlay_labels_on_input(img, labels)

        frame = np.array(results, dtype= np.uint8)

        # frame_img = Image.fromarray(frame)
        # frame_img = labels_img

        # final_img = Image.new('RGB', (frame_img.width*2, frame_img.height))
        # final_img.paste(img, (0,0))
        # final_img.paste(frame_img, (frame_img.width, 0))

        # frame = np.array(final_img, dtype=np.uint8)

        writer.writeFrame(frame)

    # Close the writer
    writer.close()

if __name__ == "__main__":
    make_semantic_overlay()