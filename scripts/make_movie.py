import os
import numpy as np
import skvideo.io

from PIL import Image

SOURCE_DIRECTORY = "./output/movie/"
OUTPUT_DIRECTORY = "./"
FPS = 12

def make_movie():
    # Load up a list of all files in order
    files = os.listdir(f"{SOURCE_DIRECTORY}/")
    files = sorted([file for file in files if file.endswith(".png") and not file.startswith(".")])

    # Load a single image to get its size
    # initial_image = Image.open(f"{SOURCE_DIRECTORY}/{files[0]}")
    
    # Prepare the ffmpeg writer
    writer = skvideo.io.FFmpegWriter(
        f"{OUTPUT_DIRECTORY}/output.mp4",
        inputdict={'-r': str(FPS)}
    )

    # For each image in the folder, write to teh writer
    for file in files:
        frame = np.array(Image.open(f"{SOURCE_DIRECTORY}/{file}"), dtype= np.uint8)
        writer.writeFrame(frame)

    # Close the writer
    writer.close()

if __name__ == "__main__":
    make_movie()