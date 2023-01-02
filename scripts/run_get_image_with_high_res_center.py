from keras.models import load_model
from semantic.unet.utils import infer, labels_to_image, overlay_labels_on_input
from PIL import Image
from semantic.unet.model import get_unet
from semantic.unet.utils import get_image_with_high_res_center, labels_to_image, rgb_image_to_input
import argparse


def main(filepath, output_dir, zoom):
    model = load_model("./checkpoints/unet.h5")
    img = Image.open(filepath)
    get_image_with_high_res_center(model, img, zoom=zoom, output_dir=output_dir)


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add an argument
    parser.add_argument('--filepath', type=str, required=False, default="./dataset/rgb/713883_3725.png")
    parser.add_argument('--output_dir', type=str, required=False, default="output")
    parser.add_argument('--zoom', type=float, required=False, default=2)

    # Parse the argument
    args = parser.parse_args()

    main(filepath=args.filepath, output_dir=args.output_dir, zoom=args.zoom)
