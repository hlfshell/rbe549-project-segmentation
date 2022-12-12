from keras.models import load_model
from PIL import Image
from semantic.unet.utils import stitch_high_res_overlays_together
import argparse
from focal_loss import SparseCategoricalFocalLoss


def main(filepath, output_dir, sub_sections):
    # model = load_model("../unet.h5")
    model = load_model("../unet_512x512.h5")
    img = Image.open(filepath)
    stitch_high_res_overlays_together(model, img, sub_sections=sub_sections, output_dir=output_dir)


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add an argument
    # parser.add_argument('--filepath', type=str, required=False, default="../dataset/rgb/713883_3725.png")
    parser.add_argument('--filepath', type=str, required=False, default="../dataset/rgb/140502_1740.png")
    parser.add_argument('--output_dir', type=str, required=False, default="../output")
    parser.add_argument('--sub_sections', type=int, required=False, default=4)

    # Parse the argument
    args = parser.parse_args()

    main(filepath=args.filepath, output_dir=args.output_dir, sub_sections=args.sub_sections)
