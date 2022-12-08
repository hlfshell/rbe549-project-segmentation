from semantic.carla_controller.labels import SEMANTIC_COLORS

import numpy as np
import os

from math import floor
from PIL import Image
from skimage.transform import resize
from tensorflow import keras
from typing import Optional, Tuple


def infer(model, img : Image) -> np.ndarray:
    '''
    infer takes a model and an image; it does all necessary
    setup and processes the image, returning a set of
    labeled pixels    
    '''
    # Get input size of network
    input_size = model.layers[0].get_output_at(0).get_shape().as_list()
    # input_size is a list with the following options:
    # [batch_size (None), width, height, channels]. Really we just want
    # width and height:
    input_size = (input_size[1], input_size[2])

    img_input = rgb_image_to_input(img, input_size=input_size)

    return model.predict(img_input)[0]


def rgb_image_to_input(img: Image, input_size : Optional[Tuple[int, int]] = None) -> np.ndarray:
    # Ensure that we have a 3 channel RGB image (RGBA breask this)
    img = img.convert("RGB")

    # We prepare our input - a (1, w, h, 3) tensor where the first 1 is our batch size
    nn_input = np.zeros((1,) + input_size + (3,), dtype="float32")
    
    # Resize image down
    if input_size is not None and img.size != input_size + (3,):
        img = img.resize(input_size)
    
    nn_input[0] = img

    return nn_input


def labels_to_image(labels : np.ndarray, output_size : Optional[Tuple[int, int]] = None) -> Image:
    # Reduce dimensionality - instead of one hot encoded pixels, do a singular dimension
    mask = np.argmax(labels, axis=-1)
    
    # This gets us to a shape of (width, height) - we want (width, height, 1)
    mask = np.expand_dims(mask, axis=-1)

    # Resize to the output size if necessary. Note that PIL expects a differently
    # ordered image, so we reverse the dimensions
    if output_size is not None and output_size != labels.shape:
        # mask = resize(mask, tuple(reversed(output_size)), order=0)
        mask = resize(mask, output_size[::-1], order=0)
    
    # Next we convert the resized labels to an rgb set
    img = np.zeros(mask.shape[0:2] + (3,), dtype="uint8")

    for key in SEMANTIC_COLORS.keys():
        img[np.all(mask == key, axis=-1)] = SEMANTIC_COLORS[key]

    return Image.fromarray(img)


def overlay_labels_on_input(img: Image, labels : np.ndarray, alpha : float = 0.4) -> Image:
    labels_img = labels_to_image(labels, output_size=img.size).convert('RGBA')

    # Pillow expects alpha to be 0 (full transparency) to 255 (full opaque)
    # so convert our percentage to an integer
    labels_img.putalpha(floor(255 * alpha))
    print(img.size, labels_img.size)
    return Image.alpha_composite(img, labels_img)


def get_image_with_high_res_center(model, img : Image, zoom : float = 2, output_dir: str = "output") -> np.ndarray:
    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get input size of network
    input_size = model.layers[0].get_output_at(0).get_shape().as_list()
    # input_size is a list with the following options:
    # [batch_size (None), width, height, channels]. Really we just want
    # width and height:
    input_size = (input_size[1], input_size[2])

    # Get the low res image as an input for the model
    low_res_input = rgb_image_to_input(img, input_size=input_size)

    # Get the dimensions of the original image
    height, width = img.size
    # print(img.size)

    # Determine the crop points based on the zoom level
    crop_points = (
        int(height/2-height/(2*zoom)),
        int(width/2-width/(2*zoom)),
        int(height/2+height/(2*zoom)),
        int(width/2+width/(2*zoom))
    )
    # print(crop_points)

    # Crop the image to allow for a zoomed and higher resolution image and convert for model input
    img_cropped = img.crop(crop_points)
    # print(img_cropped.size)
    high_res_input = rgb_image_to_input(img_cropped, input_size=input_size)

    # Make predictions of both low- and high-res images
    low_res_output_labels = model.predict(low_res_input)[0]
    high_res_output_labels = model.predict(high_res_input)[0]

    # Convert both model predictions into images
    final_image = labels_to_image(low_res_output_labels, img.size)
    final_image_cropped = labels_to_image(high_res_output_labels, img_cropped.size)

    # Save some images to help debug
    img.resize(input_size).save(os.path.join(output_dir, "img.png"))  # original image
    img_cropped.resize(input_size).save(os.path.join(output_dir, "img_cropped.png"))  # original cropped image
    final_image.save(os.path.join(output_dir, "final_image1.png"))  # Output of original image
    final_image_cropped.save(os.path.join(output_dir, "final_image_2.png"))  # Output of cropped image

    # Paste the cropped image into the center of the original image and save it
    img_w, img_h = final_image_cropped.size
    bg_w, bg_h = final_image.size
    offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
    fi3 = final_image
    fi3.paste(final_image_cropped, offset)
    fi3.save(os.path.join(output_dir, "fi3.png"))

def stitch_high_res_overlays_together(model, img : Image, sub_sections : int = 4, output_dir: str = None) -> np.ndarray:
    """

    :param str output_dir: If it exists, same the images in this directory
    """

    if output_dir:
        # Create the output directory if it does not exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # Get input size of network
    input_size = model.layers[0].get_output_at(0).get_shape().as_list()
    # input_size is a list with the following options:
    # [batch_size (None), width, height, channels]. Really we just want
    # width and height:
    input_size = (input_size[1], input_size[2])

    # Get the low res image as an input for the model
    low_res_input = rgb_image_to_input(img, input_size=input_size)

    # Make predictions of low- image
    low_res_output_labels = model.predict(low_res_input)[0]

    # Convert low-res model predictions into images
    base_image = labels_to_image(low_res_output_labels, img.size)
    if output_dir:
        base_image.save(os.path.join(output_dir, "base_image.png"))
    width, height = img.size
    subsection_width = width/sub_sections
    final_image = base_image
    for i in range(sub_sections):
        # print("Working on subection {}".format(i))
        # Determine the crop points based on the zoom level
        crop_points = (
            int(subsection_width*i),
            int((height-subsection_width)/2),
            int(subsection_width*(i+1)),
            int((height+subsection_width)/2),
        )
        print(crop_points)

        # Crop the image to allow for a zoomed and higher resolution image and convert for model input
        img_cropped = img.crop(crop_points)

        high_res_input = rgb_image_to_input(img_cropped, input_size=input_size)

        # Get the output of the high res image
        high_res_output_labels = model.predict(high_res_input)[0]

        # Get image of high res output
        print(img_cropped.size)
        overlay_image = labels_to_image(high_res_output_labels, img_cropped.size)

        # Paste the cropped image into the center of the original image and save it
        img_w, img_h = base_image.size
        bg_w, bg_h = overlay_image.size
        # offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
        overlay_np = np.array(high_res_output_labels)  # Convert the labels into an np.array
        # https://note.nkmk.me/en/python-pillow-composite/
        # 0 is white, 255 is black.  White pixels will allow for overwrites
        mask = np.argmax(high_res_output_labels, axis=-1)
        mask = np.isin(mask, [1, 6, 7]).astype(int)*255  # only update pixels where the label value is specified
        mask = resize(mask, overlay_image.size[::-1], order=0)

        # Convert numpy mask to pillow image
        mask_img = Image.fromarray(mask)
        mask_img = mask_img.convert("RGBA")  # Convert to 4 channel RGBA, needed for paste below

        if output_dir:
            # Save component images
            mask_img.save(os.path.join(output_dir, "mask_{}.png".format(i)))
            overlay_image.save(os.path.join(output_dir, "overlay_image_{}.png".format(i)))

        # Paste this overlay using the same crop points from above and save the image
        final_image.paste(overlay_image, box=crop_points, mask=mask_img)
        # final_image = Image.composite(final_image, overlay_image, box=crop_points, mask=im)

        if output_dir:
            final_image.save(os.path.join(output_dir, "fi3_{}.png".format(i)))

    return final_image
