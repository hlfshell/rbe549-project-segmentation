from semantic.carla_controller.labels import SEMANTIC_COLORS

import numpy as np

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
