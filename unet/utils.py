
import numpy as np

from math import floor
from PIL import Image
from skimage.transform import resize
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

    return model.predict(img_input)


def rgb_image_to_input(img: Image, input_size : Optional[Tuple[int, int]] = None) -> np.ndarray:
    nn_input = np.zeros((1,) + input_size + (3,), dtype="float32")
    
    if input_size is not None and img.size != input_size + (3,):
        resize(np.array(img), input_size)
    
    nn_input[0] = nn_input

    return nn_input


def labels_to_image(labels : np.ndarray, output_size : Optional[Tuple[int, int]] = None) -> Image:
    # Reduce dimensionality - instead of one hot encoded pixels, do a singular dimension
    mask = np.argmax(labels, axis=-1)
    
    # This gets us to a shape of (1, width, height) - we want just (width, height)
    mask = mask.reshape(mask.shape[1], mask.shape[2])

    # Resize to the output size if necessary
    if output_size is not None and output_size != labels.shape:
        mask = resize(mask, output_size, order=0)
    
    # Next we convert the resized labels to an rgb set
    img = np.zeros(mask.shape + (3,), dtype="uint8")

    for key in CLASS_COLORS.keys():
        img[np.all(mask == key, axis=-1)] = CLASS_COLORS[key]

    # PIL expects height by width, so we have to adjust what we feed it
    return Image.fromarray(img.transpose(1,0,2))


def overlay_labels_on_input(img: Image, labels : np.ndarray, alpha : float = 0.4) -> Image:
    labels_img = labels_to_image(labels, output_size=img.size).convert('RGBA')

    # Pillow expects alpha to be 0 (full transparency) to 255 (full opaque)
    # so convert our percentage to an integer
    labels_img.putalpha(floor(255 * alpha))
    print(img.size, labels_img.size)
    return Image.alpha_composite(img, labels_img)


# These labels are taken directly from CARLA
CLASS_COLORS = {
    0: [0, 0, 0],         # None
    1: [70, 70, 70],      # Buildings
    2: [190, 153, 153],   # Fences
    3: [72, 0, 90],       # Other
    4: [220, 20, 60],     # Pedestrians
    5: [153, 153, 153],   # Poles
    6: [157, 234, 50],    # RoadLines
    7: [128, 64, 128],    # Roads
    8: [244, 35, 232],    # Sidewalks
    9: [107, 142, 35],    # Vegetation
    10: [0, 0, 255],      # Vehicles
    11: [102, 102, 156],  # Walls
    12: [220, 220, 0]     # TrafficSigns
}