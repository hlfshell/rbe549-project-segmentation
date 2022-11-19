import keras
import numpy as np

from PIL import Image, ImageEnhance, ImageOps, ImageFilter
from random import randint, uniform
from skimage.transform import resize
from tensorflow.keras.preprocessing.image import load_img
from typing import Tuple

import skimage


class Carla(keras.utils.Sequence):
    
    def __init__(
            self,
            batch_size : int,
            img_size : Tuple[int, int],
            input_img_paths : str,
            target_img_paths : str,
            data_augmentation : bool = True
        ):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths
        self.apply_augmentation = data_augmentation

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def data_augmentation(self, img : Image, labels : np.ndarray):
        # 50% chance of flipping the image
        if randint(0,1):
            img = ImageOps.mirror(img)
            labels = np.flip(labels, axis=1)

        # Play with brightness +/- 40%
        brightness_adjust = 1.0 + uniform(-.40, 0.40)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness_adjust)
        
        # Play with contrast - 1.0 is original image. We'll be willing to go
        # a bit lower at 40% less contrast
        contrast_adjust = 1 - uniform(0.0, 0.40)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast_adjust)

        # Add blurring - the blur radius affects the blur so we'll go
        # from 0.0 to 5.0
        blur_radius = uniform(0.0, 5.0)
        img = img.filter(ImageFilter.GaussianBlur(blur_radius))

        # Add noise to the image
        noise_amount = uniform(0.0, 0.07)
        img_arr = np.asarray(img, dtype="uint8")
        img_arr = 255*skimage.util.random_noise(img_arr, mode='salt', amount=noise_amount)
        img = Image.fromarray(np.uint8(img_arr))

        return img, labels

    def __getitem__(self, start):
        i = start * self.batch_size
        batch_input_paths = self.input_img_paths[i: i + self.batch_size]
        batch_target_paths = self.target_img_paths[i : i + self.batch_size]
        
        # X is our input batch matrix
        # Y is our label batch matrix

        X = np.zeros((self.batch_size, self.img_size[0], self.img_size[1], 3), dtype="float32")
        Y = np.zeros((self.batch_size, self.img_size[0], self.img_size[1], 1), dtype="uint8")

        for index in range(0, len(batch_input_paths)):
            # Get our paths
            img_path = batch_input_paths[index]
            labels_path = batch_target_paths[index]

            # Load our RGB image
            img = load_img(img_path, target_size=self.img_size)
            
            #  Load and prepare our labels
            labels_img = load_img(labels_path)
            # Isolate the red channel as that's our labels
            labels, _, _ = labels_img.split()
            # We need this as an numpy array
            labels = np.array(labels, dtype="uint8")
            # Reduce our space forcibly to 1-13. Anything outside our
            # expected labels get set to 3 for OTHER
            labels[labels > 13] = 3
            # Reduce our labels to be from 0 to 12
            labels = labels - 1

            # Now that we have the np array of our label image, we need to resize it down
            # to the same size as our input image - *but* we must be sure to choose an
            # interpolation that uses nearest neighbor and avoids any kind of averaging
            # since that would be meaningless in a labels approach. This is what order=0
            # is doing below.
            # https://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.resize
            labels = resize(labels, self.img_size, order=0)

            # labels is (x, y) and we want labels to be (x, y, 1) in shape
            labels = np.expand_dims(labels, 2)

            # Apply data augmentation if requested
            if self.apply_augmentation:
                img, labels = self.data_augmentation(img, labels)

            # Assign out labels
            X[index] = img
            Y[index] = labels
        
        return X, Y