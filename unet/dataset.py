import keras
import numpy as np

from skimage.transform import resize
from tensorflow.keras.preprocessing.image import load_img


class Carla(keras.utils.Sequence):
    
    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, start):
        i = start * self.batch_size
        batch_input_paths = self.input_img_paths[i: i + self.batch_size]
        batch_target_paths = self.target_img_paths[i : i + self.batch_size]
        
        # X is our input batch matrix
        # Y is our label batch matrix

        X = np.zeros((self.batch_size, self.img_size[0], self.img_size[1], 3), dtype="float32")
        Y = np.zeros((self.batch_size, self.img_size[0], self.img_size[1], 1), dtype="uint8")

        for index, path in enumerate(batch_input_paths):
            img = load_img(path, target_size=self.img_size)
            X[index] = img

        for index, path in enumerate(batch_target_paths):
            img = load_img(path)
            # Isolate the red channel as that's our labels
            labels, _, _ = img.split()
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

            Y[index] = labels
        
        return X, Y