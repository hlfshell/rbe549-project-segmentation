import keras
import os

from unet.dataset import Carla

from sklearn.model_selection import train_test_split
from typing import Tuple


def train_unet(
        model,
        epochs : int,
        batch_size : int,
        img_size : Tuple[int, int] = (128,128),
        test_size : float = 0.25,
        dataset_folder : str = "./dataset"
    ):
    
    rgb_folder = f"{dataset_folder}/rgb"
    rgb_paths = sorted(
        [
            os.path.join(rgb_folder, fname)
            for fname in os.listdir(rgb_folder)
            if fname.endswith(".png")
        ]
    )

    label_folder = f"{dataset_folder}/semantic"
    label_paths = sorted(
        [
            os.path.join(label_folder, fname)
            for fname in os.listdir(label_folder)
            if fname.endswith(".png") and not fname.startswith(".")
        ]
    )

    train_rgb_paths, validation_rgb_paths, train_label_paths, validation_label_paths = train_test_split(
        rgb_paths, label_paths, test_size=test_size
    )

    training_generator = Carla(batch_size, img_size, train_rgb_paths, train_label_paths)
    validation_generator = Carla(batch_size, img_size, validation_rgb_paths, validation_label_paths, data_augmentation=False)

    # Configure the model for training.
    # We use the "sparse" version of categorical_crossentropy
    # because our target data is integers.
    model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")

    callbacks = [
        keras.callbacks.ModelCheckpoint("carla_segmentation.h5", save_best_only=True)
    ]

    model.fit(training_generator, epochs=epochs, validation_data=validation_generator, callbacks=callbacks)

    return model