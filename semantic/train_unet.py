from semantic.carla_controller.labels import SEMANTIC_CATEGORIES
from semantic.unet.model import get_unet

from datetime import datetime
from model import get_unet
from train import train_unet


img_size = (256, 256)

model = get_unet(img_size, len(SEMANTIC_CATEGORIES))
model = train_unet(model, 50, 32, img_size, test_size=0.25, dataset_folder="./dataset")

model.save(f"./unet_model_{datetime.utcnow()}")