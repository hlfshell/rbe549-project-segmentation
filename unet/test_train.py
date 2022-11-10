from model import get_unet
from train import train_unet


img_size = (256, 256)

model = get_unet(img_size, 13)
model = train_unet(model, 50, 32, img_size, test_size=0.25, dataset_folder="./dataset")

model.save("./unet_model")