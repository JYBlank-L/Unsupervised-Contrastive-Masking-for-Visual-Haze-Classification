import numpy as np
from PIL import Image


def openImage(photo_path):
    imageFile = Image.open(photo_path)
    image = np.array(imageFile)
    return image


def getRGB(image):
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]
    return R, G, B
