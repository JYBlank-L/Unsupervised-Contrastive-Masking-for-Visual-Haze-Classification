import ipdb
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


def findValue(value):
    min_num = np.min(value)
    temp = np.linspace(min_num, min_num + len(value), num=len(value), endpoint=True, dtype=np.int64)
    index = np.min(np.where(temp != value)) - 1
    row_top, row_bottom = min_num, index + min_num
    return row_top, row_bottom
