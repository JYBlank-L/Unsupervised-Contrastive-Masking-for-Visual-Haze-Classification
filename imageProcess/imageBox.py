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


def findValue(value, interval):
    min_num = np.min(value)
    temp = np.linspace(min_num, min_num + len(value), num=len(value), endpoint=True, dtype=np.int64)
    if len(np.where(temp != value)[0]) == 0:
        index = min_num + len(value) - 1
    else:
        flag = 1
        index = np.max(np.where(value - temp < interval)) - 1
        count = 1
        while flag > 0:
            if flag != count:
                break
            flag += 1
            if index + 1 < len(value) and value[index + 1] - temp[index + 1] < interval * (count + 1):
                count += 1
                new_interval = interval * count
                index = np.max(np.where(value - temp < new_interval))
            else:
                flag = 0
    row_top, row_bottom = min_num, index + min_num
    return row_top, row_bottom
