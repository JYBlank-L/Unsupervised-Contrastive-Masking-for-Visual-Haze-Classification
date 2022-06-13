import time

import numpy as np
from PIL import Image
from ucm import ucm
import matplotlib.pyplot as plt

image = './3024.png'
stime = time.time()
a, b = ucm(image, crop_size=(64, 64), filters=(40, 40, 70))
print(time.time()-stime)

