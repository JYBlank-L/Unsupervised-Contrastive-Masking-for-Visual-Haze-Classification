import os
import scipy.io as sio
import numpy as np
from PIL import Image
from PIL import ImageFilter as IF
import time
import math
import random as rand
import ipdb

from contrastAnalysis import contrastAnalysis
from maskImageCal import maskImageCal
from hazePatchCal import hazePatchCal
from imageResize import imageResize
from loadLabel import loadLabel
from sample import sample
import imageBox

if __name__ == "__main__":
    # Paths
    root_list = ["../realphotos/", "../haze-level/"]
    photo_path_list = ["../realphotos/photos/", "../haze-level/images/"]
    label_path_list = ["../realphotos/labels.mat", "../haze-level/labels.txt"]
    type = 1  # 0 for realphotos, 1 for haze_level
    # Path to root folder
    root = root_list[type]
    # Path to read image: Load Image as a m*n*3 matrix
    photo_path = photo_path_list[type]
    # Path to labels
    label_path = label_path_list[type]
    # --------------------------------------------------------------------------------------------------------------------

    label = loadLabel(type, label_path)
    number_of_photos = label.shape[1]  # label is of size [1,number_of_photos]

    # Compute statistics of classes
    Percent_of_testing_samples = 0.1
    index_all = sample(label, Percent_of_testing_samples)

    # Obtain the flag of testing samples
    flag_test = np.zeros(number_of_photos)
    flag_test[index_all] = 1

    # --------------------------------------------------------------------------------------------------------------------
    # Process photos
    for x in range(0, number_of_photos):
        print("Start to process image %d." % (x + 1))

        # Set path to save internal results of each image
        result_path = root + "results/" + str(x + 1) + "/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        # read images
        if type == 0:
            image_path = photo_path + str(x + 1) + ".jpg"
        elif type == 1:
            image_path = photo_path + str(x + 1) + ".png"
        image = imageBox.openImage(image_path)
        R, G, B = imageBox.getRGB(image)

        # calculate the running time of the whole process
        # start_time = time.clock()
        start_time = time.perf_counter()

