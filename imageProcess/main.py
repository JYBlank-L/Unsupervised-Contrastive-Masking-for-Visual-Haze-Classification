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
from deNosied import denoisedImage
import imageBox

if __name__ == "__main__":
    # Paths
    root_list = ["../realphotos/", "../haze-level/"]
    photo_path_list = ["../realphotos/photos/", "../haze-level/images/"]
    label_path_list = ["../realphotos/labels.mat", "../haze-level/labels.txt"]
    type = 0  # 0 for realphotos, 1 for haze_level
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
    channel_filter_ratio = 40
    denoised_filter_ratio = 20
    mask_filter_ratio = 60

    # --------------------------------------------------------------------------------------------------------------------
    # Process photos
    for x in range(3, 5):
        print("Start to process image %d." % (x + 1))

        # Set path to save internal results of each image
        result_path = root + "results" + str(channel_filter_ratio) + str(denoised_filter_ratio) + str(mask_filter_ratio) \
                      + "/" + str(x + 1) + "/"
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

        # Compute haze regions
        # Contrast map calculation
        contrast_time = time.perf_counter()
        contrastMap_image, darkChannelMap_dark, contrastMap2, contrastMap3 = contrastAnalysis(image, result_path,
                                                                                              channel_filter_ratio)
        print("Running time for contrast time is %f sec." % (time.perf_counter() - contrast_time))

        img = Image.fromarray(contrastMap_image)
        img.save(result_path + 'contrastMapImage.png')
        darkChannelMap_dark_image = Image.fromarray(darkChannelMap_dark)
        darkChannelMap_dark_image.save(result_path + 'darkChannelMap_dark_image1.png')

        contrast = [contrastMap_image, contrastMap2, contrastMap3]
        m, n = contrastMap_image.shape
        for i in range(len(contrast)):
            denoised_time = time.perf_counter()
            DenoisedImage = denoisedImage(contrast[i], denoised_filter_ratio)
            DenoisedImage.save(result_path + 'DenoisedImage' + str(i) + '.png')
            print("Running time for denoised time is %f sec." % (time.perf_counter() - denoised_time))

            # obtain mask image and boundary of the smallest box covering all haze regions
            mask_time = time.perf_counter()
            mask, bound = maskImageCal(DenoisedImage, m, n, mask_filter_ratio)
            print("Running time for mask time is %f sec." % (time.perf_counter() - mask_time))
            haze_region_dark = image[bound[0]:bound[1], bound[2]:bound[3]]
            haze_region_img = Image.fromarray(haze_region_dark)
            haze_region_img.save(result_path + 'hazeRegionImage' + str(i) + '.png')
        # ipdb.set_trace()

        # resize the images for training and testing data
        # target_size = np.array([224, 224])  # The size of images (height, width) we want to obtain
        # resized_image = imageResize(haze_region_dark, target_size, result_path)

