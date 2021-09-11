# -*- coding: utf-8 -*-
import ipdb
import numpy as np
import time

from PIL import Image
from darkImageCal import darkImageCal
from brightImageCal import brightImageCal
from darkChannelCal import darkChannelCal
from brightChannelCal import brightChannelCal

def contrastAnalysis(image, path, result_path):
    '''
    :param

    image1 and image2: two images for contrast comparison, image1 is brighter than image2

    path: Save dark image under the path

    :return

    contrastMap: the map hopefully with haze preserved and object shaded in dark
    '''

    # ----------------------------------------------------------------------------------------------------------------------
    # Dark/bright image calculation
    image_dark = darkImageCal(image, path)
    image_bright = brightImageCal(image, path)

    img_dark = Image.fromarray(image_dark)
    img_dark.save(result_path + 'img_dark.png')

    img_bright = Image.fromarray(image_bright)
    img_bright.save(result_path + 'img_bright.png')
    # ----------------------------------------------------------------------------------------------------------------------





    # ----------------------------------------------------------------------------------------------------------------------
    # Dark/bright channel map calculation

    # channel maps of dark image
    print("Computing dark channel.")
    start_time = time.perf_counter()
    darkChannelMap_dark = darkChannelCal(image_dark, path)
    print("Running time for dark channel is %f sec." % (time.perf_counter() - start_time))

    # channel maps of bright image
    print("Computing bright channel.")
    start_time = time.perf_counter()
    brightChannelMap_bright = brightChannelCal(image_bright, path)
    print("Running time for bright channel is %f sec." % (time.perf_counter() - start_time))
    # ----------------------------------------------------------------------------------------------------------------------

    contrastMap = brightChannelMap_bright - darkChannelMap_dark
    #histogram = np.zeros((255))
    #m,n = contrastMap.shape

    #used for checking whether there is a clear seperation between pixels with low and high intensity
    #for i, j in np.ndindex(m, n):
    #   histogram[contrastMap[i,j]-1] = histogram[contrastMap[i,j]-1]+1

    mean = np.mean(contrastMap)
    median = np.median(contrastMap)

    div = (mean + median) / 2
    contrastMap[contrastMap > div] = 255
    img = Image.fromarray(brightChannelMap_bright)
    img.save(path + '/' + 'brightChannelMap_bright_image.png')


    #mean_intensity = np.mean(image)
    #median_intensity = np.median(image)
    #threshold = (mean_intensity + median_intensity) / 2

    mean_image = np.mean(image,axis=2)

    contrastMap[np.where(mean_image < 100)] = 255 #Here use a trick that haze region pixels have intensity >= 100

    return contrastMap, darkChannelMap_dark