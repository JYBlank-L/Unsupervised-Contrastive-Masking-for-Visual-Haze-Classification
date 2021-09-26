# -*- coding: utf-8 -*-
import ipdb
import numpy as np
import time

from PIL import Image
from darkImageCal import darkImageCal
from brightImageCal import brightImageCal
from darkChannelCal import darkChannelCal
from brightChannelCal import brightChannelCal

def contrastAnalysis(image, path, filter):
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
    img_dark.save(path + 'img_dark.png')

    img_bright = Image.fromarray(image_bright)
    img_bright.save(path + 'img_bright.png')
    # ----------------------------------------------------------------------------------------------------------------------
    # Dark/bright channel map calculation

    # channel maps of dark image
    print("Computing dark channel.")
    start_time = time.perf_counter()
    darkChannelMap_dark = darkChannelCal(image_dark, filter)
    print("Running time for dark channel is %f sec." % (time.perf_counter() - start_time))

    # channel maps of bright image
    print("Computing bright channel.")
    start_time = time.perf_counter()
    brightChannelMap_bright = brightChannelCal(image_bright, filter)
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
    contrast_img = Image.fromarray(contrastMap)
    contrast_img.save(path + '/' + 'contrast_img.png')
    img = Image.fromarray(brightChannelMap_bright)
    img.save(path + '/' + 'brightChannelMap_bright_image.png')


    #mean_intensity = np.mean(image)
    #median_intensity = np.median(image)
    #threshold = (mean_intensity + median_intensity) / 2

    mean_image = np.mean(image,axis=2)

    contrastMap1 = contrastMap
    contrastMap2 = contrastMap1 + 0
    contrastMap3 = contrastMap2 + 0

    contrastMap1[np.where(mean_image < 100)] = 255 #Here use a trick that haze region pixels have intensity >= 100
    contrast1_img = Image.fromarray(contrastMap1)
    contrast1_img.save(path + '/' + 'contrast1_img.png')

    contrastMap3[np.where(mean_image < 130)] = 255
    contrast3_img = Image.fromarray(contrastMap3)
    contrast3_img.save(path + '/' + 'contrast3_img.png')

    contrastMap2[np.where(mean_image < 150)] = 255
    contrast2_img = Image.fromarray(contrastMap2)
    contrast2_img.save(path + '/' + 'contrast2_img.png')


    return contrastMap1, darkChannelMap_dark, contrastMap2, contrastMap3