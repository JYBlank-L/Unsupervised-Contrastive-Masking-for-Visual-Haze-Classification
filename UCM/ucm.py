import numpy as np
from PIL import Image
from UCM.utils import openImage, contrastAnalysis, denoisedImage, maskImageCal, imageResize, getFilterHaze


def ucm(image_path, crop_size, filters, method=0):
    channel_filter_ratio, denoised_filter_ratio, mask_filter_ratio = filters
    # image is numpy array, imageFile is PIL file
    image, imageFile = openImage(image_path)
    darkChannelMap_dark, contrast = contrastAnalysis(image, channel_filter_ratio)
    m, n = contrast.shape
    try:
        DenoisedImage = denoisedImage(contrast, denoised_filter_ratio)
        # obtain mask image and boundary of the smallest box covering all haze regions
        mask, bound = maskImageCal(DenoisedImage, m, n, mask_filter_ratio)
        if bound[1] - bound[0] < crop_size[0] or bound[3] - bound[2] < crop_size[1]:
            haze_region = image
            mask = np.ones((m, n))
        else:
            haze_region = image[bound[0]:bound[1], bound[2]:bound[3]]
    except:
        haze_region = image
        mask = np.ones((m, n))
        # print("{} is in the evening".format(x + 1))

    # original_img = imageFile.resize(crop_size)
    resized_original_image = imageResize(imageFile, crop_size)

    if method != 2:
        haze_region_img = Image.fromarray(haze_region)
        # segment_img = haze_region_img.resize(crop_size)
        segment_resized_image = imageResize(haze_region_img, crop_size)
        return segment_resized_image, resized_original_image
    else:
        haze_filter = getFilterHaze(mask, image)
        haze_filter_img = Image.fromarray(haze_filter)
        filter_resized_image = imageResize(haze_filter_img, crop_size)
        return filter_resized_image, resized_original_image
