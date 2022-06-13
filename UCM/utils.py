import torch
import numpy as np
import torch.nn as nn
import torchvision
from PIL import Image, ImageOps
from PIL import ImageFilter as IF


def brightChannelProcess(image, patch_size_radius):
    '''
    :param

    image: gray image - can be dark or bright image

    patch_size_radius: the radius of patch, equal to kernel_size()/2

    :return

    brightChannelMap: The bright channel map for haze estimation
    '''

    m = nn.MaxPool2d(kernel_size=patch_size_radius*2, stride=1, padding=patch_size_radius)
    input = torch.from_numpy(image.astype(float)).unsqueeze(0)

    brightChannelMap = m(input).squeeze().numpy()[:, :-1]

    return brightChannelMap

def darkChannelProcess(image, patch_size_radius):
    '''
    :param

    image: gray image - can be dark or bright image

    patch_size_radius: the radius of patch, equal to kernel_size()/2

    :return

    darkChannelMap: The dark channel map for haze estimation
    '''

    m = nn.MaxPool2d(kernel_size=patch_size_radius*2, stride=1, padding=patch_size_radius)
    image = image*(-1)
    input = torch.from_numpy(image.astype(float)).unsqueeze(0)

    darkChannelMap = m(input).squeeze().numpy()[:, :-1]*(-1)

    return darkChannelMap

def brightImageCal(image):
    '''
    #build and save the dark image: The gray image with highest values in RGB channels

    :param

    image: The original image

    path: Save bright image under the path

    :return

    BrightImage: The bright image
    '''

    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    BrightImage = np.maximum(B, np.maximum(R, G))

    return BrightImage

def darkImageCal(image):
    '''
    #build and save the dark image: The gray image with lowest values in RGB channels

    :param

    image: The original image

    path: Save dark image under the path

    :return

    DarkImage: The dark image
    '''

    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    DarkImage = np.minimum(B, np.minimum(R, G))

    return DarkImage

def brightChannelCal(image, filter):
    '''
    :param

    image: gray image - can be dark or bright image

    path: Save dark image under the path

    :return

    brightChannelMap: The bright channel map for haze estimation
    '''


# ---------calculate darkChannelMap according to dark channel prior------------
    m, n = image.shape
    patch_size_radius = min(m, n) // (filter * 2)  # make filter size to be 1/40 of the shorter length of the image

    brightChannelMap = brightChannelProcess(image, patch_size_radius).astype(np.uint8)

    return brightChannelMap

def darkChannelCal(image, filter):
    '''
    :param

    image: gray image - can be dark or bright image

    path: Save dark image under the path

    :return

    darkChannelMap: The dark channel map for haze estimation
    '''

    # ---------calculate darkChannelMap according to dark channel prior------------
    m, n = image.shape
    patch_size_radius = min(m, n) // (filter * 2)  # make filter size to be 1/40 of the shorter length of the image

    darkChannelMap = darkChannelProcess(image, patch_size_radius).astype(np.uint8)

    return darkChannelMap

def openImage(photo_path):
    imageFile = Image.open(photo_path)
    imageFile = ImageOps.exif_transpose(imageFile)
    image = np.array(imageFile)
    return image, imageFile

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
        index = np.max(np.where(value - temp < interval))
        while True:
            flag += 1
            if index + 1 < len(value) and value[index + 1] - temp[index + 1] < interval + value[index] - temp[index]:
                new_interval = interval + value[index] - temp[index]
                index = np.max(np.where(value - temp < new_interval))
            else:
                break
    row_top, row_bottom = min_num, index + min_num
    return row_top, row_bottom

def contrastAnalysis(image, filter):
    '''
    :param

    image1 and image2: two images for contrast comparison, image1 is brighter than image2

    path: Save dark image under the path

    :return

    contrastMap: the map hopefully with haze preserved and object shaded in dark
    '''

    # ----------------------------------------------------------------------------------------------------------------------
    # Dark/bright image calculation
    image_dark = darkImageCal(image)
    image_bright = brightImageCal(image)

    # ----------------------------------------------------------------------------------------------------------------------
    # Dark/bright channel map calculation

    # channel maps of dark image
    darkChannelMap_dark = darkChannelCal(image_dark, filter)

    # channel maps of bright image
    brightChannelMap_bright = brightChannelCal(image_bright, filter)
    # ----------------------------------------------------------------------------------------------------------------------

    contrastMap = brightChannelMap_bright - darkChannelMap_dark

    mean = np.mean(contrastMap)
    median = np.median(contrastMap)

    div = (mean + median) / 2
    contrastMap[contrastMap > div] = 255

    contrastMap1 = contrastMap
    contrastMap2 = contrastMap1 + 0

    mean_image = np.mean(image, axis=2)
    R, G, B = getRGB(image)
    gray = np.mean(0.3 * R + 0.33 * G + 0.45 * B)

    contrastMap1[np.where(mean_image < 100)] = 255  # Here use a trick that haze region pixels have intensity >= 100

    temp = contrastMap2 + 0
    temp[np.where(mean_image < gray)] = 255
    ratio = np.where(temp != 255)
    ratio1 = np.where(contrastMap1 != 255)
    # Here if the size of contrastMap1 is smaller than 0.2*contrastMap2, we choose contrastMap1 as the final contrast map
    if len(ratio[0]) <= 0.2 * len(ratio1[0]):
        contrastMap2 = contrastMap1
    else:
        contrastMap2 = temp

    return darkChannelMap_dark, contrastMap2

def denoisedImage(contrastMap_image, denoised_filter):

    m, n = contrastMap_image.shape
    patch_size_radius = min(m, n) // (denoised_filter * 2)  # radius of filter

    img = Image.fromarray(contrastMap_image)

    DenoisedImage = img.filter(IF.ModeFilter(patch_size_radius))

    return DenoisedImage

def maskImageCal(DenoisedImage, m, n, mask_filter_ratio):
    '''
    :param

    DenoisedImage: gray image - contrast map after Gaussian smoothing

    (m,n): the size of original image


    :return

    mask: binary matrix with 1 indicating a pixel of haze region
    bound: the box covering all haze regions
    '''

    temp = np.array(DenoisedImage)

    mask = np.zeros((m, n), dtype=np.uint8)  # operate on temp and store value to image_filtered
    mask1 = np.zeros((m, n), dtype=np.uint8)  # operate on temp and store value to image_filtered

    # This filter tries to find out available patches
    patch_size = np.array([m // mask_filter_ratio, n // mask_filter_ratio])  # smallest patch size we want
    patch_size_radius = patch_size // 2  # radius of filter

    flag = 1
    while flag == 1:
        max_out = nn.MaxPool2d(kernel_size=[patch_size_radius[0] * 2, patch_size_radius[1] * 2], stride=1)

        input = torch.from_numpy(temp.astype(float)).unsqueeze(0)
        output = max_out(input)
        value = output.squeeze().numpy()
        value = np.array(value)
        shape = np.where(value != 255)
        if len(shape[0]) == 0:
            patch_size_radius = patch_size_radius // 2
            continue
        flag = 0
    row_top, row_bottom = findValue(np.unique(shape[0]), m // 2)
    row_bottom += patch_size_radius[0]*2 - 1
    vol_left, vol_right = np.min(np.where(value[row_top:row_bottom] != 255)[1]), np.max(np.where(value[row_top:row_bottom] != 255)[1])
    vol_right += patch_size_radius[1]*2 - 2
    bound1 = np.array([row_top, row_bottom, vol_left, vol_right])

    shape1 = (shape[0] + patch_size_radius[0]*2, shape[1] + patch_size_radius[1]*2)

    for i in range(len(shape[0])):
        mask1[shape[0][i]:shape1[0][i], shape[1][i]:shape1[1][i]] = 1
    mask1[:, vol_right+1] = 0
    mask1[row_bottom+1:] = 0

    return mask1, bound1

def imageResize(image, target_size):
    """
    this def is defined to crop the image by using centerCrop then resize it to the target size
    :param
    image: the image to be resized - must be a 2D matrix
    target_size: The size of the resized image, of size [height, width]

    :return
    resized_image: The resized image
    """

    m, n = image.size
    size = min(m, n)
    center_crop = torchvision.transforms.CenterCrop(size)
    center_image = center_crop(image)
    resize_image = center_image.resize(target_size)
    return resize_image

def getFilterHaze(mask, image):
    mask_RGB = np.zeros((image.shape[0], image.shape[1], 3))
    if mask.shape[0] != image.shape[0]:
        a = image.shape[0] - mask.shape[0]
        if a > 0:
            a = -1 * a
            mask = mask[0:a, :]
    if mask.shape[1] != image.shape[1]:
        b = image.shape[1] - mask.shape[1]
        if b > 0:
            b = -1 * b
            mask = mask[:, 0:b]
    mask_RGB[:, :, 0] = mask * image[:, :, 0]
    mask_RGB[:, :, 1] = mask * image[:, :, 1]
    mask_RGB[:, :, 2] = mask * image[:, :, 2]
    mask_RGB = np.uint8(mask_RGB)
    mask_RGB_img = Image.fromarray(mask_RGB)
    return mask_RGB_img
