# -*- coding: utf-8 -*-

import ipdb
import numpy as np
import torch
import torch.nn as nn

from imageBox import findValue





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

    bound = np.array([255, 0, 255, 0],
                     dtype=np.uint16)  # obtain the bound of smallest box covering the haze region - [row_top, row_bottom, vol_left, vol_right]

    # empty_haze_region = 1
    # flag = 1  # To divide haze region from black road on the bottom.. - indicate whether no patch-size haze is found in a row
    #
    # for i in range(patch_size_radius[0], m - patch_size_radius[0]):
    #
    #     if flag == 0 and empty_haze_region == 0:  # to ensure we just scan the first region at the upper side of the image
    #         break
    #         # 循环结束条件是图片的雾霾区域在行保持连续，即第一行存在雾霾区域，紧接着的下一行也应该存在雾霾区域
    #     flag = 0
    #
    #     for j in range(patch_size_radius[1], n - patch_size_radius[1]):
    #
    #         # filter
    #         batch = temp[(i - patch_size_radius[0]): (i + patch_size_radius[0]),
    #                 (j - patch_size_radius[1]): (j + patch_size_radius[1])]
    #         count = np.count_nonzero(batch == 255)
    #
    #
    #         if count == 0:  # if the patch does not contain surely non-haze pixels
    #
    #             # calculate the mask - image_filtered
    #             empty_haze_region = 0
    #
    #             flag = 1
    #
    #             mask[(i - patch_size_radius[0]): (i + patch_size_radius[0]),
    #             (j - patch_size_radius[1]): (j + patch_size_radius[1])] = 1
    #
    #             bound[0] = min(bound[0], i - patch_size_radius[0])
    #             bound[1] = max(bound[1], i + patch_size_radius[0] - 1)  # be careful to such edges
    #             bound[2] = min(bound[2], j - patch_size_radius[1])
    #             bound[3] = max(bound[3], j + patch_size_radius[1] - 1)

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
    row_top, row_bottom = findValue(np.unique(shape[0]))
    row_bottom += patch_size_radius[0]*2 - 1
    vol_left, vol_right = np.min(np.where(value[row_top:row_bottom] != 255)[1]), np.max(np.where(value[row_top:row_bottom] != 255)[1])
    vol_right += patch_size_radius[1]*2 - 2
    bound = np.array([row_top, row_bottom, vol_left, vol_right])

    shape1 = (shape[0] + patch_size_radius[0]*2, shape[1] + patch_size_radius[1]*2)

    for i in range(len(shape[0])):
        mask[shape[0][i]:shape1[0][i], shape[1][i]:shape1[1][i]] = 1
    mask[:, vol_right+1] = 0
    mask[row_bottom+1:] = 0

    # mm = mask - mask1
    # ipdb.set_trace()

    # 理论上来讲应该二者一样，但是有的图片会出现不同，怀疑跟maxpooling实现方式有关，比如realphoto的第一张图的57行

    return mask, bound