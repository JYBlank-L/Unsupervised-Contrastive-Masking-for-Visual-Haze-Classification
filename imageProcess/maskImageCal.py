# -*- coding: utf-8 -*-

import ipdb
import numpy as np
import torch
import torch.nn as nn





def maskImageCal(DenoisedImage, m, n):
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

    # This filter tries to find out available patches
    patch_size = np.array([m // 30, n // 30])  # smallest patch size we want
    patch_size_radius = patch_size // 2  # radius of filter

    bound = np.array([255, 0, 255, 0],
                     dtype=np.uint16)  # obtain the bound of smallest box covering the haze region - [row_top, row_bottom, vol_left, vol_right]

    empty_haze_region = 1
    flag = 1  # To divide haze region from black road on the bottom.. - indicate whether no patch-size haze is found in a row

    for i in range(patch_size_radius[0], m - patch_size_radius[0]):

        if flag == 0 and empty_haze_region == 0:  # to ensure we just scan the first region at the upper side of the image
            break
            # 循环结束条件是图片的雾霾区域在行保持连续，即第一行存在雾霾区域，紧接着的下一行也应该存在雾霾区域
        flag = 0

        for j in range(patch_size_radius[1], n - patch_size_radius[1]):

            if i == 392 and j == 131:
                ipdb.set_trace()

            # filter
            batch = temp[(i - patch_size_radius[0]): (i + patch_size_radius[0]),
                    (j - patch_size_radius[1]): (j + patch_size_radius[1])]
            count = np.count_nonzero(batch == 255)


            if count == 0:  # if the patch does not contain surely non-haze pixels

                # calculate the mask - image_filtered
                empty_haze_region = 0

                flag = 1

                mask[(i - patch_size_radius[0]): (i + patch_size_radius[0]),
                (j - patch_size_radius[1]): (j + patch_size_radius[1])] = 1

                bound[0] = min(bound[0], i - patch_size_radius[0])
                bound[1] = max(bound[1], i + patch_size_radius[0] - 1)  # be careful to such edges
                bound[2] = min(bound[2], j - patch_size_radius[1])
                bound[3] = max(bound[3], j + patch_size_radius[1] - 1)

    max_out = nn.MaxPool2d(kernel_size=[patch_size_radius[0]*2, patch_size_radius[1]*2],
                     stride=1,
                     return_indices=True)

    input = torch.from_numpy(temp.astype(float)).unsqueeze(0)
    output = max_out(input)
    values = output[0].squeeze().numpy()
    value = np.array(values)
    position = output[1].squeeze().numpy()
    shape = np.where(value != 255)
    value1 = np.unique(shape[0])
    ipdb.set_trace()

    return mask, bound, shape