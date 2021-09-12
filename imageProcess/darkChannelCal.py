# -*- coding: utf-8 -*-
import ipdb
import numpy as np
from Maxpooling import darkChannelProcess


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

    # img = Image.fromarray(darkChannelMap)
    # img.save(path + '/' + 'DarkChannelMap.png')

    return darkChannelMap
