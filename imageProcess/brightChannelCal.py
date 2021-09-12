# -*- coding: utf-8 -*-
import ipdb
import numpy as np

from Maxpooling import brightChannelProcess




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

    #img = Image.fromarray(brightChannelMap)
    #img.save(path + '/' + 'brightChannelMap.png')

    return brightChannelMap