# -*- coding: utf-8 -*-

import math
import numpy as np
from PIL import Image



def darkImageCal(image, path):
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

    #img = Image.fromarray(DarkImage)
    #img.save(path + '/' + 'DarkImage.png')


    return DarkImage