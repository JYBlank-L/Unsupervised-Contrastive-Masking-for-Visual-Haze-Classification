# -*- coding: utf-8 -*-


import numpy as np




def brightImageCal(image, path):
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

    #img = Image.fromarray(BrightImage)
    #img.save(path + '/' + 'BrightImage.png')


    return BrightImage