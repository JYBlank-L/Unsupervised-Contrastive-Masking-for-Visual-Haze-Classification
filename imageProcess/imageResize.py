# -*- coding: utf-8 -*-
import ipdb
import numpy as np
from PIL import Image
from PIL import ImageOps as op



def imageResize(image, target_size, result_path):
    '''
    :param

    image: the image to be resized - must be a 2D matrix
    target_size: The size of the resized image, of size [height, width]


    :return

    resized_image: The resized image
    '''
    m, n, _ = image.shape


    image_ratio = m / n
    target_ratio = target_size[0] / target_size[1]

    # ---
    dark = image
    size = target_size
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    if image_ratio == target_ratio:  # if image is of the right size ratio - directly do resize

        img = Image.fromarray(dark)
        dark = op.fit(img, [size[1], size[0]], Image.ANTIALIAS)
        img = Image.fromarray(R)
        R = op.fit(img, [size[1], size[0]], Image.ANTIALIAS)
        img = Image.fromarray(G)
        G = op.fit(img, [size[1], size[0]], Image.ANTIALIAS)
        img = Image.fromarray(B)
        B = op.fit(img, [size[1], size[0]], Image.ANTIALIAS)

    else:

        # reduce rows/columns starting from central
        if image_ratio < target_ratio:  # if image is wider - reduce the width

            number_to_reduce_half = int((n - m / target_ratio) / 2)

            # ---------------------------------------------------------------------------------------------------
            dark = dark[:, number_to_reduce_half: (n - number_to_reduce_half) + 1]
            R = R[:, number_to_reduce_half: (n - number_to_reduce_half) + 1]
            G = G[:, number_to_reduce_half: (n - number_to_reduce_half) + 1]
            B = B[:, number_to_reduce_half: (n - number_to_reduce_half) + 1]

            img = Image.fromarray(dark)
            dark = op.fit(img, [size[1], size[0]], Image.ANTIALIAS)
            img = Image.fromarray(R)
            R = op.fit(img, [size[1], size[0]], Image.ANTIALIAS)
            img = Image.fromarray(G)
            G = op.fit(img, [size[1], size[0]], Image.ANTIALIAS)
            img = Image.fromarray(B)
            B = op.fit(img, [size[1], size[0]], Image.ANTIALIAS)

            # ---------------------------------------------------------------------------------------------------


        else:  # if the image is higher - reduce the height

            number_to_reduce_half = int((m - n * target_ratio) / 2)

            # ---------------------------------------------------------------------------------------------------
            dark = dark[number_to_reduce_half: (m - number_to_reduce_half) + 1, :]
            R = R[number_to_reduce_half: (m - number_to_reduce_half) + 1, :]
            G = G[number_to_reduce_half: (m - number_to_reduce_half) + 1, :]
            B = B[number_to_reduce_half: (m - number_to_reduce_half) + 1, :]

            img = Image.fromarray(dark)
            dark = op.fit(img, [size[1], size[0]], Image.ANTIALIAS)
            img = Image.fromarray(R)
            R = op.fit(img, [size[1], size[0]], Image.ANTIALIAS)
            img = Image.fromarray(G)
            G = op.fit(img, [size[1], size[0]], Image.ANTIALIAS)
            img = Image.fromarray(B)
            B = op.fit(img, [size[1], size[0]], Image.ANTIALIAS)

    dark.save(result_path + 'resized_image.png')


    return dark, R, G, B