# -*- coding: utf-8 -*-


import numpy as np
from PIL import Image




def hazePatchCal(patch_size, max_patch_per_image, m, n, bound, mask, number_of_patch, darkChannelMap_dark,R, G, B, result_path):
    '''
    :param

    patch_size, max_patch_per_image: define the required patches

    bound: box covering all haze regions

    mask: binary matrix with 1 indicating a pixel of haze region

    number_of_patch: total number of patches

    darkChannelMap_dark,R, G, B: Images of original photo for patch extraction

    result_path: the path to store the extracted patches


    :return


    '''

    haze_patch_size = np.array([patch_size, patch_size])  # smallest patch size we want
    haze_patch_size_radius = haze_patch_size // 2  # radius of filter

    local_number_of_patch = 0

    # ensure at least 10 patches per row/column can be evaluated - a trick to control the step length for patch evaluation
    if haze_patch_size[0]*10 > m:
        row_step = (m - 2 * haze_patch_size_radius[0]) // 10
    else:
        row_step = haze_patch_size[0]

    if haze_patch_size[1]*10 > n:
        column_step = (n - 2 * haze_patch_size_radius[1]) // 10
    else:
        column_step = haze_patch_size[1]



    for i in range(bound[0] + haze_patch_size_radius[0], bound[1] - haze_patch_size_radius[0] + 1, #bound[1] - 2 * haze_patch_size[0] + 1,
                   row_step):  # non-overlap patches
        for j in range(bound[2] + haze_patch_size_radius[1], bound[3] - haze_patch_size_radius[1] + 1, column_step):

            batch = mask[(i - haze_patch_size_radius[0]): (i + haze_patch_size_radius[0]),
                    (j - haze_patch_size_radius[1]): (j + haze_patch_size_radius[1])]
            count = np.count_nonzero(batch == 0)

            if count == 0:  # if the patch does not contain non-haze pixels


                if local_number_of_patch <= max_patch_per_image:
                    number_of_patch += 1

                    local_number_of_patch += 1

                    # store patches in terms of dark channel and RGB
                    patch_dark = darkChannelMap_dark[(i - haze_patch_size_radius[0]): (i + haze_patch_size_radius[0]),
                                 (j - haze_patch_size_radius[1]): (j + haze_patch_size_radius[1])]

                    img = Image.fromarray(patch_dark)
                    img.save(result_path + 'D' + str(local_number_of_patch) + '.png')

                    patch_R = R[
                              (i - haze_patch_size_radius[0]): (i + haze_patch_size_radius[0]),
                              (j - haze_patch_size_radius[1]): (j + haze_patch_size_radius[1])]

                    patch_G = G[(i - haze_patch_size_radius[0]): (i + haze_patch_size_radius[0]),
                              (j - haze_patch_size_radius[1]): (j + haze_patch_size_radius[1])]

                    patch_B = B[(i - haze_patch_size_radius[0]): (i + haze_patch_size_radius[0]),
                              (j - haze_patch_size_radius[1]): (j + haze_patch_size_radius[1])]

                    patch_RGB = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                    patch_RGB[:, :, 0] = patch_R
                    patch_RGB[:, :, 1] = patch_G
                    patch_RGB[:, :, 2] = patch_B

                    img = Image.fromarray(patch_RGB)
                    img.save(result_path + 'O' + str(local_number_of_patch) + '.png')

                    print("Processing patch %d." % (local_number_of_patch))


    return 0