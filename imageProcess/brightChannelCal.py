# -*- coding: utf-8 -*-


import numpy as np




def brightChannelCal(image, path):
    '''
    :param

    image: gray image - can be dark or bright image

    path: Save dark image under the path

    :return

    brightChannelMap: The bright channel map for haze estimation
    '''


# ---------calculate darkChannelMap according to dark channel prior------------
    m, n = image.shape
    patch_size_radius = min(m, n) // (40 * 2)  # make filter size to be 1/40 of the shorter length of the image


    brightChannelMap = np.zeros((m,n),dtype=np.uint8) #a trick that PIL.fromarray handles uint8 format only

    #for i in range(0, m):
     #   for j in range(0, n):
    for i,j in np.ndindex(m,n):
            patch = image[max(i-patch_size_radius,0):min(i+patch_size_radius+1,m+1),max(j-patch_size_radius,0):min(j+patch_size_radius+1,n+1)]

            brightChannelMap[i,j] = np.max(patch)#np.max(patch) or np.min(patch) for bright or dark channel map

    #img = Image.fromarray(brightChannelMap)
    #img.save(path + '/' + 'brightChannelMap.png')

    return brightChannelMap