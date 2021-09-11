# -*- coding: utf-8 -*-

import math
import numpy as np
from PIL import Image



def readImage(root,path):
    '''
    :param

    root: root folder

    path: image name (with possible sub-folders)

    :return

    jpgfile: the image file
    '''

    direc = root + '/' + path

    jpgfile = Image.open(direc)


    return jpgfile