
import numpy as np
import os

def labelProcessor(path): #apply only to the synthetic dataset of 3024 haze images


    # Open file
    file = open(path,'r')
    rawlist = file.read().split('\n')

    length = len(rawlist)-1 #don't know why, the length of rawlist should be 3024 but the result is 3025
    label = np.zeros(length,dtype=np.uint8)


    for i in range(0, length):

        label[i] = int(rawlist[i].split(' ')[1])

    # convert to range [0,8]
    label = (label // 5) - 2


    return label