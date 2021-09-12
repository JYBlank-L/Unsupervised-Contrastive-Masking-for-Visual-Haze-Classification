import torch
import torch.nn as nn
import numpy as np

def brightChannelProcess(image, patch_size_radius):
    '''
    :param

    image: gray image - can be dark or bright image

    patch_size_radius: the radius of patch, equal to kernel_size()/2

    :return

    brightChannelMap: The bright channel map for haze estimation
    '''

    m = nn.MaxPool2d(kernel_size=patch_size_radius*2, stride=1, padding=patch_size_radius)
    input = torch.from_numpy(image.astype(float)).unsqueeze(0)

    brightChannelMap = m(input).squeeze().numpy()[:, :-1]

    return brightChannelMap

def darkChannelProcess(image, patch_size_radius):
    '''
    :param

    image: gray image - can be dark or bright image

    patch_size_radius: the radius of patch, equal to kernel_size()/2

    :return

    darkChannelMap: The dark channel map for haze estimation
    '''

    m = nn.MaxPool2d(kernel_size=patch_size_radius*2, stride=1, padding=patch_size_radius)
    image = image*(-1)
    input = torch.from_numpy(image.astype(float)).unsqueeze(0)

    darkChannelMap = m(input).squeeze().numpy()[:, :-1]*(-1)

    return darkChannelMap