import ipdb
from PIL import Image
from PIL import ImageFilter as IF
import numpy as np


def denoisedImage(contrastMap_image, denoised_filter):

    m, n = contrastMap_image.shape
    patch_size_radius = min(m, n) // (denoised_filter * 2)  # radius of filter

    img = Image.fromarray(contrastMap_image)

    DenoisedImage = img.filter(IF.ModeFilter(patch_size_radius))

    return DenoisedImage