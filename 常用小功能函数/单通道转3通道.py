
import numpy as np
from PIL import Image
import os


def save3channels(im_np, im_save_path):
    image = np.expand_dims(im_np, axis=2)
    image = np.concatenate((image, image, image), axis=-1)
    Image.fromarray(image).save(im_save_path)


