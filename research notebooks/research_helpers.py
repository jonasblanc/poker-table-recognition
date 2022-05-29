import sys
sys.path.insert(0, '../src/')

import os
import PIL.Image
from constants import *
import numpy as np


def load_train_imgs():
    images = []

    for i in range(28):
        file = os.path.join(TRAIN_DATA, f"train_{str(i).zfill(2)}.jpg")
        im = PIL.Image.open(file)
        images.append(np.array(im))
        
    return images