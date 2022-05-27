import sys
sys.path.insert(0, '../src/')

import os
import PIL.Image
from constants import *
import numpy as np

BOTTOM_CARD_BOUNDARIES = [0.75,1,0.15,0.9]
LEFT_CARD_BOUNDARIES  = [0.30,0.65,0,0.25]
RIGHT_CARD_BOUNDARIES = [0.30,0.65,0.75,1]
TOP_LEFT_CARD_BOUNDARIES = [0,0.25,0.15,0.45]
TOP_RIGHT_CARD_BOUNDARIES = [0,0.25,0.50,0.9]
CHIPS_BOUNDARIES = [0.25,0.75,0.25,0.75]

def load_train_imgs():
    images = []

    for i in range(28):
        file = os.path.join(TRAIN_DATA, f"train_{str(i).zfill(2)}.jpg")
        im = PIL.Image.open(file)
        images.append(np.array(im))
        
    return images
        

def partition_image(img):
    """Partition the image into cards/chips sections"""
    result = {}

    result['bottom_row'] = crop(img,BOTTOM_CARD_BOUNDARIES)
    result['left_cards'] = crop(img,LEFT_CARD_BOUNDARIES)
    result['right_cards'] =  crop(img,RIGHT_CARD_BOUNDARIES)
    result['top_left_cards'] = crop(img,TOP_LEFT_CARD_BOUNDARIES)
    result['top_right_cards'] = crop(img,TOP_RIGHT_CARD_BOUNDARIES)
    result['chips'] = crop(img,CHIPS_BOUNDARIES)

    return result