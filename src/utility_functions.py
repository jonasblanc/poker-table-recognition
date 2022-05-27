import numpy as np

BOTTOM_CARD_BOUNDARIES = [0.75,1,0.15,0.9]
LEFT_CARD_BOUNDARIES  = [0.30,0.65,0,0.25]
RIGHT_CARD_BOUNDARIES = [0.30,0.65,0.75,1]
TOP_LEFT_CARD_BOUNDARIES = [0,0.25,0.15,0.45]
TOP_RIGHT_CARD_BOUNDARIES = [0,0.25,0.50,0.9]
CHIPS_BOUNDARIES = [0.25,0.75,0.25,0.75]

def crop(img,fractional_boundaries):
    """All values in boundaries (x_min,x_max,y_min,y_max) are fraction of the corresponding length
    Axis starts at top corner of img with first axis pointing down and second axis pointing right"""
    
    integer_boundaries = [0]*len(fractional_boundaries)
    x_len = img.shape[0]
    y_len = img.shape[1]


    integer_boundaries[0] = int(fractional_boundaries[0] * x_len)
    integer_boundaries[1] = int(fractional_boundaries[1] * x_len)
    integer_boundaries[2] = int(fractional_boundaries[2] * y_len)
    integer_boundaries[3] = int(fractional_boundaries[3] * y_len)

    return img[integer_boundaries[0]:integer_boundaries[1], integer_boundaries[2]: integer_boundaries[3]]

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

def partition_images(imgs):
    """Partition the images into cards/chips sections"""
    partitions = []
    return [partition_image(img) for img in imgs]
    