import numpy as np

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

    