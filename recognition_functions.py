import cv2
from PIL import Image
import numpy as np

BOTTOM_CARD_BOUNDARIES = [0.75,1,0.15,0.9]
LEFT_CARD_BOUNDARIES  = [0.30,0.65,0,0.25]
RIGHT_CARD_BOUNDARIES = [0.30,0.65,0.75,1]
TOP_LEFT_CARD_BOUNDARIES = [0,0.25,0.15,0.45]
TOP_RIGHT_CARD_BOUNDARIES = [0,0.25,0.70,0.9]
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

    result['bottom_cards'] = crop(img,BOTTOM_CARD_BOUNDARIES)
    result['left_cards'] = crop(img,LEFT_CARD_BOUNDARIES)
    result['right_cards'] =  crop(img,RIGHT_CARD_BOUNDARIES)
    result['top_left_cards'] = crop(img,TOP_LEFT_CARD_BOUNDARIES)
    result['top_right_cards'] = crop(img,TOP_RIGHT_CARD_BOUNDARIES)
    result['chips'] = crop(img,CHIPS_BOUNDARIES)

    return result


def rectify(h):
    h = h.reshape((4,2))
    hnew = np.zeros((4,2),dtype = np.float32)

    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]

    diff = np.diff(h,axis = 1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew

def table_extraction(img, table_size=3800):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(1,1),1000) 
    flag, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)
    
    closed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel= np.ones((11,11), np.uint8))

    contours, hierarchy = cv2.findContours(closed,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea,reverse=True) 
    
    table_cont = contours[0]
    peri = cv2.arcLength(table_cont,True)
    approx = rectify(cv2.approxPolyDP(table_cont,0.02*peri,True)[:4])

    h = np.array([ [0,0],[table_size,0],[table_size,table_size],[0,table_size] ],np.float32)
    transform = cv2.getPerspectiveTransform(approx, h)
    warp = cv2.warpPerspective(img,transform,(table_size,table_size)) 
    
    return warp