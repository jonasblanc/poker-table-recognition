import cv2
from PIL import Image
import numpy as np

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
    approx = rectify(cv2.approxPolyDP(table_cont,0.02*peri,True))

    h = np.array([ [0,0],[table_size,0],[table_size,table_size],[0,table_size] ],np.float32)
    transform = cv2.getPerspectiveTransform(approx, h)
    warp = cv2.warpPerspective(img,transform,(table_size,table_size)) 
    
    return warp