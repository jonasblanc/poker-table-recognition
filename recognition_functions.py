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
    blur = cv2.GaussianBlur(gray,(11,11),1000) 
    flag, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)
    
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel= np.ones((51,51), np.uint8))

    contours, hierarchy = cv2.findContours(opened,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea,reverse=True) 
    
    table_cont = contours[0]
    
    #rot_rect = cv2.minAreaRect(contours[0])
    #box = cv2.boxPoints(rot_rect)
    peri = cv2.arcLength(table_cont,True)
    approx = np.array((cv2.approxPolyDP(table_cont,0.02*peri,True)))[:,0]
    diff = np.abs(approx[:,0]-approx[:,1])
    approx = rectify(approx[np.argsort(diff)[::-1][:4]])

    h = np.array([ [0,0],[table_size,0],[table_size,table_size],[0,table_size] ],np.float32)
    transform = cv2.getPerspectiveTransform(approx, h)
    warp = cv2.warpPerspective(img,transform,(table_size,table_size)) 
    
    return warp