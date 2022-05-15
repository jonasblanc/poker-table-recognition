import cv2
import matplotlib.pyplot as plt

import numpy as np
import math

#Notes: OTSU tresholding is bad because it assumes the grey-values follow a bimodal distribution whichisnot the case with our images

def plot_contours(contours,background,title,ax=None):
    img = cv2.drawContours(background.copy(), contours, -1,(0,255,0),20)

    if(ax==None):
        fig,ax = plt.subplots(1,1,tight_layout=True)

    ax.imshow(img)
    ax.axis('off')
    ax.set_title(title)

