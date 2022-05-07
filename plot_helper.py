import cv2
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import numpy as np

#TODO: sometimes, have to invert the manual threholding to find the contours

def plot_HSV_Contour(img):
    fig, axes = plt.subplots(9, 3, figsize=(30, 18),tight_layout=True)
    img = cv2.GaussianBlur(img,(11,11),100) 
    img_HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    for i in range(3):
        img_grey = img_HSV[:,:,i]
        axes[0][i].imshow(img_grey)
        axes[0][i].axis('off')
        axes[0][i].set_title(f'Image with {i+1}th HSV component')

        print(f"Background color for {i+1}th HSV component: {img_grey[40,40]}")
        print(f"Card colour for {i+1}th HSV component: {img_grey[550,200]}")

        colours = img_grey.flatten()
        axes[1][i].hist(colours, bins=255)
        axes[1][i].set_title(f'Image colour distribution')
        
        color_background = img_grey[40,40]
        color_card = img_grey[550,200]
        manual_threshold = np.mean([color_background,color_card])

        flag, manual_thresh = cv2.threshold(img_grey, manual_threshold, 255, cv2.THRESH_BINARY)

        axes[2][i].imshow(manual_thresh)
        axes[2][i].axis('off')
        axes[2][i].set_title(f'Image with manual threshold at {manual_threshold}')

        treshold, otsu_tresh = cv2.threshold(img_grey, 0, 255, cv2.THRESH_OTSU)
        print(f"OTSU chosen treshold is: {treshold}")

        axes[3][i].imshow(otsu_tresh)
        axes[3][i].axis('off')
        axes[3][i].set_title(f'Image with OTSU thresholding')

        #Opening
        kernel_size = (50,50)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        opened = cv2.morphologyEx(otsu_tresh, cv2.MORPH_OPEN, kernel)
        axes[4][i].imshow(opened,cmap='gray')
        axes[4][i].axis('off')
        axes[4][i].set_title(f'Image after opening (OTSU)')
        
        #Closing
        kernel_size = (40,40)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

        axes[5][i].imshow(closed,cmap='gray')
        axes[5][i].axis('off')
        axes[5][i].set_title(f'Image after closing (OTSU)')

        #cv.RETR_EXTERNAL: only keeps external contours
        #cv.RETR_TREE: hierarchical
        contours, hierarchy = cv2.findContours(otsu_tresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        #lastarguments are contour color
        #Contour thickness
        tresh_color = cv2.cvtColor(otsu_tresh,cv2.COLOR_GRAY2RGB)*255
        img = cv2.drawContours(tresh_color.copy(), contours, -1,(0,255,0),20)

        axes[6][i].imshow(img)
        axes[6][i].axis('off')
        axes[6][i].set_title(f'OTSU thresholding Contours')

        #cv.RETR_EXTERNAL: only keeps external contours
        #cv.RETR_TREE: hierarchical
        contours, hierarchy = cv2.findContours(manual_thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        #lastarguments are contour color
        #Contour thickness
        tresh_color = cv2.cvtColor(manual_thresh,cv2.COLOR_GRAY2RGB)*255
        img = cv2.drawContours(tresh_color.copy(), contours, -1,(0,255,0),20)

        axes[7][i].imshow(img)
        axes[7][i].axis('off')
        axes[7][i].set_title(f'Manual thresholding Contours')

        contours, hierarchy = cv2.findContours(~manual_thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        #lastarguments are contour color
        #Contour thickness

        tresh_color = cv2.cvtColor(manual_thresh,cv2.COLOR_GRAY2RGB)*255
        img = cv2.drawContours(tresh_color.copy(), contours, -1,(0,255,0),20)

        axes[8][i].imshow(img)
        axes[8][i].axis('off')
        axes[8][i].set_title(f'Manual thresholding Inv Contours')

    plt.show()