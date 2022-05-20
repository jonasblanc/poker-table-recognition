import cv2
import numpy as np
import time
from collections import Counter
import matplotlib.pyplot as plt

HSV_COLOR_BOUNDS = {'red':[np.array([136, 87, 111], np.uint8), np.array([180, 255, 255], np.uint8)],
                    'green':[np.array([25, 52, 72], np.uint8), np.array([102, 255, 255], np.uint8)],
                    'blue':[np.array([94, 80, 2], np.uint8), np.array([120, 255, 255], np.uint8)]}

COLOR_TO_SIMBOL = {'red':'R','blue':'B','green':'G','black':'K'}


def count_chips(chips_rgb,plot=False):        
    chips_rgb = cv2.medianBlur(chips_rgb,ksize=51)
    chips_hsv = cv2.cvtColor(chips_rgb, cv2.COLOR_RGB2HSV)

    if(plot):
        plt.imshow(chips_rgb)
        plt.show()
        fig, axes = plt.subplots(4, 3, figsize=(10, 10),tight_layout=True)
        for i in range(3):
            img_grey = chips_hsv[:,:,i]
            ax_index = 0
            ax = axes[ax_index][i]
            ax_index+=1
            ax.imshow(img_grey)
            ax.axis('off')
            ax.set_title(f'Image with {i+1}th HSV component')
        
            ax = axes[ax_index][i]
            ax_index+=1
            colors = img_grey[::8,::8].flatten()
            ax.hist(colors, bins=255)
            ax.set_title(f'Image HSV colour distribution')

            img_grey = chips_rgb[:,:,i]
            ax = axes[ax_index][i]
            ax_index+=1
            ax.imshow(img_grey)
            ax.axis('off')
            ax.set_title(f'Image with {i+1}th RGB component')
        
            ax = axes[ax_index][i]
            colors = img_grey[::8,::8].flatten()
            ax.hist(colors, bins=255)
            ax.set_title(f'Image RGB colour distribution')
        plt.show()

    colors = HSV_COLOR_BOUNDS.keys()
    color_counts = Counter()
    if(plot):
        fig, axes = plt.subplots(4, len(colors),figsize=(10, 10),tight_layout=True)

    for i,color in enumerate(colors):
        low = HSV_COLOR_BOUNDS[color][0]
        high = HSV_COLOR_BOUNDS[color][1]
        thresh = cv2.inRange(chips_hsv,low,high)
        idx=0

        if(plot):
            axes[idx][i].imshow(thresh)
            axes[idx][i].set_title(f"Tresholding for color {color}")
            idx+=1

        # noise removal
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

        # sure background area
        sure_bg = cv2.dilate(opening,kernel,iterations=3)
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
        #Only keeping maximums
        ret, sure_fg = cv2.threshold(dist_transform,0.8*dist_transform.max(),255,0)
        sure_fg = sure_fg.astype(np.uint8)

        if(plot):
            axes[idx][i].imshow(sure_bg)
            axes[idx][i].set_title(f"Sure background")
            idx+=1

            axes[idx][i].imshow(dist_transform)
            axes[idx][i].set_title(f"Distance transform")
            idx+=1

            axes[idx][i].imshow(sure_fg)
            axes[idx][i].set_title(f"Distance transform thresholded")
            idx+=1

        
        
        connectivity=8
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sure_fg , connectivity , cv2.CV_32S)
        num_chips = num_labels-1 #Remove the background
        color_counts[color]=num_chips
        print(f"Detected {num_chips} tokens of color {color}")

    if(plot):
        plt.show()

    output = {}

    for color in colors:
        output[COLOR_TO_SIMBOL[color]]=color_counts[color]
        
    return output

    

def window_treshold(img,hsv_color_bounds,color):

    def nothing(x):
        pass

    def get_window_title(low_hsv,high_hsv):
        window_title = f"""Tresholding for {color} | low_HSV:{low_hsv} | high_HSV:{high_hsv}"""

        return window_title

    low_hsv=hsv_color_bounds[color][0]
    high_hsv=hsv_color_bounds[color][1]

    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Create a window
    window_name='window'
    cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)
    cv2.setWindowTitle(window_name, get_window_title(low_hsv,high_hsv))
    
    # create trackbars for color change
    cv2.createTrackbar('lowH',window_name,low_hsv[0],255,nothing)
    cv2.createTrackbar('highH',window_name,high_hsv[0],255,nothing)
    
    cv2.createTrackbar('lowS',window_name,low_hsv[1],255,nothing)
    cv2.createTrackbar('highS',window_name,high_hsv[1],255,nothing)
    
    cv2.createTrackbar('lowV',window_name,low_hsv[2],255,nothing)
    cv2.createTrackbar('highV',window_name,high_hsv[2],255,nothing)


    while(True):
        
        low_hsv = np.array([cv2.getTrackbarPos('lowH', window_name),cv2.getTrackbarPos('lowS', window_name),cv2.getTrackbarPos('lowV', window_name)])
        high_hsv = np.array([cv2.getTrackbarPos('highH', window_name),cv2.getTrackbarPos('highS', window_name),cv2.getTrackbarPos('highV', window_name)])

        # Apply the cv2.inrange method to create a mask
        thresh = cv2.inRange(img_hsv, low_hsv, high_hsv)

        tresh_RGB = cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB)

        joint_image = np.concatenate((img, tresh_RGB), axis=1)
        joint_image = cv2.cvtColor(joint_image,cv2.COLOR_RGB2BGR)
        cv2.setWindowTitle(window_name, get_window_title(low_hsv,high_hsv))
        cv2.imshow(window_name, joint_image)

        

        # wait for n ms
        k=cv2.waitKey(300)
        print(k)
        if(k==113): 
            cv2.destroyAllWindows()
            for i in range (1,5):
                cv2.waitKey(1)
            break
        

    return low_hsv,high_hsv