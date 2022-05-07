import cv2
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import numpy as np
import math

#Notes: OTSU tresholding is bad because it assumes the grey-values follow a bimodal distribution whichisnot the case with our images

def plot_HSV_Contour(img, shape_count ,background_idx,card_idx):

    fig, axes = plt.subplots(5, 3, figsize=(20, 10),tight_layout=True)
    img = cv2.GaussianBlur(img,(11,11),100) 
    img_HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    contour_candidates = []
    for i in range(3):
        img_grey = img_HSV[:,:,i]
        axes[0][i].imshow(img_grey)
        axes[0][i].axis('off')
        axes[0][i].set_title(f'Image with {i+1}th HSV component')

        
        colours = img_grey.flatten()
        axes[1][i].hist(colours, bins=255)
        axes[1][i].set_title(f'Image colour distribution')
        
        color_background = img_grey[background_idx[0],background_idx[1]]
        color_card = img_grey[card_idx[0],card_idx[1]]

        print(f"Background color for {i+1}th HSV component: {color_background}")
        print(f"Card colour for {i+1}th HSV component: {color_card}")
        
        manual_threshold = np.mean([color_background,color_card])

        #flag, manual_thresh = cv2.threshold(img_grey, manual_threshold, 255, cv2.THRESH_BINARY)
        flag, manual_thresh = cv2.threshold(img_grey, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        axes[2][i].imshow(manual_thresh)
        axes[2][i].axis('off')
        axes[2][i].set_title(f'Image with manual threshold at {manual_threshold}')


        #cv.RETR_EXTERNAL: only keeps external contours
        #cv.RETR_TREE: hierarchical
        contours, hierarchy = cv2.findContours(manual_thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        if(len(contours)>=shape_count):
            contours = sorted(contours, key=cv2.contourArea,reverse=True)[:shape_count]  
            contour_candidates.append(contours)
        tresh_color = cv2.cvtColor(manual_thresh,cv2.COLOR_GRAY2RGB)*255
        img = cv2.drawContours(tresh_color.copy(), contours, -1,(0,255,0),20)

        axes[3][i].imshow(img)
        axes[3][i].axis('off')
        axes[3][i].set_title(f'Manual thresholding Contours')

        contours, hierarchy = cv2.findContours(~manual_thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        if(len(contours)>=shape_count):
            contours = sorted(contours, key=cv2.contourArea,reverse=True)[:shape_count]  
            contour_candidates.append(contours)

        tresh_color = cv2.cvtColor(manual_thresh,cv2.COLOR_GRAY2RGB)*255
        img = cv2.drawContours(tresh_color.copy(), contours, -1,(0,255,0),20)

        axes[4][i].imshow(img)
        axes[4][i].axis('off')
        axes[4][i].set_title(f'Manual thresholding Inv Contours')
    plt.show()

    #best_contour = pick_best_contours(contour_candidates)
    best_contour = pick_best_contours_overlapping_pair(contour_candidates,img[:,:,0].size)
    tresh_color = cv2.cvtColor(manual_thresh,cv2.COLOR_GRAY2RGB)*255
    img = cv2.drawContours(tresh_color.copy(), best_contour, -1,(0,255,0),20)
    plt.imshow(img)
    plt.title('Best extracted contour')
    plt.show()


def pick_best_contours_overlapping_pair(contour_candidates,img_area):
    
    def compacity(contour):
        peri = cv2.arcLength(contour,False)
        area = cv2.contourArea(contour)
        
        return peri**2/(area*4*math.pi)

    best_candidate = []
    lowest_vertices= np.inf
    for contour in contour_candidates:
        #Only one contour kep in image
        contour = contour[0]
        peri = cv2.arcLength(contour,False)
        #Set a strict precision treshold on polynomial approximation
        approx = cv2.approxPolyDP(contour,0.001*peri,True)
        #print(approx)
        number_points = len(approx)
        normalized_compacity = compacity(contour) #compacity(approx)
        area =  cv2.contourArea(contour)
        print(normalized_compacity)
        if(number_points>5 and number_points<lowest_vertices and normalized_compacity>1.15 and normalized_compacity <2 and area < 0.7 * img_area):
            lowest_vertices = number_points
            best_candidate = contour

    return best_candidate

def pick_best_contours(contour_candidates):
    #CAVEAT: may fail if smaller other regular objects are detected =>could add a minimum contour area
    #Assume only provided the 5 biggest
    min_variance = np.inf
    best_contour = []

    #Iterate over different methods of contours extraction
    for i,contour_candidate in enumerate(contour_candidates):
        contour_areas = []
        for contour in contour_candidate:
            contour_areas.append(cv2.contourArea(contour))
        area_variance = np.var(contour_areas)
        area_mean = np.mean(contour_areas)
        area_standardised_variance = area_variance/(area_mean**2)
        
        if(area_standardised_variance<min_variance):
            print(f"Contour areas for contour {i}:{contour_areas} ")
            print(f"Area variance for contour {i}:{area_standardised_variance} ")
            min_variance = area_standardised_variance
            best_contour = contour_candidate

    return best_contour




    # treshold, otsu_tresh = cv2.threshold(img_grey, 0, 255, cv2.THRESH_OTSU)
    # print(f"OTSU chosen treshold is: {treshold}")

    # axes[3][i].imshow(otsu_tresh)
    # axes[3][i].axis('off')
    # axes[3][i].set_title(f'Image with OTSU thresholding')

    # #cv.RETR_EXTERNAL: only keeps external contours
    # #cv.RETR_TREE: hierarchical
    # #For draw conttours lastarguments are contour color contour thickness
    # contours, hierarchy = cv2.findContours(otsu_tresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    # if(len(contours)>=EXPECTED_NUMBER_CONTOURS):
    #     contours = sorted(contours, key=cv2.contourArea,reverse=True)[:EXPECTED_NUMBER_CONTOURS]  
    #     contour_candidates.append(contours)
    
    # tresh_color = cv2.cvtColor(otsu_tresh,cv2.COLOR_GRAY2RGB)*255
    # img = cv2.drawContours(tresh_color.copy(), contours, -1,(0,255,0),20)

    # axes[4][i].imshow(img)
    # axes[4][i].axis('off')
    # axes[4][i].set_title(f'OTSU thresholding Contours')

    # #Opening
    # kernel_size = (50,50)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
# opened = cv2.morphologyEx(otsu_tresh, cv2.MORPH_OPEN, kernel)
    # axes[4][i].imshow(opened,cmap='gray')
    # axes[4][i].axis('off')
    # axes[4][i].set_title(f'Image after opening (OTSU)')
    
    # #Closing
    # kernel_size = (40,40)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    # closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    # axes[5][i].imshow(closed,cmap='gray')
    # axes[5][i].axis('off')
    # axes[5][i].set_title(f'Image after closing (OTSU)')