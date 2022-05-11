import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import math

#Notes: OTSU tresholding is bad because it assumes the grey-values follow a bimodal distribution whichisnot the case with our images

def plot_contour(ax,contours,background,title):
    img = cv2.drawContours(background.copy(), contours, -1,(0,255,0),20)

    ax.imshow(img)
    ax.axis('off')
    ax.set_title(title)


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
