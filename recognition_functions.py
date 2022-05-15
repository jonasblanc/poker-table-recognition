import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from plot_helper import plot_contours
from sklearn.cluster import KMeans
import math 

BOTTOM_CARD_BOUNDARIES = [0.75,1,0.15,0.9]
LEFT_CARD_BOUNDARIES  = [0.30,0.65,0,0.25]
RIGHT_CARD_BOUNDARIES = [0.30,0.65,0.75,1]
TOP_LEFT_CARD_BOUNDARIES = [0,0.25,0.15,0.45]
TOP_RIGHT_CARD_BOUNDARIES = [0,0.25,0.50,0.9]
CHIPS_BOUNDARIES = [0.25,0.75,0.25,0.75]

#Table extraction constants
# TABLE_MIN_AREA = 0.5
# TABLE_MAX_AREA = 0.6
#TABLE_CONTOUR_APPROX_MARGIN = 0.001

#Pair card contour extraction constants
PAIR_CARD_MIN_AREA = 0.2
PAIR_CARD_MAX_AREA = 0.4
PAIR_CARD_CONTOUR_APPROX_MARGIN = 0.001

#Bottom card contour extraction constants
BOTTOM_CARD_MIN_AREA = 0.05
BOTTOM_CARD_MAX_AREA = 0.15
BOTTOM_CARD_APPROX_MARGIN = 0.02



########################################CONTOUR EXTRACTION###############################################

def extract_candidate_contours(img, shape_count,n_thresholds = 2, opening_kernel_size = None, plot = False):

    number_plot_per_HSV = 2+4*n_thresholds  # (image_HSV, hist, thresholded image, opened_img, contour, countour_inv)
    if(plot):
        fig, axes = plt.subplots(number_plot_per_HSV, 3, figsize=(20, 20),tight_layout=True)

    img = cv2.GaussianBlur(img,(11,11),100) 
    img_HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    contour_candidates = []

    for i in range(3):
        img_grey = img_HSV[:,:,i]

        #Index by stride of 16 to reduce by 256 the number of pixels K-means has to process
        colors = img_grey[::16,::16].flatten()
        kmeans = KMeans(n_clusters=n_thresholds+1, random_state=0).fit(colors.reshape(-1,1))
        centers = sorted(kmeans.cluster_centers_.flatten())
        thresholds = np.array(list(zip(centers,centers[1:]))).mean(axis=1)

        if(plot):
            ax_index = 0
            ax = axes[ax_index][i]
            ax.imshow(img_grey)
            ax.axis('off')
            ax.set_title(f'Image with {i+1}th HSV component')
            ax_index+=1

            ax = axes[ax_index][i]
            ax.hist(colors, bins=255)
            ax.set_title(f'Image colour distribution')

            for k,center in enumerate(centers):
                if(k==0):
                    ax.axvline(x=center, color='black', linestyle='--',label='Cluster centers')
                else:
                    ax.axvline(x=center, color='black', linestyle='--')

            for k,threshold in enumerate(thresholds):
                if(k==0):
                    ax.axvline(x=threshold, color='green', linestyle='-',label='Thresholds')
                else:
                    ax.axvline(x=threshold, color='green', linestyle='-')
            ax.legend()
            ax_index+=1
       
        for k, threshold in enumerate(thresholds):
            flag, thresh_img = cv2.threshold(img_grey, threshold, 255, cv2.THRESH_BINARY)
            opened_img = thresh_img
            opened_img_inv = ~thresh_img
            if(opening_kernel_size!=None):
                kernel_size = (opening_kernel_size,opening_kernel_size)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
                opened_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN,kernel)
                opened_img_inv = cv2.morphologyEx(~thresh_img, cv2.MORPH_OPEN,kernel)


            contours = extract_biggest_contours(opened_img, shape_count)
            #Invert the image
            contours_inv = extract_biggest_contours(opened_img_inv, shape_count)

            if(len(contours)>0):
                contour_candidates.append(contours)
            if(len(contours_inv)>0):
                contour_candidates.append(contours_inv)

            if(plot):
                ax = axes[ax_index][i]
                ax.imshow(thresh_img)
                ax.axis('off')
                ax.set_title(f'Image with threshold at {threshold:.2f}')
                ax_index+=1

                ax = axes[ax_index][i]
                ax.imshow(opened_img)
                ax.axis('off')
                ax.set_title(f'Image after opening at {threshold:.2f}')
                ax_index+=1
            
                ax = axes[ax_index][i]
                plot_contours(contours, np.zeros_like(img), "Thresholding contours",ax)
                ax_index+=1

                ax = axes[ax_index][i]
                plot_contours(contours_inv, np.zeros_like(img), "Inverted Thresholding contours",ax)
                ax_index+=1

    if(plot):
        plt.show()

    return contour_candidates

def extract_biggest_contours(bin_img,shape_count):
        contours, hierarchy = cv2.findContours(bin_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        if(len(contours)>=shape_count):
            contours = sorted(contours, key=cv2.contourArea,reverse=True)[:shape_count]  
        return contours

##########################################TABLE EXTRACTION#######################################################



#######################################################################################################################

##########################################PAIR_CARD EXTRACTION#######################################################

def choose_best_pair_contour(candidate_contours,img_area):
    """Assumes we only get 1 shape per candidate contour"""

    best_candidate = []
    min_vertices_count = np.inf
    for contour in candidate_contours:
        #Only one contour kept in image
        contour = contour[0]
        peri = cv2.arcLength(contour,False)
        #Set a strict precision treshold on polynomial approximation
        approx = cv2.approxPolyDP(contour,PAIR_CARD_CONTOUR_APPROX_MARGIN*peri,True)
        vertices_count = len(approx)
        area = cv2.contourArea(contour)
        if(vertices_count>5 and vertices_count<min_vertices_count and area>PAIR_CARD_MIN_AREA * img_area and area < PAIR_CARD_MAX_AREA * img_area):
            min_vertices_count = vertices_count
            best_candidate = contour

    return best_candidate

#######################################################################################################################

#########################################BOTTOM_CARD_EXTRACTION########################################################

def choose_best_bottom_contour(candidate_contours,img_area):
    """contour_candidates is list[[list[contour]]] and we have to chose the best list  """
    #Assume only provided the biggest n shapes
    min_variance = np.inf
    best_contour = []

    #Iterate over different methods of contours extraction
    for candidate_contour in candidate_contours:
        contour_areas = []
        for contour in candidate_contour:
            contour_areas.append(cv2.contourArea(contour))
        area_variance = np.var(contour_areas)
        area_mean = np.mean(contour_areas)
       
        area_standardised_variance = area_variance/(area_mean**2)
        
        if(len(candidate_contour)==5 and area_standardised_variance<min_variance and area_mean>BOTTOM_CARD_MIN_AREA*img_area and area_mean<BOTTOM_CARD_MAX_AREA*img_area):
            min_variance = area_standardised_variance
            best_contour = candidate_contour

    return best_contour

def reorder_corners(corners):
    """ Reorder the corners so they have a deterministic order: top_left,top_right,bottom_right,bottom_left] 
    corners: numpy array of (4,2)
    """
    #NOTE: in the corners opencv representation, the first axis goes left 
    # and the second goes down (inverse of standardimamge representation)

    new_corners = np.empty((4,2),np.float32)

    mean_x = corners[:,0].mean()
    mean_y = corners[:,1].mean()

    for x,y in corners:
        corner = np.array([x,y])
        if(x<mean_x and y<mean_y):
            new_corners[0][0] = x
            new_corners[0][1] = y
        elif(x>mean_x and y<mean_y):
            new_corners[1][0] = x
            new_corners[1][1] = y
        elif(x>mean_x and y>mean_y):
            new_corners[2][0] = x
            new_corners[2][1] = y
        else:
            new_corners[3][0] = x
            new_corners[3][1] = y

    return new_corners

def leftmost_coordinate(corners):
    #NOTE: uses open cv corner representation (first axis points left)
    return np.min(corners[:,0])

def extract_bottom_cards(bottom_row,plot=False):

    img_area = bottom_row.shape[0]*bottom_row.shape[1]
    candidate_contours = extract_candidate_contours(bottom_row,shape_count=5)
    best_contours = choose_best_bottom_contour(candidate_contours,img_area)

    if(plot):

        fig, axes = plt.subplots(1, 2, figsize=(10, 10),tight_layout=True)
        axes[0].imshow(bottom_row)
        axes[0].set_title(f'Original image')
        background = np.zeros_like(bottom_row)
        contour_img = cv2.drawContours(background.copy(), best_contours, -1,(0,255,0),20)
        axes[1].imshow(contour_img)
        axes[1].set_title(f'Best contours')
        plt.show()

    card_size = [300,400]
    contour_corners = []
    for card_contour in best_contours:
        peri = cv2.arcLength(card_contour,True)
        approx = cv2.approxPolyDP(card_contour,BOTTOM_CARD_APPROX_MARGIN*peri,True)
        corners = np.array(approx[:,0,:],np.float32) #Just remove useless middle dimension
        if(len(corners)!=4):
            print(f"WARNING: Card approximation has more than 4 corners")
            corners = corners[:4]

        corners = reorder_corners(corners)
        contour_corners.append(corners)

    #Sort contours from leftmost to right_most
    contour_corners = sorted(contour_corners,key = lambda corners: leftmost_coordinate(corners))
    extracted_cards = []

    for corners in contour_corners:

        h = np.array([[0,0],[card_size[0],0],[card_size[0],card_size[1]],[0,card_size[1]] ],np.float32)
        
        transform = cv2.getPerspectiveTransform(corners, h)
        card = cv2.warpPerspective(bottom_row,transform,card_size)
        extracted_cards.append(card)

    if(plot and len(extracted_cards)>0):
        fig, axes = plt.subplots(1, len(extracted_cards), figsize=(10, 15),tight_layout=True)
        for i,card in enumerate(extracted_cards):
            axes[i].imshow(card)
            axes[i].set_title(f"Card number {i}")
        plt.show()

    return extracted_cards
  
################################################################################################# 

###########################################PARTITION IMAGE###################################################### 

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

##########################################LEGACY#################################################

def _rectangularity(contour,rect):
    #Allows rotated rectangle => property, scale, rotation and translation invariant
    contour_area = cv2.contourArea(contour)
    (x,y),(w,h),rot_angle = rect
    rect_area = w*h
    rectangularity = contour_area/rect_area
    return rectangularity

#In previous processing, do massive opening => provide opening shape to  the extract candidate contour function => if none: ignore
#Do a simple 4 corner approximation with good enough perimeter margin
#IDEA: Take the contour with less corners => plot and to warp, find the smallest enclosing/inclosing rectangle
def _choose_best_table_contour(candidate_contours,img_area,plot=False):
    best_candidate = None
    best_box = None
    best_approx = None
    best_rect_area_fit = np.inf #Should be closest to 0
    for candidate in candidate_contours:
        
        background = np.zeros((4000,6000,3))

        contour = candidate[0] #Assume only took the biggest contour
        peri = cv2.arcLength(contour,True)
        approx = cv2.approxPolyDP(contour,0.005*peri,True)
        approx_area = cv2.contourArea(approx)


        # print(f"Approximation area {approx_area}")
        
        contour_area =  cv2.contourArea(contour)

        #Convert grey-scale contour to color
        rect = cv2.minAreaRect(contour)
        box = np.int0(cv2.boxPoints(rect))
        rectangularity = compute_rectangularity(contour, rect)
        rect_area_fit = abs(rectangularity-1)

        # print(f"Rectangularity: {rectangularity}")
        # print(f"Contour proportional area: {contour_area/img_area}")

        proportional_area = contour_area/img_area
        #Optimal rectangularity is 1
        if(proportional_area>TABLE_MIN_AREA and proportional_area<TABLE_MAX_AREA and rect_area_fit<best_rect_area_fit):
            best_candidate = contour
            best_box = box
            best_approx = approx
            best_rect_area_fit = rect_area_fit


        # plot_contours(contour, background, "Contour")
        # plt.show()

        # plot_contours([approx], background, "Approximate Contour")
        # plt.show()

        # plot_contours([box], background, "Min enclosing box")
        # plt.show()
        
        

    
    if(plot):
        plot_contours(best_candidate, background, "Contour")
        plt.show()

        plot_contours([approx], background, "Approximate Contour")
        plt.show()

        plot_contours([best_box], background, "Enclosing rectangle")
        plt.show()


    return best_box





    return best_contour


def _compacity(contour):
        peri = cv2.arcLength(contour,False)
        area = cv2.contourArea(contour)
        
        return peri**2/(area*4*math.pi)