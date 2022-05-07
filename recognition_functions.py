import cv2
from PIL import Image
import numpy as np

BOTTOM_CARD_BOUNDARIES = [0.75,1,0.15,0.9]
LEFT_CARD_BOUNDARIES  = [0.30,0.65,0,0.25]
RIGHT_CARD_BOUNDARIES = [0.30,0.65,0.75,1]
TOP_LEFT_CARD_BOUNDARIES = [0,0.25,0.15,0.45]
TOP_RIGHT_CARD_BOUNDARIES = [0,0.25,0.70,0.9]
CHIPS_BOUNDARIES = [0.25,0.75,0.25,0.75]

def optimal_Gaussian_mixture_tresholding(grey_img,n_centroids, background_is_darker, set_other_centroids_to_background=False):
    """Does optimal thresholding by assuming the grey_img follows a n_centroid Gaussian Mixture models and choses
    to optimally treshold by only considering the two Gaussian clusters with the most points"""
    #Idea: use Kmeans to classify all pixels or Gaussian mixture
    # The top 2 clusters in number of points are our interesting clusters
    # Set other clusters to background or not
    #Idea: use OTSU after having assigned removed 
    #For cards: assign all pixels that are not background to the card label
    pass

def optimal_contour_extraction(img,number_shapes,background_idx,target_shape_idx):
    """Extracts number_shapes from the image assuming that all shapes are similar
    Works well for extracting the contour of multiple similar elements from the image (eg: cards) 
    Assumes provided image is in RGB"""
    #NOTE: use provided pixel indices to extract colour for correct manual thresholding => could be latter replaced 
    # with K-means to only keep 2 colors and then use thresholding
    #NOTE: in the case of wanted rectangular shapes, we could have tried to apporximate the contour with cv2.approxPolyDP and 
    #verify that we obtain 4 points

    def filter_contour(thresh,contour_candidates):
        """If contour_candidate has enough threshold, adds it to candidates list"""
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        if(len(contours)>=number_shapes):
            contours = sorted(contours, key=cv2.contourArea,reverse=True)[:number_shapes]  
            contour_candidates.append(contours)

    def pick_best_contours(contour_candidates):
        #CAVEAT: may fail if smaller other regular objects are detected =>could add a minimum contour area
        #Assume only provided the number_shapes biggest
        min_variance = np.inf
        best_contour = []

        #Iterate over different methods of contours extraction
        for contour_candidate in contour_candidates:
            contour_areas = []
            for contour in contour_candidate:
                contour_areas.append(cv2.contourArea(contour))
            area_variance = np.var(contour_areas)
            area_mean = np.mean(contour_areas)
            area_standardised_variance = area_variance/(area_mean**2)
            
            if(area_standardised_variance<min_variance):
                min_variance = area_standardised_variance
                best_contour = contour_candidate

        return best_contour
    
    img = cv2.GaussianBlur(img,(11,11),100) 
    img_HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    contour_candidates = []
    for i in range(3):
        img_grey = img_HSV[:,:,i]
        color_background = img_grey[background_idx[0],background_idx[1]]
        color_card = img_grey[target_shape_idx[0],target_shape_idx[1]]
        manual_threshold = np.mean([color_background,color_card])

        flag, manual_thresh = cv2.threshold(img_grey, manual_threshold, 255, cv2.THRESH_BINARY)

        filter_contour(manual_thresh,contour_candidates)
        #Do contouring on image inverse
        filter_contour(~manual_thresh,contour_candidates)

    best_contour = pick_best_contours(contour_candidates)

    return best_contour
        
    


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
