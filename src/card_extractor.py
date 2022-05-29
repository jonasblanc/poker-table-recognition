import cv2
import math 
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

from utility_functions import *
from contour_helper import ContourHelper

class CardExtractor:
    """
    Module to extract table and player cards from an image
    """
    # Assumed pixel size of the cards in input image
    CARD_SIZE = [400, 582]
    
    # Accepted card delta for pair extraction
    ACCEPTED_HEIGHT_DELTA = 0.08
    ACCEPTED_WIDTH_DELTA = 0.08
    
    # Extend card for pair extraction
    HEIGHT_CARD_MARGIN_PIX = 5
    WIDTH_CARD_MARGIN_PIX = 5

    TABLE_CARDS = "T"
    P_1_CARDS = "P1"
    P_2_CARDS = "P2"
    P_3_CARDS = "P3"
    P_4_CARDS = "P4"

    name_to_cropped_imgs = None
    
    TABLE_CARD_BOUNDARIES = [0.75,1,0.15,0.9]
    PLAYER_1_CARDS_BOUNDARIES = [0.30,0.65,0.70,1]
    PLAYER_2_CARDS_BOUNDARIES = [0,0.25,0.50,0.9]
    PLAYER_3_CARDS_BOUNDARIES = [0,0.25,0.15,0.45]
    PLAYER_4_CARDS_BOUNDARIES  = [0.30,0.65,0,0.25]

    #Pair card contour extraction constants
    PAIR_CARD_MIN_AREA = 0.2
    PAIR_CARD_MAX_AREA = 0.4
    PAIR_CARD_CONTOUR_APPROX_MARGIN = 0.001

    #Table card contour extraction constants
    TABLE_CARD_MIN_AREA = 0.05
    TABLE_CARD_MAX_AREA = 0.15
    TABLE_CARD_APPROX_MARGIN = 0.02
    
    
    def __init__(self, table_img):
        self.name_to_cropped_imgs = self._partition_image(table_img)
        
    def extract_cards_area(self, position):
        """
        Crop the table image to return a image of the area of table/player's cards 
        position: ['T', 'P1', 'P2', 'P3', 'P4']
        """
        return self.name_to_cropped_imgs[position]
        
    def extract_player_cards(self, player, plot=False):
        """
        Crop the table image to return two images of the cards of a player
        player: ['P1', 'P2', 'P3', 'P4']
        """
        if (not self.name_to_cropped_imgs):
            raise ValueError
            
        corners_approx_margin_factors = [0.005, 0.01, 0.05]
        bottom_card, top_card = (None, None)
        
        player_cards_img = self.name_to_cropped_imgs[player]
        
        # Try out multiple approximation margin to maximise the possibility of card extraction
        for f in corners_approx_margin_factors:
            pot_bottom_card, pot_top_card = self._extract_pair_cards(player_cards_img, corners_approx_margin=f,plot=plot)
            if type(top_card) == type(None):
                top_card = pot_top_card
            if type(bottom_card) == type(None):
                bottom_card = pot_bottom_card
            if (type(top_card) != type(None) and type(bottom_card) != type(None)):
                break
                
        return bottom_card, top_card
          
    def extract_table_cards(self, plot=False):
        """
        Crop the table image to return five images of the table cards of a player
        player: ['P1', 'P2', 'P3', 'P4']
        """
        if (not self.name_to_cropped_imgs):
            raise ValueError
        
        table_cards_img = self.name_to_cropped_imgs[self.TABLE_CARDS]

        img_area = table_cards_img.shape[0] * table_cards_img.shape[1]
        candidate_contours = ContourHelper.extract_candidate_contours(table_cards_img, number_contour=5, plot=False)
        best_contours = self._choose_best_table_cards_contour(candidate_contours, img_area)
        extracted_cards = self.extract_cards_from_contours(table_cards_img, best_contours, self.TABLE_CARD_APPROX_MARGIN)

        if(plot):
            fig, axes = plt.subplots(1, 2, figsize=(10, 10), tight_layout=True)
            axes[0].imshow(table_cards_img)
            axes[0].set_title(f'Original image')
            background = np.zeros_like(table_cards_img)
            contour_img = cv2.drawContours(background.copy(), best_contours, -1,(0,255,0), 20)
            axes[1].imshow(contour_img)
            axes[1].set_title(f'Best contours')
            plt.show()
            
            if(len(extracted_cards) > 0):
                fig, axes = plt.subplots(1, len(extracted_cards), figsize=(10, 15),tight_layout=True)
                if len(extracted_cards) == 1:
                    axes.imshow(extracted_cards[0])
                    axes.set_title(f"Card number {0}")
                else:
                    for i,card in enumerate(extracted_cards):
                        axes[i].imshow(card)
                        axes[i].set_title(f"Card number {i}")
                plt.show()            

        return extracted_cards
    
    
    @classmethod
    def extract_cards_from_contours(cls, img, contours, approx_margin):
        """
        Crop image based on a list of contours
        approx_margin: the factor of deviation from perimeter allowed 
        """
        contour_corners = []
        for card_contour in contours:
            peri = cv2.arcLength(card_contour, True)
            approx = cv2.approxPolyDP(card_contour, approx_margin * peri,True)
            corners = np.array(approx[:, 0, :], np.float32) #Just remove useless middle dimension
            if(len(corners)>4):
                print(f"WARNING: Card approximation has more than 4 corners")
                corners = corners[:4]
            elif (len(corners)<4):
                print(f"WARNING: Card approximation has less than 4 corners")
            else:
                corners = ContourHelper.reorder_corners(corners)
                contour_corners.append(corners)

        #Sort contours from leftmost to right_most
        contour_corners = sorted(contour_corners,key = lambda corners: cls._leftmost_coordinate(corners))
        extracted_cards = []

        for corners in contour_corners:

            h = np.array([[0,0],[cls.CARD_SIZE[0],0],[cls.CARD_SIZE[0],cls.CARD_SIZE[1]],[0,cls.CARD_SIZE[1]] ],np.float32)

            transform = cv2.getPerspectiveTransform(corners, h)
            card = cv2.warpPerspective(img, transform, cls.CARD_SIZE)
            extracted_cards.append(card)
            
        return extracted_cards
        
    
    @classmethod
    def _extract_pair_cards(cls, cards_img, corners_approx_margin=0.005, plot=False):
        """
        Return first card on the bottom then card on the top
        """

        # Extract contour
        candidate_contours = ContourHelper.extract_candidate_contours(cards_img, number_contour = 1,n_thresholds=2,plot=False)
        img_area = cards_img.shape[0] * cards_img.shape[1]
        best_contour = cls._choose_best_pair_contour(candidate_contours, img_area)
        
        if(len(best_contour) == 0):
            print("Warning: No contour detected for player card extraction")
            return None, None

        peri = cv2.arcLength(best_contour,False)
        approx_vertices = cv2.approxPolyDP(best_contour,corners_approx_margin * peri,True)[:,0]

        diff = []

        # Distance between two consecutive points
        for i in range(approx_vertices.shape[0]):
            diff.append(np.linalg.norm(approx_vertices[i]-approx_vertices[i-1]))
        diff = np.array(diff)

        distance_to_height = np.abs(diff - cls.CARD_SIZE[1]) / cls.CARD_SIZE[1]
        distance_to_width = np.abs(diff - cls.CARD_SIZE[0]) / cls.CARD_SIZE[0]

        min_height_dist_idxes = np.argsort(distance_to_height)
        min_width_dist_idxes = np.argsort(distance_to_width)


        top_card = None
        bottom_card = None
        top_card_corners = []
        bottom_card_corners = []
        
        is_height_done = False
        is_width_done = False
        
        h_idx = 0
        w_idx = 0
        
        while(type(top_card) == type(None) or type(bottom_card) == type(None)):
            
            # Height corners are done if no more corner are above the accepted delta
            is_height_done |= h_idx >= len(min_height_dist_idxes) or distance_to_height[min_height_dist_idxes[h_idx]] > cls.ACCEPTED_HEIGHT_DELTA
        
            # Width corners are done if no more corner are above the accepted delta
            is_width_done |= w_idx >= len(min_width_dist_idxes) or distance_to_width[min_width_dist_idxes[w_idx]] > cls.ACCEPTED_WIDTH_DELTA
                 
            # If both inputs are done, exit the loop even if we were not able to extract both images
            if (is_height_done and is_width_done):
                break
   
            # If height is done or distance is bigger for height than width => select width pipeline    
            elif(is_height_done or (distance_to_height[min_height_dist_idxes[h_idx]] > distance_to_width[min_width_dist_idxes[w_idx]])):
                c1 = approx_vertices[min_width_dist_idxes[w_idx]]
                c2 = approx_vertices[min_width_dist_idxes[w_idx] - 1]
                 
                card, is_card_on_top, corners = cls._extract_card_from_two_corners_width(cards_img, c1, c2, approx_vertices)
                w_idx += 1
                
                if plot: print(f"Extracted player card with width pipeline: is top card:{is_card_on_top}")
                    
            else:
                
                c1 = approx_vertices[min_height_dist_idxes[h_idx]]
                c2 = approx_vertices[min_height_dist_idxes[h_idx]-1]
                 
                card, is_card_on_top, corners = cls._extract_card_from_two_corners_height(cards_img, c1, c2, approx_vertices)
                h_idx +=1
                             
                if plot: print(f"Extracted player card with height pipeline: is top card:{is_card_on_top}")
                    
            if (not cls._are_points_in_image(corners, cards_img.shape)):
                if plot: print(f"Detected side lead to corners outisde the image")
            elif (is_card_on_top and type(top_card) == type(None)):
                top_card = card
                top_card_corners = corners


            elif((not is_card_on_top) and type(bottom_card) == type(None)): 
                bottom_card = card
                bottom_card_corners = corners

        if plot:
            fig, axes = plt.subplots(1, 5, figsize=(7, 30),tight_layout=True)
            axes[0].set_title(f"Original")
            axes[0].imshow(cards_img)
            
            contours_img = cv2.drawContours(np.zeros(cards_img.shape), best_contour, -1,(255,0,0),20)
            for v in approx_vertices:
                cv2.circle(contours_img, v, radius=10, color=(0, 255, 0), thickness=15)
            axes[1].set_title(f"Contours")
            axes[1].imshow(contours_img)
            
            pot_corner_img = cv2.drawContours(np.zeros(cards_img.shape), best_contour, -1,(255,0,0),20)
           
            for c in top_card_corners:
                cv2.circle(pot_corner_img, (int(c[0]), int(c[1])), radius=10, color=(255, 255, 0), thickness=15)
                
            for c in bottom_card_corners:
                cv2.circle(pot_corner_img, (int(c[0]), int(c[1])), radius=10, color=(0, 255 , 255), thickness=15)
                
            axes[2].set_title(f"Corners")
            axes[2].imshow(pot_corner_img)
            
            if(type(bottom_card) != type(None)):
                axes[3].set_title(f"Bottom")
                axes[3].imshow(bottom_card)
            else:
                axes[3].set_title(f"No success")
                axes[3].imshow(np.zeros(cls.CARD_SIZE))

            if(type(top_card) != type(None)):
                axes[4].set_title(f"Top")
                axes[4].imshow(top_card)
            else:
                axes[4].set_title(f"No success")
                axes[4].imshow(np.zeros(cls.CARD_SIZE))


            plt.show()

        return bottom_card, top_card
    
    @classmethod
    def _is_point_in_image(cls, pt, img_shape):
        """
        Check if the point is inside the image 
        """
        y, x = pt
        width, height, _ = img_shape
        return 0 <= x and x < width and 0 <= y and y < height
        
    @classmethod
    def _are_points_in_image(cls, pts, img_shape):
        """
        Check if the points are inside the image 
        """
        for pt in pts:
            if(not cls._is_point_in_image(pt, img_shape)):
                return False
        return True
        
    @classmethod
    def _perpendicular_unit_vector_2d(cls, v):
        """
        Compute a unit perpendicular vector to v
        """
        per = [v[1], - v[0]]
        return per / np.linalg.norm(per)
    
    @classmethod
    def _is_top_card(cls,  corner_1, corner_2, mean_pts):
        """
        Return true if the corner are part of top card corners
        """
        
        # Signed distance to righ most corner
        dist_to_right =  max(corner_1[0], corner_2[0]) - mean_pts[0]
        
        # Signed distance to left most corner
        dist_to_left = mean_pts[0] - min(corner_1[0], corner_2[0])
        
        # If the corner have a larger distance to right then it's the card on the top
        return dist_to_right > dist_to_left
         


    @classmethod
    def _extract_card_from_two_corners_width(cls, img, pot_corner_1, pot_corner_2, approx_vertices):
        """
        All point are received with (x, y) coords
        """
        width_vec = pot_corner_2 - pot_corner_1

        # Find third corner along the height of the card
        height_vec = cls._perpendicular_unit_vector_2d(width_vec) * cls.CARD_SIZE[1]
        mean_pts = approx_vertices.mean(axis = 0)

        # If the right_most corner is on the right of the mean then is the top card
        is_card_on_top = cls._is_top_card(pot_corner_1, pot_corner_2, mean_pts)
        
        is_top_corners = mean_pts[1] > min(pot_corner_1[1], pot_corner_2[1])
                
        if is_top_corners:
            if height_vec[1] < 0:
                height_vec = height_vec * -1
        else:
            if height_vec[1] > 0:
                height_vec = height_vec * -1

        pot_corner_3 = pot_corner_1 + height_vec
        pot_corner_4 = pot_corner_2 + height_vec

        # Reorder, and extend by a small margin the side of the card
        corners = ContourHelper.reorder_corners(np.array([pot_corner_1, pot_corner_2, pot_corner_3, pot_corner_4]))
        corners = ContourHelper.add_margin(corners, cls.HEIGHT_CARD_MARGIN_PIX, cls.WIDTH_CARD_MARGIN_PIX)

        # Extract the card
        h = np.array([[0,0],[cls.CARD_SIZE[0],0],[cls.CARD_SIZE[0],cls.CARD_SIZE[1]],[0,cls.CARD_SIZE[1]] ],np.float32)
        transform = cv2.getPerspectiveTransform(corners, h)
        extracted_card = cv2.warpPerspective(img,transform,cls.CARD_SIZE)

        return extracted_card, is_card_on_top, corners
    
    @classmethod
    def _extract_card_from_two_corners_height(cls, img, pot_corner_1, pot_corner_2, approx_vertices):
        """
        All point are received with (x, y) coords
        """
        height_vec = pot_corner_2 - pot_corner_1

        # Find third corner along the width
        width_vec = cls._perpendicular_unit_vector_2d(height_vec) * cls.CARD_SIZE[0]
        mean_pts = approx_vertices.mean(axis = 0)

        # If the right_most corner is on the right of the mean then is the top card
        is_card_on_top = cls._is_top_card(pot_corner_1, pot_corner_2, mean_pts)
        
        # If it's the top card then we detected the right side => left side should have smaller x
        if is_card_on_top:
            if width_vec[0] > 0:
                width_vec = width_vec * -1
                
         # If it's the bottom card then we detected the left side => right side should have bigger x
        else:
            if width_vec[0] < 0:
                width_vec = width_vec * -1

        pot_corner_3 = pot_corner_1 + width_vec
        pot_corner_4 = pot_corner_2 + width_vec

        # Reorder, and extend by a small margin the side of the card
        corners = ContourHelper.reorder_corners(np.array([pot_corner_1, pot_corner_2, pot_corner_3, pot_corner_4]))
        corners = ContourHelper.add_margin(corners, cls.HEIGHT_CARD_MARGIN_PIX, cls.WIDTH_CARD_MARGIN_PIX)

        # Extract the card
        h = np.array([[0,0],[cls.CARD_SIZE[0],0],[cls.CARD_SIZE[0],cls.CARD_SIZE[1]],[0,cls.CARD_SIZE[1]] ],np.float32)
        transform = cv2.getPerspectiveTransform(corners, h)
        extracted_card = cv2.warpPerspective(img,transform,cls.CARD_SIZE)

        return extracted_card, is_card_on_top, corners
        
    @classmethod
    def _leftmost_coordinate(cls, corners):
        #NOTE: uses open cv corner representation (first axis points left)
        return np.min(corners[:,0])
    

    @classmethod
    def _partition_image(cls, img):
        """Partition the image into cards/chips sections"""
        result = {}
        
        result[cls.TABLE_CARDS] = crop(img, cls.TABLE_CARD_BOUNDARIES)
        result[cls.P_1_CARDS] = np.rot90(crop(img, cls.PLAYER_1_CARDS_BOUNDARIES), 3)
        result[cls.P_2_CARDS] = np.rot90(crop(img, cls.PLAYER_2_CARDS_BOUNDARIES), 2)
        result[cls.P_3_CARDS] = np.rot90(crop(img, cls.PLAYER_3_CARDS_BOUNDARIES), 2)
        result[cls.P_4_CARDS] = np.rot90(crop(img, cls.PLAYER_4_CARDS_BOUNDARIES))

        return result
    
    @classmethod
    def _choose_best_pair_contour(cls, candidate_contours,img_area):
        """Assumes we only get 1 shape per candidate contour"""

        best_candidate = []
        min_vertices_count = np.inf
        for contour in candidate_contours:
            #Only one contour kept in image
            contour = contour[0]
            peri = cv2.arcLength(contour,False)
            #Set a strict precision threshold on polynomial approximation
            approx = cv2.approxPolyDP(contour, cls.PAIR_CARD_CONTOUR_APPROX_MARGIN*peri,True)
            vertices_count = len(approx)
            area = cv2.contourArea(contour)
            if(vertices_count>5 and vertices_count<min_vertices_count and area> cls.PAIR_CARD_MIN_AREA * img_area and area < cls.PAIR_CARD_MAX_AREA * img_area):
                min_vertices_count = vertices_count
                best_candidate = contour

        return best_candidate
    
    @classmethod
    def _choose_best_table_cards_contour(cls, candidate_contours,img_area):
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

            if(len(candidate_contour) == 5 and area_standardised_variance < min_variance and area_mean > cls.TABLE_CARD_MIN_AREA*img_area and area_mean < cls.TABLE_CARD_MAX_AREA*img_area):
                min_variance = area_standardised_variance
                best_contour = candidate_contour

        return best_contour
    
    