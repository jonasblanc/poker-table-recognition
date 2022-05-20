import pickle
import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt


class CardClassifier:
    
    # Symbol position on card
    TOP_SYMBOL = 0.13
    BOTTOM_SYMBOL = 0.26
    LEFT_SYMBOL = 0.016
    RIGHT_SYMBOL = 0.17
    
    # Char position
    TOP_CHAR = 0
    BOTTOM_CHAR = 0.16
    LEFT_CHAR = 0.016
    RIGHT_CHAR = 0.18
    
    DEFAULT_CHAR = "10"
    DEFAULT_RED_SYMBOL= "D"
    DEFAULT_BLACK_SYMBOL= "C"

    MIN_PIXELS_IN_CHAR = 5
    MIN_PIXELS_IN_SYMBOL = 5
    
    PATH_CARD_CLASSIFIER_MASKS = "data/card_classifier_masks/"

    with open(PATH_CARD_CLASSIFIER_MASKS + 'name_to_mask.pickle', 'rb') as handle:
        name_to_mask = pickle.load(handle)
        for name, mask in name_to_mask.items():
            name_to_mask[name]= mask.astype(bool)
                
    with open(PATH_CARD_CLASSIFIER_MASKS + 'char_to_mask.pickle', 'rb') as handle:
        char_to_mask = pickle.load(handle)
        for name, mask in char_to_mask.items():
            char_to_mask[name]= mask.astype(bool)
   
    @classmethod       
    def classify_cards(cls, card_imgs, card_names, result_dict, plot=False):
        for name, card_img in zip(card_names, card_imgs):
            label = cls.classify_card(card_img, name, plot)
            result_dict[name] = label
                
    @classmethod     
    def classify_card(cls, card_img, card_name="Default_name", plot=False):
        # Color + symbol mask        
        symbol_mask, mean_symbol_color  = cls._extract_centered_mask_from_img(cls._crop_symbol(card_img))
        symbol = cls._classify_symbol_mask(mean_symbol_color, symbol_mask, plot)

        # Char mask    
        char_mask, _  = cls._extract_centered_mask_from_img(cls._crop_char(card_img))
        char = cls._classify_char_mask(char_mask, plot)
        
        label = char + symbol
        
        if plot:
            print(f"Predicted: {label}")
       
        return label
           
        
    @classmethod 
    def _extract_centered_mask_from_img(cls, zone_img, card_name="Default_name", plot=False):
   
        #cropped = cls._crop_char(card_img)
    
        labels, target_label, centroids = cls._extract_second_biggest_connected_component(zone_img, cls.MIN_PIXELS_IN_CHAR)
        if type(labels) == type(None):
            return None, zone_img.mean(axis=(0))
               
        centered_mask = cls._create_centered_mask_from_cc(labels, target_label, centroids)
        
        # Mean color
        mean_color =  zone_img[np.where(labels == target_label)].mean(axis=(0))

        if(plot): 
            print(title)
            fig, axes = plt.subplots(1, 5, figsize=(7, 30),tight_layout=True)
            axes[0].set_title(f"Img")
            axes[0].imshow(card_img)
            axes[1].set_title(f"CC")
            axes[1].imshow(labels)
            img = zone_img.copy()
            img[np.where(labels != target_label)] = [255, 255, 255]
            axes[2].set_title(f"Zone of interest")
            axes[2].imshow(img)
            axes[3].set_title(f"Centered mask")
            axes[3].imshow(centered_mask)

        plt.show()

        return centered_mask, mean_color
    
    @classmethod     
    def _create_centered_mask_from_cc(cls, cc_labels, target_label, centroids):
  
        h, w = cc_labels.shape
        center_y, center_x = centroids[target_label]
     
        #If center is nan set it to the middle
        if center_y != center_y:
            center_y = h/2
        if center_x != center_x :
            center_x = w/2
        x, y = (cc_labels == target_label).nonzero()
        centered_x = x - int(center_x) + h
        centered_y = y - int(center_y) + w
        centered_mask = np.zeros((2*h, 2*w), dtype=bool)
        centered_mask[(centered_x, centered_y)] = True
        
        return centered_mask
    
    @classmethod
    def _extract_second_biggest_connected_component(cls, img, min_pixel_in_component):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        flag, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

        # Extract connected components
        connectivity = 4
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(~thresh, connectivity)

        if(num_labels < 2):
            return None, None, None
        
        # Find label with the second biggest area (after background)
        target_label = np.argsort(stats[:,4])[::-1][1]
        
        if((labels==target_label).sum() < min_pixel_in_component):
            return None, None, None
        
        return labels, target_label, centroids
        
            
# ======================== CHAR Pipeline (2,3,4,5,6,7,8,9,10,J,Q,K,A) ========================
     
    @classmethod 
    def _crop_char(cls, card_img):
        h, w, _ = card_img.shape
        cropped = card_img[int(cls.TOP_CHAR * h):int(cls.BOTTOM_CHAR * h), int(cls.LEFT_CHAR *  w): int(cls.RIGHT_CHAR * w)]
        return cropped
     
    @classmethod 
    def _classify_char_mask(cls, mask, plot = False):
        
        if type(mask) == type(None):
            char = cls.DEFAULT_CHAR  
            if plot: print(f"No char detected, set to default: {char}")
            return char
        best_score = -1
        best_label = None
        for label, label_mask in cls.char_to_mask.items():
            score = (mask & label_mask).sum() / (mask | label_mask).sum()
            if (score > best_score):
                best_score = score
                best_label = label
        return best_label
            
# ======================== Symbol Pipeline (S, C, H, D) ========================

    @classmethod 
    def _crop_symbol(cls, card_img):
        h, w, _ = card_img.shape
        cropped = card_img[int(cls.TOP_SYMBOL * h):int(cls.BOTTOM_SYMBOL * h), int(cls.LEFT_SYMBOL *  w): int(cls.RIGHT_SYMBOL * w)]
        return cropped        
    
    @classmethod 
    def _classify_symbol_mask(cls, mean_color, mask, plot=False):
        
        # Determine closest color between red and black
        dist_R = np.abs([255, 0, 0] - mean_color).sum()
        dist_B = mean_color.sum()
        color = 'r' if dist_B > dist_R  else 'b'
        
        if type(mask) == type(None):
            default_symbol = cls.DEFAULT_RED_SYMBOL if color == "r" else  cls.DEFAULT_BLACK_SYMBOL 
            if plot:
                print(f"No symbol detected, set to default: {default_symbol}")   
            return default_symbol
            
        if color == "r":
            score_H = (cls.name_to_mask["H"] & mask).sum() / (cls.name_to_mask["H"] | mask).sum()
            score_D = (cls.name_to_mask["D"] & mask).sum() / (cls.name_to_mask["D"] | mask).sum()

            return 'H' if score_H > score_D else 'D'
        elif color == "b":
            score_C = (cls.name_to_mask["C"] & mask).sum() / (cls.name_to_mask["C"] | mask).sum()
            score_S = (cls.name_to_mask["S"] & mask).sum() / (cls.name_to_mask["S"] | mask).sum()
            return 'C' if score_C > score_S else 'S'    