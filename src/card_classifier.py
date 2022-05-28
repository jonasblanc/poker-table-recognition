import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt

class CardClassifier:
    """
    Module to classify a playing card
    """
    
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
    
    # Defaut value in case of bad detection (returned with a 0 score)
    DEFAULT_CHAR = "10"
    DEFAULT_RED_SYMBOL= "D"
    DEFAULT_BLACK_SYMBOL= "C"

    MIN_PIXEL_IN_CONNECTED_COMPONENT_OF_INTEREST = 5
    
    # Threshold under which a card is considered to be face down
    BACK_CARD_SIMILARITY_THRESHOLD = 0.15
    
    PATH_CARD_CLASSIFIER_MASKS = "data/card_classifier_masks/"

    # Load boolean symbol masks dict
    with open(PATH_CARD_CLASSIFIER_MASKS + 'symbol_mask_dict.pickle', 'rb') as handle:
        symbol_mask_dict = pickle.load(handle)
        for name, mask in symbol_mask_dict.items():
            symbol_mask_dict[name]= mask.astype(bool)
         
    # Load boolean letter/character masks dict
    with open(PATH_CARD_CLASSIFIER_MASKS + 'char_mask_dict.pickle', 'rb') as handle:
        char_mask_dict = pickle.load(handle)
        for name, mask in char_mask_dict.items():
            char_mask_dict[name]= mask.astype(bool)
   
    @classmethod       
    def classify_cards(cls, card_imgs, card_names, result_dict, can_use_both_corners=False, plot=False):
        """ Classify each card image from card_imgs, and store the result in result_dict with the corresponding card_name
        Arguments:
            card_imgs: List of card images (image = 2D RGB numpyp array)
            card_names: List of keys to store the resulting classification in result_dict
            result_dict: Dict of result {card_name -> card predicted label)
        """
        for name, card_img in zip(card_names, card_imgs):
            label, _ = cls.classify_card(card_img, can_use_both_corners, plot)
            result_dict[name] = label
                
    @classmethod     
    def classify_card(cls, card_img, can_use_both_corners=False ,plot=False):
        """ Classify card_img, and return the predicted label
        Arguments:
            card_img: the card images (image = 2D RGB numpyp array)
            plot: if True (plot / print) the classification processus
        Returns:
            The predicted label of the card
        """
        
        char, score_char = cls._identify_char(card_img, plot)
        
        symbol, score_symbol = cls._identify_symbol(card_img, can_use_both_corners, plot)
        
            
        label = char + symbol
        
        if plot:
            print(f"Predicted: {label} with score char:{score_char}, symbol:{score_symbol}")
       
        return label, (score_char, score_symbol)
    
    @classmethod
    def is_card_face_down(cls, card_img, plot = False):
        """ Predict of the card is face down (back side visible)
        Arguments:
            card_img: the card images (image = 2D RGB numpyp array)
            plot: if True (plot / print) the classification processus
        Returns:
            True if the card is detected as facing the table
        """
        gray = cv2.cvtColor(card_img,cv2.COLOR_BGR2GRAY)
        flag, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        bin_thresh = thresh > 0
        
        # Extract connected components
        connectivity = 4
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(~thresh, connectivity)
        
        diff = abs(bin_thresh.sum() - (~bin_thresh).sum())     
        is_face_down_v1 = ((diff / bin_thresh.size) < cls.BACK_CARD_SIMILARITY_THRESHOLD)
        is_face_down = num_labels > 370
        if (is_face_down and plot):
            print("FACE DOWN")
        return is_face_down
           
        
    @classmethod 
    def _extract_centered_mask_from_img(cls, img, card_name="Default_name", plot=False):
        """
        Extract a binary mask on the biggest connected component centered on his center of mass
        The centered mask the double of width and height of img (to be sure the connected component fits in it)
        Returns:
            centered mask (None if only background) , mean_color of the extracted component
    
        """
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        flag, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

        # Extract the connected component with the biggest area (after background)
        has_cc_of_intrest = True
        connectivity = 4
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(~thresh, connectivity)

        if(num_labels < 2):
            has_cc_of_intrest = False
        else:
            target_label = np.argsort(stats[:,4])[::-1][1]
            
            if((labels==target_label).sum() < cls.MIN_PIXEL_IN_CONNECTED_COMPONENT_OF_INTEREST):
                has_cc_of_intrest = False
        
        # If only background or small component
        if (not has_cc_of_intrest):
            if plot:
                print("No connected component of interest detected")
            return None, img.mean(axis=(0))
             
        # Init the mask
        h, w = labels.shape
        center_y, center_x = centroids[target_label]
     
        #If center is nan set it to the middle
        if center_y != center_y:
            center_y = h/2
        if center_x != center_x :
            center_x = w/2
        
        # Create the centered mask
        x, y = (labels == target_label).nonzero()
        centered_x = x - int(center_x) + h
        centered_y = y - int(center_y) + w
        centered_mask = np.zeros((2*h, 2*w), dtype=bool)
        centered_mask[(centered_x, centered_y)] = True
                       
        # Mean color
        mean_color =  img[np.where(labels == target_label)].mean(axis=(0))

        if(plot): 
            print(title)
            fig, axes = plt.subplots(1, 5, figsize=(7, 30),tight_layout=True)
            axes[0].set_title(f"Img")
            axes[0].imshow(img)
            axes[1].set_title(f"CC")
            axes[1].imshow(labels)
            img_copy = img.copy()
            img_copy[np.where(labels != target_label)] = [255, 255, 255]
            axes[2].set_title(f"Zone of interest")
            axes[2].imshow(img_copy)
            axes[3].set_title(f"Centered mask")
            axes[3].imshow(centered_mask)

        plt.show()

        return centered_mask, mean_color      
            
    # ================== CHAR Pipeline (2,3,4,5,6,7,8,9,10,J,Q,K,A) ================== #
    
    @classmethod     
    def _identify_char(cls, card_img, can_use_both_corners = False, plot=False):
        """
        Predict the char of the card, by default use only top left corner
        If can_use_both_corners= True also use bottom right corner
        """
        
        def crop_char(card_img):
            """
            Crop the card image to select only the character on the top left corner
            """
            h, w, _ = card_img.shape
            cropped = card_img[int(cls.TOP_CHAR * h):int(cls.BOTTOM_CHAR * h), int(cls.LEFT_CHAR *  w): int(cls.RIGHT_CHAR * w)]
            return cropped
        
        def identify_char(card_img, plot):
            """
            Extract the mask of the char and classify it
            Returns:
                The predicted label
                The score (intersection/union) obtained with the mask of predicted label
            If no connected component in the image returns (DEFAULT_CHAR, 0)
            """
            mask, _  = cls._extract_centered_mask_from_img(crop_char(card_img))
            
            if type(mask) == type(None):
                char = cls.DEFAULT_CHAR  
                if plot: 
                    print(f"No char detected, set to default: {char}")
                return char, 0
            
            char, score = cls._classify_char_mask(mask, plot)
            
            return char, score
        
        # Predict label of char on the top left corner
        char_1, score_1 = identify_char(card_img, plot)
       
        # Try to predict on the botton right corner by rotating the image by 180deg
        if can_use_both_corners:
            card_img_rotated = np.rot90(card_img, 2)
            char_2, score_2 = identify_char(card_img_rotated, plot)
            
            # Keep the prediction with the best score
            return (char_1, score_1) if score_1 > score_2 else (char_2, score_2)

        return (char_1, score_1)
         
    
    @classmethod 
    def _classify_char_mask(cls, mask, plot = False):
        """
        Classify a binary mask for a character to the labelled mask with the biggest score
        Score = intersection / union
        """
        best_score = -1
        best_label = None
        for label, label_mask in cls.char_mask_dict.items():
            score = (mask & label_mask).sum() / (mask | label_mask).sum()
            if (score > best_score):
                best_score = score
                best_label = label
        return best_label, best_score
            
    # ================= Symbol Pipeline (S, C, H, D) ================= #

    @classmethod     
    def _identify_symbol(cls, card_img, can_use_both_corners = False, plot=False):
        """
        Predict the symbol of the card, by default use only top left corner
        If can_use_both_corners=True also use bottom right corner
        """
        
        def crop_symbol(card_img):
            """
            Crop the card image to select only the symbol on the top left corner
            """
            h, w, _ = card_img.shape
            cropped = card_img[int(cls.TOP_SYMBOL * h):int(cls.BOTTOM_SYMBOL * h), int(cls.LEFT_SYMBOL *  w): int(cls.RIGHT_SYMBOL * w)]
            return cropped 
        
        def identify_symbol(card_img, plot):
            """
            Extract the mask of the symbol and classify it
            Returns:
                The predicted label
                The score (intersection/union) obtained with the mask of predicted label
            If no connected component in the image returns (DEFAUT_SYMBOL based on the color, 0)
            """
            mask, mean_color  = cls._extract_centered_mask_from_img(crop_symbol(card_img))
            
            # Determine closest color between red and black
            dist_R = np.abs([255, 0, 0] - mean_color).sum()
            dist_B = mean_color.sum()
            color = 'r' if dist_B > dist_R  else 'b'
            
            if type(mask) == type(None):
                default_symbol = cls.DEFAULT_RED_SYMBOL if color == "r" else  cls.DEFAULT_BLACK_SYMBOL 
                if plot:
                    print(f"No symbol detected, set to default: {default_symbol}")   
                return default_symbol, 0
            
            symbol, score = cls._classify_symbol_mask(color, mask, plot)
            
            return symbol, score
        
        symbol_1, score_1 = identify_symbol(card_img, plot)
        
        # Try to predict symbol on bottom right corner by rotating the image by 180deg
        if can_use_both_corners:
            card_img_rotated = np.rot90(card_img, 2)
            
            symbol_2, score_2 = identify_symbol(card_img_rotated, plot)
            
            # Keep the prediction with the best score
            return (symbol_1, score_1) if score_1 > score_2 else (symbol_2, score_2)
        
        return symbol_1, score_1 

           
    
    @classmethod 
    def _classify_symbol_mask(cls, color, mask, plot=False):
        """
        Classify binary mask of a symbol the labelled mask that has the biggest score
        First classify the symbol based on its mean color (either red or black)
        Then compare it the labelled masks
        """
       
        def compute_score(mask, symbol_name):
            return (cls.symbol_mask_dict[symbol_name] & mask).sum() / (cls.symbol_mask_dict[symbol_name] | mask).sum()
        
        if color == "r":
            score_H = compute_score(mask, "H")
            score_D = compute_score(mask, "D")

            return ('H', score_H) if score_H > score_D else ('D', score_D)
        elif color == "b":
            score_C = compute_score(mask, "C")
            score_S = compute_score(mask, "S")
            return ('C', score_C) if score_C > score_S else ('S', score_S)