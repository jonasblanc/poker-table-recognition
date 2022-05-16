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
        
    def __init__(self):
        PATH_CARD_CLASSIFIER_MASKS = "data/card_classifier_masks/"

        with open(PATH_CARD_CLASSIFIER_MASKS + 'name_to_mask.pickle', 'rb') as handle:
            self.name_to_mask = pickle.load(handle)
            for name, mask in self.name_to_mask.items():
                self.name_to_mask[name]= mask.astype(bool)
                
        with open(PATH_CARD_CLASSIFIER_MASKS + 'char_to_mask.pickle', 'rb') as handle:
            self.char_to_mask = pickle.load(handle)
            for name, mask in self.char_to_mask.items():
                self.char_to_mask[name]= mask.astype(bool)
                
                
    def classify_card(self, card_img, card_name="Default_name", plot=False):
        color, symbol_mask = self._extract_symbol_feature(card_img, card_name, plot)
        symbol = self._classify_symbol_mask(color, symbol_mask)

        char_mask = self._extract_card_character(card_img, card_name, plot)
        char = self._classify_char_mask(char_mask)
        
        return char + symbol
        
                
    def _create_centered_mask(self, cc_labels, target_label, centroids):
  
        h, w = cc_labels.shape
        center_y, center_x = centroids[target_label]
        x, y = (cc_labels == target_label).nonzero()
        centered_x = x - int(center_x) + h
        centered_y = y - int(center_y) + w
        centered_mask = np.zeros((2*h, 2*w), dtype=bool)
        centered_mask[(centered_x, centered_y)] = True
        
        return centered_mask
            
# ======================== CHAR Pipeline (2,3,4,5,6,7,8,9,10,J,Q,K,A) ========================
     
    def _crop_char(self, card_img):
        h, w, _ = card_img.shape
        cropped = card_img[int(self.TOP_CHAR * h):int(self.BOTTOM_CHAR * h), int(self.LEFT_CHAR *  w): int(self.RIGHT_CHAR * w)]
        return cropped
    
    def _plot_pipeline(self, title, cropped, thresh, cc_labels, target_label, centered_mask):
        print(title)
        fig, axes = plt.subplots(1, 5, figsize=(7, 30),tight_layout=True)
        axes[0].set_title(f"Cropped")
        axes[0].imshow(cropped)
        axes[1].set_title(f"Threshold")
        axes[1].imshow(thresh)
        axes[2].set_title(f"CC")
        axes[2].imshow(cc_labels)
        img = cropped.copy()
        img[np.where(cc_labels != target_label)] = [255, 255, 255]
        axes[3].set_title(f"Extracted symbol")
        axes[3].imshow(img)
        axes[4].set_title(f"Centered mask")
        axes[4].imshow(centered_mask)

        plt.show()
            
    def _extract_card_character(self, card_img, card_name="Default_name", plot=False):
   
        cropped = self._crop_char(card_img)

        gray = cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY)
        flag, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

        # Extract connected components
        connectivity = 4
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(~thresh, connectivity)

        # Find label with the second biggest area (after background)
        char_label = np.argsort(stats[:,4])[::-1][1]
        
        centered_mask = self._create_centered_mask(labels, char_label, centroids)

  
        if(plot): 
            self._plot_pipeline(f"Char extraction pipeline for {card_name}", cropped, thresh, labels, char_label, centered_mask)

        return centered_mask
    
    def _classify_char_mask(self, mask): 
        best_score = -1
        best_label = None
        for label, label_mask in self.char_to_mask.items():
            score = (mask & label_mask).sum() / (mask | label_mask).sum()
            if (score > best_score):
                best_score = score
                best_label = label
        return best_label
            
# ======================== Symbol Pipeline (S, C, H, D) ========================


    def _crop_symbol(self, card_img):
        h, w, _ = card_img.shape
        cropped = card_img[int(self.TOP_SYMBOL * h):int(self.BOTTOM_SYMBOL * h), int(self.LEFT_SYMBOL *  w): int(self.RIGHT_SYMBOL * w)]
        return cropped        
    
    def _extract_symbol_feature(self, card_img, card_name="Default_name", plot=False):
        """
        Extract features about symbol/color from an image of a card
        Extracted feature:
             Cropped image of the symbol
             Detected color ('r'-red or 'b'-black)
             A mask of the symbol centered at (25,25)
        Assume:
            The symbol is always at the same place defined by the constant below
            The symbol is the second largest component after the background at that location
            The symbol fits in 50x50

        """

        # Focus on symbol
        cropped = self._crop_symbol(card_img)

        gray = cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY)
        flag, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

        # Extract connected components
        connectivity = 4
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(~thresh, connectivity)

        # Find label with the second biggest area (after background)
        symbol_label = np.argsort(stats[:,4])[::-1][1]

        centered_mask = self._create_centered_mask(labels, symbol_label, centroids)

        # Mean color in symbol
        mean_color =  cropped[np.where(labels == symbol_label)].mean(axis=(0))

        # Determine closest color between red and black
        dist_R = np.abs([255, 0, 0] - mean_color).sum()
        dist_B = mean_color.sum()
        color = 'r' if dist_B > dist_R  else 'b'

        if(plot):
            self._plot_pipeline(f"Symbol extraction pipeline for {card_name}", cropped, thresh, labels, symbol_label, centered_mask)

        return color, centered_mask

    def _classify_symbol_mask(self, color, mask):
        if color == "r":
            score_H = (self.name_to_mask["H"] & mask).sum() / (self.name_to_mask["H"] | mask).sum()
            score_D = (self.name_to_mask["D"] & mask).sum() / (self.name_to_mask["D"] | mask).sum()

            return 'H' if score_H > score_D else 'D'
        elif color == "b":
            score_C = (self.name_to_mask["C"] & mask).sum() / (self.name_to_mask["C"] | mask).sum()
            score_S = (self.name_to_mask["S"] & mask).sum() / (self.name_to_mask["S"] | mask).sum()
            return 'C' if score_C > score_S else 'S'    