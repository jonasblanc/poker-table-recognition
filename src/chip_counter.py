import cv2
import time
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from utility_functions import *


class ChipCounter:
    
    HSV_COLOR_BOUNDS = {
        'blue': [(np.array([96, 15, 100]), np.array([104, 255, 255]))],
        'green': [(np.array([31, 80, 0]), np.array([97, 255, 255]))],
        'red':[(np.array([140, 70, 70]), np.array([200, 255, 255])), (np.array([0, 140, 50]), np.array([16, 255, 155]))],
        'black': [(np.array([0, 0, 0]), np.array([130, 194, 100])), (np.array([100, 145, 85]), np.array([118, 221, 120]))],
        'white': [(np.array([78, 0, 178]), np.array([132, 85, 235]))]
        }

    COLOR_TO_SIMBOL = {'red':'CR','blue':'CB','green':'CG','black':'CK', 'white': 'CW'}

    CHIPS_CENTER_INTENSITY_TRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9] # list(np.linspace(0.5, 1, 20))
    CHIPS_BOUNDARIES = [0.25,0.75,0.25,0.75]
    
    MEAN_BRIGHTNESS = 180.0
    STD_BRIGHTNESS = 47.0

    @classmethod           
    def _crop_chips(cls, table_img):
        """
        Crop the table image to select only the chips area
        """
        return crop(table_img, cls.CHIPS_BOUNDARIES)

    @classmethod
    def _brightness_stats_hsv(cls, img_hsv):
        """
        Compute the mean and standard deviation of the brightness
        """
        h,s,v = cv2.split(img_hsv)
        return v.mean(), v.std()

    @classmethod
    def _normalize_brightness_hsv(cls, img_hsv):
        """
        Align the brightness of the image to the mean computed on training images
        """
        img_v_mean, img_v_std = cls._brightness_stats_hsv(img_hsv)
        h,s,v = cv2.split(img_hsv.astype("float32"))

        v -= img_v_mean
        v *= (img_v_std / cls.STD_BRIGHTNESS)
        v += cls.MEAN_BRIGHTNESS
        v = np.clip(v, 0, 255)

        reconstructed_img_hsv = cv2.merge([h, s, v])

        return reconstructed_img_hsv.astype("uint8") 
    
    @classmethod
    def count_chips(cls, table_img, results_dict, plot=False):
        """
        Count chips by color on a rgb image
        Store result in results_dict with color key
        """
        
        def detect_circle(img):
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            gray = cv2.medianBlur(gray, 15)

            rows = gray.shape[0]
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100,
                                   param1=40, param2=10,
                                   minRadius=128, maxRadius=135)
            return circles
            
        def create_circular_mask(h, w, center=None, radius=None):

            if center is None: # use the middle of the image
                center = (int(w/2), int(h/2))
            if radius is None: # use the smallest distance between the center and image walls
                radius = min(center[0], center[1], w-center[0], h-center[1])

            Y, X = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

            mask = dist_from_center <= radius
            return mask
        
        # Select chips area
        chips_rgb = cls._crop_chips(table_img)
        
        # Preprocess chips for color thresholding
        blured_chips_rgb = cv2.medianBlur(chips_rgb,ksize=51)
        chips_hsv = cv2.cvtColor(blured_chips_rgb, cv2.COLOR_RGB2HSV)
        chips_hsv = cls._normalize_brightness_hsv(chips_hsv)

        
        # Create binary mask for each color
        colors_maks = {}
        color_names = cls.HSV_COLOR_BOUNDS.keys()
        for i, color_name in enumerate(color_names):
            chips_mask = np.zeros((chips_hsv.shape[0], chips_hsv.shape[1]), bool)
                    
            for low, high in cls.HSV_COLOR_BOUNDS[color_name]:
                chips_mask |= cv2.inRange(chips_hsv,low,high).astype(bool)
                
            colors_maks[color_name] = chips_mask
        
        # Detect circles and assign it color with the largest intersection
        color_counts = Counter()
        circles = detect_circle(chips_rgb)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                radius = i[2]
                
                # Coord of pixel inside the circle
                mask = create_circular_mask(chips_rgb.shape[0], chips_rgb.shape[1], center, radius)
                
                max_count = 0
                best_color_name = None
                for i, color_name in enumerate(color_names):
                    
                    # Number of pixel in the circle of the given color 
                    intersection = mask & colors_maks[color_name]
                    count = intersection.sum()
                  
                    if count > max_count:
                        max_count = count
                        best_color_name = color_name

                if best_color_name != None:
                    color_counts[best_color_name]+=1
                    
                    if plot:
                        fig, axes = plt.subplots(1, 3, figsize=(4, 4), tight_layout=True)

                        axes[0].imshow(chips_rgb)
                        axes[0].set_title(f"Original")
                        
                        axes[1].imshow(colors_maks[best_color_name])
                        axes[1].set_title(f"Best color mask")

                        axes[2].imshow(mask)
                        axes[2].set_title(f"circle mask")
                        plt.show()       

                else:
                    print("Warning circle with no thresholded color inside")
        

        for color in color_names:
            results_dict[cls.COLOR_TO_SIMBOL[color]]=color_counts[color]

        return results_dict
    
    @classmethod
    def count_chips_distance_map(cls, table_img, results_dict, plot=False):
        """
        Count chips by color on a rgb image
        Store result in results_dict with color key
        """
                
        chips_rgb = cls._crop_chips(table_img)        
        blured_chips_rgb = cv2.medianBlur(chips_rgb,ksize=51)
        chips_hsv = cv2.cvtColor(blured_chips_rgb, cv2.COLOR_RGB2HSV)
        chips_hsv = cls._normalize_brightness_hsv(chips_hsv)

        color_names = cls.HSV_COLOR_BOUNDS.keys()

        if(plot):
            fig, axes = plt.subplots(1, 2, figsize=(4, 4), tight_layout=True)

            axes[0].imshow(chips_rgb)
            axes[0].set_title(f"Blurred")

            axes[1].imshow(cv2.cvtColor(chips_hsv, cv2.COLOR_HSV2RGB))
            axes[1].set_title(f"Norm brightness")
            plt.plot()       

            fig, axes = plt.subplots(4, len(color_names), figsize=(10, 10), tight_layout=True)

        color_counts = Counter()
        for i, color_name in enumerate(color_names):

            # Some color may have multiple ranges
            chips_mask = np.zeros((chips_hsv.shape[0], chips_hsv.shape[1]), bool)
            for low, high in cls.HSV_COLOR_BOUNDS[color_name]:
                chips_mask |= cv2.inRange(chips_hsv,low,high).astype(bool)
            chips_mask = chips_mask.astype("uint8")

            # Remove noise from mask
            kernel = np.ones((81,81),np.uint8)
            chips_mask_cleaned = cv2.morphologyEx(chips_mask,cv2.MORPH_OPEN,kernel)


            # Extract centers of chips in mask
            chips_distance_map = cv2.distanceTransform(chips_mask_cleaned, cv2.DIST_C, 5)
            max_distance = chips_distance_map.max()

            max_count = 0
            best_labels = np.zeros((chips_distance_map.shape[0], chips_distance_map.shape[1]))
            max_centers = []
            best_threshold = 0
            for treshold in cls.CHIPS_CENTER_INTENSITY_TRESHOLDS:

                _, chips_centers = cv2.threshold(chips_distance_map, treshold * max_distance,255,0)
                chips_centers = chips_centers.astype(np.uint8)

                # Count number of connected components (chips)
                connectivity=8
                num_labels, labels, _, centers = cv2.connectedComponentsWithStats(chips_centers , connectivity , cv2.CV_32S)
                num_chips = num_labels - 1 # Remove the background
                if num_chips > max_count:
                    max_count = num_chips
                    best_labels = labels
                    max_centers = centers[1:]
                    best_threshold = treshold


            color_counts[color_name] = max_count

            if(plot):
                print(f"Detected {max_count} tokens of color {color_name} with tresh: {best_threshold}")

                idx=0
                axes[idx][i].imshow(chips_mask_cleaned)
                axes[idx][i].set_title(f"Tresh: {color_name}")
                idx+=1

                axes[idx][i].imshow(chips_distance_map)
                axes[idx][i].set_title(f"Distance map")
                idx+=1

                axes[idx][i].imshow(best_labels)
                axes[idx][i].set_title(f"Top distance")
                idx+=1

                corners_img = chips_rgb.copy()
                for x, y in max_centers:
                    cv2.circle(corners_img, (int(x), int(y)), radius=10, color=(0, 255, 0), thickness=15)
                axes[idx][i].imshow(corners_img)
                axes[idx][i].set_title(f"Centers")

                idx+=1


        if(plot):
            plt.show()

        for color in color_names:
            results_dict[cls.COLOR_TO_SIMBOL[color]]=color_counts[color]

        return results_dict

    @classmethod
    def window_treshold(cls, img, color, range_idx=0):

        def nothing(x):
            pass

        def get_window_title(low_hsv,high_hsv):
            window_title = f"""Tresholding for {color} | low_HSV:{low_hsv} | high_HSV:{high_hsv}"""

            return window_title

        low_hsv= cls.HSV_COLOR_BOUNDS[color][range_idx][0]
        high_hsv= cls.HSV_COLOR_BOUNDS[color][range_idx][1]

        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # Create a window
        window_name='window'
        cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)
        cv2.setWindowTitle(window_name, get_window_title(low_hsv,high_hsv))

        # create trackbars for color change
        cv2.createTrackbar('highH',window_name,high_hsv[0],255,nothing)
        cv2.createTrackbar('lowH',window_name,low_hsv[0],255,nothing)


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