import cv2
import numpy as np
from sklearn.cluster import KMeans

class ContourHelper:
    """
    Helper module to extract contours from an image
    """
    
    @classmethod
    def reorder_corners(cls, corners):
        """ 
        Reorder the corners so they have a deterministic order: top_left,top_right,bottom_right,bottom_left] 
            corners: numpy array of (4,2)
        """       
        corners = corners.reshape((4,2))
        new_corners = np.zeros((4,2),dtype = np.float32)

        add = corners.sum(1)
        new_corners[0] = corners[np.argmin(add)]
        new_corners[2] = corners[np.argmax(add)]

        diff = np.diff(corners,axis = 1)
        new_corners[1] = corners[np.argmin(diff)]
        new_corners[3] = corners[np.argmax(diff)]

        return new_corners 
    
    @classmethod
    def add_margin(cls, corners, margin_height, margin_width):
        """
        Assume corners are in order (top left, top right, bottom right, bottom left)
        """
        down_height_vec = corners[3] - corners[0]
        down_height_margin_vec =  down_height_vec / np.linalg.norm(down_height_vec) * margin_height

        right_width_vec = corners[1] - corners[0]
        right_width_margin_vec =  right_width_vec / np.linalg.norm(right_width_vec) * margin_width

        new_corners = []
        new_corners.append(list(corners[0] - down_height_margin_vec - right_width_margin_vec)) # top left - down - right
        new_corners.append(list(corners[1] - down_height_margin_vec + right_width_margin_vec)) # top right - down + right
        new_corners.append(list(corners[2] + down_height_margin_vec + right_width_margin_vec)) # bottom right + down + right
        new_corners.append(list(corners[3] + down_height_margin_vec - right_width_margin_vec)) # bottom left + down - right

        return np.array(new_corners)
    
    @classmethod
    def plot_contours(cls, contours, background, title,ax=None):
        """
        Plot the contour on top of background image
        """
        img = cv2.drawContours(background.copy(), contours, -1,(0,255,0),20)

        if(ax==None):
            fig,ax = plt.subplots(1,1,tight_layout=True)

        ax.imshow(img)
        ax.axis('off')
        ax.set_title(title)
    
    @classmethod
    def extract_biggest_contours(cls, bin_img, number_contour):
        """ Extract the number_contour biggest area contours in the binary image
        Arguments:
            bin_img: 2D binary image (numpy array)
            number_contour:number of biggest contours we want
        Returns:
            A list of contours in decreasing order of area (may return less contours than number_contour)
        """
        contours, hierarchy = cv2.findContours(bin_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=cv2.contourArea,reverse=True)[:number_contour]  
        return contours
    
    @classmethod
    def extract_candidate_contours(cls, img, number_contour, n_thresholds = 2, opening_kernel_size = None, plot = False):
        """ Extract mutliple set of contours canddiates from an image based on each HSV channel 
        Arguments:
            img: image (2D RGB image - numpy array)
            number_contour: number contours to be extracted from the image
            n_thresholds: number of threshold for color thresholding (default 2 => 3 color clusters)
            opening_kernel_size: size of the kernel for opening after thresholding (by default no opening)
            plot: if True (plot / print) the extraction / thresholding processus
        Returns:
            A list of list of contours in decreasing order of area 
        """

        if(plot):
            
            number_plot_per_HSV = 2+4*n_thresholds  # (image_HSV, hist, thresholded image, opened_img, contour, countour_inv)
            fig, axes = plt.subplots(number_plot_per_HSV, 3, figsize=(20, 20),tight_layout=True)
        
        # Image preprocessing
        img = cv2.GaussianBlur(img,(11,11),100) 
        img_HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        contour_candidates = []
        for i in range(img_HSV.shape[2]):
            img_grey = img_HSV[:,:,i]

            # K-mean thresholding
            # Index by stride of 16 to reduce by 256 the number of pixels K-means has to process
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

            # For every threshold, create a binary image and extract candidate contours
            for k, threshold in enumerate(thresholds):
                flag, thresh_img = cv2.threshold(img_grey, threshold, 255, cv2.THRESH_BINARY)
                opened_img = thresh_img
                opened_img_inv = ~thresh_img
                
                if(opening_kernel_size!=None):
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (opening_kernel_size,opening_kernel_size))
                    opened_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel)
                    opened_img_inv = cv2.morphologyEx(~thresh_img, cv2.MORPH_OPEN, kernel)

                # Extract contours for the thresholded image and it's inverse
                contours = cls.extract_biggest_contours(opened_img, number_contour)
                if(len(contours)>0):
                    contour_candidates.append(contours)
                    
                contours_inv = cls.extract_biggest_contours(opened_img_inv, number_contour)
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
                    cls.plot_contours(contours, np.zeros_like(img), "Thresholding contours",ax)
                    ax_index+=1

                    ax = axes[ax_index][i]
                    cls.plot_contours(contours_inv, np.zeros_like(img), "Inverted Thresholding contours",ax)
                    ax_index+=1

        if(plot):
            plt.show()

        return contour_candidates