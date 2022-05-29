import cv2
import math 
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

from line_helper import LineHelper
from contour_helper import ContourHelper

class TableExtractor:
    """
    Module to identify and extract the table from a top view image.
    """
    
    EXTRACTION_MARGIN_HEIGHT = 20
    EXTRACTION_MARGIN_WIDTH = 20
    
    @classmethod
    def extract_tables(cls, imgs, table_size_px=3800, plot = False):
        """ Extract images of the table from alist of top view images 
        Arguments:
            imgs: list of images to be analysed (image = 2D RGB numpy array)
            table_size_px: the width of the resulting image in pixel
            plot: if True (plot / print) the extraction processus
        Returns: 
            The extracted images of the table
        """
        tables = []
        for img in imgs:
            tables.append(cls.extract_table(img, table_size_px, plot))
        return tables
    
    @classmethod
    def extract_table(cls, img, table_size_px=3800, plot = False):
        """ Extract an image of the table from a top view image 
        Arguments:
            img: the original image (2D RGB numpy array)
            table_size_px: the width of the resulting image in pixel
            plot: if True (plot / print) the extraction processus
        Returns: 
            The extracted image of the table
        """
        
        # Image preprocessing
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 101)

        flag, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel= np.ones((11,11), np.uint8))
        
        contour = ContourHelper.extract_biggest_contours(opened, 1)
        contour_img = cv2.drawContours(np.zeros_like(opened), contour, -1,(1),1)
        
        # Detect main lines
        lines = cv2.HoughLines(contour_img, 1, np.pi / 180, 150, None, 0, 0)
        
        # Convert lines represetation
        lines_pts = LineHelper.transform_lines_from_polar_to_points(lines)
        
        # Find all intersections points within the image
        intersection_pts = LineHelper.compute_intersection_points_in_img(lines_pts, img.shape)        

        #centers_pts = cls._find_intersection_points_clusters_grid(intersection_pts)
        centers_pts = np.array(cls._find_intersection_points_clusters_distance(intersection_pts))
        centers_pts = ContourHelper.reorder_corners(centers_pts)
        centers_pts = ContourHelper.add_margin(centers_pts, cls.EXTRACTION_MARGIN_HEIGHT, cls.EXTRACTION_MARGIN_WIDTH)

        h = np.array([ [0,0],[table_size_px,0],[table_size_px,table_size_px],[0,table_size_px] ],np.float32)
        transform = cv2.getPerspectiveTransform(centers_pts, h)
        table_img = cv2.warpPerspective(img,transform,(table_size_px,table_size_px)) 
        
        if (plot):
            debug_img = cv2.cvtColor(contour_img, cv2.COLOR_GRAY2BGR)
            for pt1, pt2 in lines_pts:
                LineHelper.draw_line(debug_img, pt1, pt2)
                
            for pts in intersection_pts:
                cv2.circle(debug_img, pts, radius=13, color=(255, 255, 255), thickness=5)
                
            for x,y in centers_pts:
                cv2.circle(debug_img, (int(x),int(y)), radius=100, color=(0, 0,255), thickness=10)  
                
            fig, axes = plt.subplots(1, 3, figsize=(10, 30),tight_layout=True)
            axes[0].set_title(f"Orginal")
            axes[0].imshow(img)
            axes[1].set_title(f"Lines")
            axes[1].imshow(debug_img)
            axes[2].set_title(f"Table")
            axes[2].imshow(table_img)

            plt.show()
                
        return table_img
    
    @classmethod
    def _find_intersection_points_clusters_distance(cls, pts):
        """
        Find biggest cluster separated by a certain distance
        """
        clusters = dict()
    
        max_dist_cluster =  50
        min_dist_between_clusters = 2000

        for i in range(len(pts)):
            cluster_i = []
            for j in range(len(pts)):
                a,b = pts[i]
                x,y = pts[j]
                dist = norm(np.array((x-a, y-b)))
                if max_dist_cluster >= dist:
                    cluster_i.append((x,y))

            clusters[i] = (len(cluster_i), np.array(cluster_i).mean(axis=0))

        sortedKeys = sorted(clusters.keys(),key= lambda x: clusters[x][0])
        sortedKeys.reverse()

        center_pts = [clusters[sortedKeys[0]][1]]
        for k in sortedKeys:
            if(len(center_pts) == 4):
                break

            a,b = clusters[k][1]

            is_far_enough = True
            for x,y in center_pts:
                dist = norm(np.array((x-a, y-b)))
                is_far_enough = is_far_enough and (dist >= min_dist_between_clusters)

            if (is_far_enough):
                center_pts.append((a,b))

        return center_pts
                
    @classmethod
    def _find_intersection_points_clusters_grid(cls,intersection_pts):
        grid_step = 500
        
        buckets = dict()
        for a, b in intersection_pts:
            idx = (a//grid_step, b//grid_step)
            bucket = buckets.get(idx, [])
            bucket.append((a,b))
            buckets[idx] = bucket
        sortedKeys = sorted(buckets.keys(),key= lambda x:len(buckets[x]))
        sortedKeys.reverse()

        center_pts = []
        for i in range(min(len(sortedKeys), 4)):
            key = sortedKeys[i]
            b = buckets[key]
            center_point = (np.array(b).sum(axis=0) / len(b)).astype(int)
            center_pts.append(center_point)
        return center_pts