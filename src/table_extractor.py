import cv2
import math 
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

class LineHelper:
    """
    Helper to handle all line related operations
    """

    @classmethod
    def transform_lines_from_polar_to_points(cls, lines):
        """ Transform lines from polar coordinates to a two point cartesian description
        Arguments:
            lines: List of line encoded in polar coordinates
        Returns: 
            list of line encoded by two points on the line on a x,y plan
        """
        lines_pts = []
        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                lines_pts.append((pt1, pt2))

        return lines_pts

    @classmethod
    def compute_line_slope(cls, p1, p2):
        """ Compute the slope of the line passing by points p1 and p2
        Arguments:
            p1: (x,y) first point
            p2: (x,y) second point
        Returns: 
            slope of the line going trough p1 and p2, 'NA' in case of vertical line
        """
        x1, y1 = p1
        x2, y2 = p2
        if x2!=x1:
            return((y2-y1)/(x2-x1))
        else:
            return 'NA'

        lines_cosine_sim
    @classmethod
    def compute_line_cosine_similarity(cls, line1, line2):
        """ Compute the cosine similarity between two lines
        Arguments:
            line1: ((x1, y1),(x2, y2)) first line 
            line2: ((x1, y1),(x2, y2)) second line 
        Returns: 
            The cosine similarity between the line1 and line2
        """
        
        line1_vec = np.array((line1[0][0] - line1[1][0], line1[0][1] - line1[1][1]))
        line2_vec = np.array((line2[0][0] - line2[1][0], line2[0][1] - line2[1][1]))
        cosine = np.dot(line1_vec,line2_vec)/(norm(line1_vec)*norm(line2_vec))
        return cosine

    @classmethod
    def compute_lines_intersection_point(cls, line1, line2):
        """ Compute intersection point between two lines
        Arguments:
            line1: ((x1, y1),(x2, y2)) first line 
            line2: ((x1, y1),(x2, y2)) second line 
        Returns: 
            The intersection point between line1 and line2
            Raise an exception if the line does not intersect
        """
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            raise Exception('lines do not intersect')

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y

    @classmethod
    def compute_intersection_points_in_img(cls, lines, img_shape):
        """ Find all the intersection points between lines that respect the following criterias:
            - Lines intersects within the image
            - Lines have an abs cosine similarity < 0.5
        Arguments:
            lines: [(x1, y1),(x2, y2)] list of lines encoded by two points
        Returns: 
            A list of intersection points
        """
        intersection_pts = []
        for pt1_a, pt2_a in lines:
            for pt1_b, pt2_b in lines:
                if(abs(LineHelper.compute_line_cosine_similarity((pt1_a, pt2_a), (pt1_b, pt2_b))) < 0.5):
                    ip_1, ip_2 = LineHelper.compute_lines_intersection_point([pt1_a, pt2_a], [pt1_b, pt2_b])

                    if(0 <= ip_1 and ip_1 < img_shape[1] and 0 <= ip_2 and ip_2 < img_shape[0]):
                        intersection_pts.append((int(ip_1), int(ip_2)))
        return intersection_pts

    @classmethod
    def draw_line(cls, image, p1, p2):
        """
        Draw a "infinite" line on image passing by points p1 and p2
        """

        h , w = image.shape[:2]

        x1, y1 = p1
        x2, y2 = p2
        slope = LineHelper.compute_line_slope(p1,p2)

        if slope!='NA':
            # extending the line to border x = 0
            px = 0
            py =- (x1 - 0) * slope + y1

            # extending the line to border x = width
            qx = w
            qy =- (x2 - w) * slope + y2
        else:
            # create vertical line
            px, py = x1, 0
            qx, qy = x1, h

        cv2.line(image, (int(px), int(py)), (int(qx), int(qy)), (0, 255, 0), 2)


class TableExtractor:
    """
    Module to identify and extract the table from a top view image.
    """
    
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
        blur = cv2.GaussianBlur(gray,(101,101),10) 
        flag, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel= np.ones((11,11), np.uint8))
        
        # Edge detection
        edges = cv2.Canny(opened, 50, 200, None, 3)
        
        # Detect main lines
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 150, None, 0, 0)
        
        # Convert lines represetation
        lines_pts = LineHelper.transform_lines_from_polar_to_points(lines)
        
        # Find all intersections points within the image
        intersection_pts = LineHelper.compute_intersection_points_in_img(lines_pts, img.shape)

        centers_pts = TableExtractor._find_intersection_points_clusters(intersection_pts)
        
        h = np.array([ [0,0],[table_size_px,0],[table_size_px,table_size_px],[0,table_size_px] ],np.float32)
        transform = cv2.getPerspectiveTransform(rectify(np.array(centers_pts)), h)
        table_img = cv2.warpPerspective(img,transform,(table_size_px,table_size_px)) 
        
        
        if (plot):
            debug_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
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
    def _find_intersection_points_clusters(cls,intersection_pts):
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
                
    