import cv2
import matplotlib.pyplot as plt

def plot_HSV_Contour(img):
    fig, axes = plt.subplots(6, 3, figsize=(30, 18))
    img = cv2.GaussianBlur(img,(11,11),100) 
    img_HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    for i in range(3):
        axes[0][i].imshow(img_HSV[:,:,i])
        axes[0][i].axis('off')
        axes[0][i].set_title(f'Image with {i+1}th HSV component')
        
        edges = cv2.Canny(img_HSV[:,:,i],100,200)
        axes[1][i].imshow(edges)
        axes[1][i].axis('off')
        axes[1][i].set_title(f'Image with Canny')

        flag, tresh = cv2.threshold(img_HSV[:,:,i], 0, 255, cv2.THRESH_OTSU)
        axes[2][i].imshow(tresh,cmap='gray')
        axes[2][i].axis('off')
        axes[2][i].set_title(f'Image with tresholding')
        

        #Opening
        kernel_size = (50,50)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        opened = cv2.morphologyEx(tresh, cv2.MORPH_OPEN, kernel)
        axes[3][i].imshow(opened,cmap='gray')
        axes[3][i].axis('off')
        axes[3][i].set_title(f'Image after opening')
        
        #Closing
        kernel_size = (40,40)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

        axes[4][i].imshow(closed,cmap='gray')
        axes[4][i].axis('off')
        axes[4][i].set_title(f'Image after closing')

        #cv.RETR_EXTERNAL: only keeps external contours
        #cv.RETR_TREE: hierarchical
        contours, hierarchy = cv2.findContours(tresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        #lastarguments are contour color
        #Contour thickness
        tresh_color = cv2.cvtColor(tresh,cv2.COLOR_GRAY2RGB)*255
        img = cv2.drawContours(tresh_color.copy(), contours, -1,(0,255,0),20)

        axes[5][i].imshow(img)
        axes[5][i].axis('off')
        axes[5][i].set_title(f'Image with Contours')

    plt.show()