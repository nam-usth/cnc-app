import cv2
import imutils

import glob

import numpy as np

import os
import time

from sklearn.cluster import KMeans

# %% Initial values

frame_id_max = -1
val_max = -1
search_flag = 0
sharpness = 0
sharpness_list = [] # frame_max list

THRESHOLD = 350

# %% Auxiliary functions

def crop_img(image):
    
    h, w = image.shape[0], image.shape[1]
    
    x1 = 0 #w//4
    y1 = 0 #h//4
    x2 = w #3*w//4
    y2 = h #3*h//4

    crop_image = image[y1:y2, x1:x2]
    return crop_image

def axial_angle_measurement_reset():
    global angle, error_id
    
    files = glob.glob('./storage/*.jpg')
    for f in files:
        os.remove(f)

# %% API definition
    
def API_axial_angle_measurement(image):
    global angle, error_id
    
    error_id = 0
    angle = 0
    
    try:
        image_clone = image.copy()
        
        cropped_img = crop_img(image)
        gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        
        #thresh_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 199, 5)
        _, thresh_img = cv2.threshold(gray_img, 190, 255, cv2.THRESH_BINARY)
        
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closing = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel1)
    
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel2)
        
        contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                                      
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt = contours[max_index]
        
        # Bounding quadrangle
        hull = cv2.convexHull(cnt)
        
        # Approx
        approximations = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), closed = True)
    
        #cv2.fillPoly(opening, [approximations], (0, 0, 0))
                                   
        edge_image = cv2.Canny(opening, 50, 100, apertureSize=3)
    
        cv2.imshow('Binary', thresh_img)
        cv2.imshow('Opening', opening)
        cv2.imshow('Edge', edge_image)
    
        lines = []
        
        try:
            lines = cv2.HoughLines(edge_image, 1, np.pi/180, 50)
            
            best_line = lines[0]
            rho, theta = best_line[0]
            
            a = np.cos(theta)
            b = np.sin(theta)
            
            x0 = a*rho
            y0 = b*rho
            
            x1 = int(x0 + 999999*(-b))
            y1 = int(y0 + 999999*(a))
            
            x2 = int(x0 - 999999*(-b))
            y2 = int(y0 - 999999*(a))
            
            cv2.line(crop_img(image_clone), (x1, y1), (x2, y2), (0, 0, 255), 1)
        
            cv2.imshow('Lines', crop_img(image_clone))
            
            x = time.time()
            cv2.imwrite(f'./storage1/AXIAL_ANGLE_Lines_{x}.jpg', crop_img(image_clone))
            
            angle = theta #np.pi/2 - theta
        except:
            pass
            
    except:
        if len(lines) == 0:
            error_id = 1004
    
    return angle, 0, error_id

# %% Main function

if __name__ == "__main__":

    # For testing ONLY
    image = cv2.imread(f'D:/Working/KhoanCNC/Nam_work/experiment/axial-measure/gash-4-200.jpeg')
    angle, move_focus_flg, error_id = API_axial_angle_measurement(image)
    
    print(angle)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    pass
