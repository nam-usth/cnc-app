import cv2
import imutils

import glob

import math
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

def pocket_angle_measurement_reset():
    global angle, error_id
    
    files = glob.glob('./storage/*.jpg')
    for f in files:
        os.remove(f)
        
def angle_between_lines(rho1, theta1, rho2, theta2):
    x1, y1 = rho1 * math.cos(theta1), rho1 * math.sin(theta1)
    x2, y2 = rho2 * math.cos(theta2), rho2 * math.sin(theta2)
    dot_product = x1 * x2 + y1 * y2
    norm1 = math.sqrt(x1**2 + y1**2)
    norm2 = math.sqrt(x2**2 + y2**2)
    angle = math.acos(dot_product / (norm1 * norm2))
    return angle

# %% API definition
    
def API_pocket_angle_measurement(image):
    global angle, error_id
    
    error_id = 0
    angle = 0
    
    try:
        image_clone = image.copy()
        
        cropped_img = crop_img(image)
        gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        
        #thresh_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 199, 5)
        _, thresh_img = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY)
        
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
        closing = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel1)
    
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
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

        #cv2.imshow('Binary', thresh_img)
        #cv2.imshow('Opening', opening)
        #cv2.imshow('Edge', edge_image)
        
        # Define the range of angles to detect lines with HoughLine
        angle_range1 = np.deg2rad(np.array([150, 170]))
        angle_range2 = np.deg2rad(np.array([60, 80]))
        
        # Detect lines with HoughLine algorithm
        lines = cv2.HoughLines(edge_image, 1, np.pi/180, threshold=65)
        
        valid_lines = []
        
        # Loop through each detected line and draw it on the image
        for line in lines:
            rho, theta = line[0]
            angle = theta % np.pi  # Get the angle of the line
            if angle_range1[0] <= angle <= angle_range1[1]:
                # Line with angle in range of angle_range1
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 10000 * (-b))
                y1 = int(y0 + 10000 * (a))
                x2 = int(x0 - 10000 * (-b))
                y2 = int(y0 - 10000 * (a))
                #cv2.line(crop_img(image_clone), (x1, y1), (x2, y2), (0, 0, 255), 2)
                valid_lines.append(line)
            elif angle_range2[0] <= angle <= angle_range2[1]:
                # Line with angle in range of angle_range2
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 10000 * (-b))
                y1 = int(y0 + 10000 * (a))
                x2 = int(x0 - 10000 * (-b))
                y2 = int(y0 - 10000 * (a))
                #cv2.line(crop_img(image_clone), (x1, y1), (x2, y2), (0, 255, 0), 2)
                valid_lines.append(line)
                
        # Convert line_data to a 2D array
        valid_lines = np.squeeze(valid_lines)
        
        # Create a feature matrix with rho and theta
        X = valid_lines[:, :2]
        
        # Use KMeans to cluster the lines
        kmeans = KMeans(n_clusters=2, random_state=43).fit(X)
        
        # Get the centroids of the clusters
        centroids = kmeans.cluster_centers_

        '''
        # Plot the lines with different colors based on their cluster
        for i in range(len(valid_lines)):
            rho = valid_lines[i][0]
            theta = valid_lines[i][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            if kmeans.labels_[i] == 0:
                cv2.line(crop_img(image_clone), (x1, y1), (x2, y2), (0, 0, 255), 2)
            else:
                cv2.line(crop_img(image_clone), (x1, y1), (x2, y2), (0, 255, 0), 2)
        '''     
        
        # Plot the average lines based on their cluster
        for i in range(2):
            avg_rho = centroids[i][0]
            avg_theta = centroids[i][1]

            a = np.cos(avg_theta)
            b = np.sin(avg_theta)
            x0 = a * avg_rho
            y0 = b * avg_rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            
            cv2.line(crop_img(image_clone), (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.imshow('Lines', crop_img(image_clone))
        
        x = time.time()
        cv2.imwrite(f'./storage1/POCKET_ANGLE_Lines_{x}.jpg', crop_img(image_clone))
        
        angle = angle_between_lines(centroids[0][0], centroids[0][1], centroids[1][0], centroids[1][1])

    except:
        if len(lines) == 0:
            error_id = 1004
    
    return angle, 0, error_id

# %% Main function

if __name__ == "__main__":

    # For testing ONLY
    image = cv2.imread(f'D:/Working/KhoanCNC/Nam_work/experiment/pocket-measure/goc-pocket-001.jpeg')
    angle, move_focus_flg, error_id = API_pocket_angle_measurement(image)
    
    print(angle)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    pass
