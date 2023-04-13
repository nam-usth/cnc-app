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

def rake_angle_measurement_reset():
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
    
def API_rake_angle_measurement_step1(image, length, px2mm):
    global angle, error_id
    
    error_id = 0
    angle = 0
    
    x_center = image.shape[1] // 2
    y_center = image.shape[0] // 2
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

        edge_image = cv2.Canny(opening, 50, 100, apertureSize=3)

        #cv2.imshow('Binary', thresh_img)
        #cv2.imshow('Opening', opening)
        cv2.imshow('Edge', edge_image)
        
        # Define the range of angles to detect lines with HoughLine
        angle_range = np.deg2rad(np.array([60, 80]))
        
        # Detect lines with HoughLine algorithm
        lines = cv2.HoughLines(edge_image, 1, np.pi/180, threshold=65)
        
        valid_lines = []
        
        # Loop through each detected line and draw it on the image
        for line in lines:
            rho, theta = line[0]
            angle = theta % np.pi  # Get the angle of the line
            if angle_range[0] <= angle <= angle_range[1]:
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
                
        # Convert line_data to a 2D array
        valid_lines = np.squeeze(valid_lines)

        # Create a feature matrix with rho and theta
        X = valid_lines[:, :2]
        
        # Use KMeans to cluster the lines
        kmeans = KMeans(n_clusters=1, random_state=43).fit(X)
        
        # Get the centroids of the clusters
        centroids = kmeans.cluster_centers_
        
        # Plot the average lines based on their cluster
        avg_rho = centroids[0][0]
        avg_theta = centroids[0][1]

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
        cv2.imwrite(f'./storage1/RAKE_ANGLE_angle_step_{x}.jpg', crop_img(image_clone))
        
        angle = angle_between_lines(avg_rho, avg_theta, x_center, 0)

    except:
        if len(lines) == 0:
            error_id = 1004
    
    return angle, 0, error_id

def API_rake_angle_measurement_step2(image, length, px2mm):
    global p1, p2, p3, error_id
    
    error_id = 0
    p1 = []
    p2 = []
    p3 = []
    
    x_center = image.shape[1] // 2
    y_center = image.shape[0] // 2
   # try:
    image_clone = image.copy()
    
    cropped_img = crop_img(image) # length
    gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    
    #thresh_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 199, 5)
    _, thresh_img = cv2.threshold(gray_img, 175, 255, cv2.THRESH_BINARY)
    
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closing = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel1)

    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel2)
    
    #cv2.imshow('Step 2 Openning', opening)
    
    x1 = time.time()
    cv2.imwrite(f'./storage1/RAKE_ANGLE_opening_{x1}.jpg', opening)
    
    contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]
    
    hull = cv2.convexHull(cnt)
    
    approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), closed = True)

    cv2.drawContours(image_clone, [approx], -1, (255, 0, 255), 2)
    
    '''
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.1*cv2.arcLength(cnt, True), True)
        if len(approx) == 3:
           cv2.drawContours(image_clone, [approx], -1, (255, 0, 255), 2)
           break
    '''
    
    # The highest point means lowest y
    # The left most point means lowest x
    # The right most point is the last point in the array
    
    #A = np.reshape(np.array(approx).flatten(), (3,2))
    
    approx_2D = np.reshape(np.array(approx).flatten(), (len(np.array(approx).flatten()) // 2, 2))
    
    kmeans = KMeans(n_clusters=3, random_state=0).fit(approx_2D)
    
    A = kmeans.cluster_centers_
    
    highest_point = A[np.argmin(A[:, 1])]
    left_most_point = A[np.argmin(A[:, 0])]
    
    B = np.reshape(A[A != highest_point], (2,2))
    right_most_point = B[B != left_most_point]

    '''
    p1.append(highest_point + [-10, 20])
    p2.append(left_most_point + [20, -10])
    p3.append(right_most_point - [20, 10])
            
    cv2.circle(image_clone, p1[0], 2, (0, 0, 255), -1)
    cv2.circle(image_clone, p2[0], 2, (0, 255, 0), -1)
    cv2.circle(image_clone, p3[0], 2, (255, 0, 0), -1)
    '''       
    centroid_of_triangle = np.mean(A, axis=0)
    
    res = (2*centroid_of_triangle + A) / 3
    res = res.astype(int)
    
    cv2.circle(image_clone, res[0], 2, (20, 20, 255), -1)
    cv2.circle(image_clone, res[1], 2, (20, 255, 20), -1)
    cv2.circle(image_clone, res[2], 2, (255, 20, 20), -1)
    
    x = time.time()
    #cv2.imshow('Rake 3 points', image_clone)
    cv2.imwrite(f'./storage1/RAKE_ANGLE_Lines_3_points_{x}.jpg', crop_img(image_clone))
    
    '''
    p1_mm = [round((p1[0][0] - x_center) * px2mm, 3), round((p1[0][1] - y_center) * px2mm, 3)]
    p2_mm = [round((p2[0][0] - x_center) * px2mm, 3), round((p2[0][1] - y_center) * px2mm, 3)]
    p3_mm = [round((p3[0][0] - x_center) * px2mm, 3), round((p3[0][1] - y_center) * px2mm, 3)]
    '''
    
    p1.append(res[0][0])
    p2.append(res[1][0])
    p3.append(res[2][0])

    p1_mm = [round((res[0][0] - x_center) * px2mm, 3), round((res[0][1] - y_center) * px2mm, 3)]
    p2_mm = [round((res[1][0] - x_center) * px2mm, 3), round((res[1][1] - y_center) * px2mm, 3)]
    p3_mm = [round((res[2][0] - x_center) * px2mm, 3), round((res[2][1] - y_center) * px2mm, 3)]
    '''    
    except Exception as e:
        print("aaaaaa", e)
        if len(p1)*len(p2)*len(p3) == 0:
            p1_mm = p2_mm = p3_mm = [0, 0]
            error_id = 1004
    '''
    return p1_mm, p2_mm, p3_mm, 1, error_id

# %% Main function

if __name__ == "__main__":

    # For testing ONLY
    #image1 = cv2.imread(f'D:/Working/KhoanCNC/Nam_work/experiment/thinning-measure/step1.png')
    #angle, move_focus_flg, error_id = API_rake_angle_measurement_step1(image1, 0, 0)
    
    #print("Rad : ", angle)

    #image2 = cv2.imread(f'D:/Working/KhoanCNC/Nam_work/experiment/thinning-measure/rake1.jpeg')
    #p1, p2, p3, detect_flg, error_id = API_rake_angle_measurement_step2(image2, 0, 0.048)

    #print(p1, p2, p3)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    pass
