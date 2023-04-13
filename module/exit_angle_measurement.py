import cv2
import imutils

import glob

import numpy as np

import os
import time

from sklearn.cluster import KMeans
from tools.Static import STATIC
from tools.Static import *

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

def angle_measurement_reset():
    global angle, error_id
    
    files = glob.glob('./storage/*.jpg')
    for f in files:
        os.remove(f)

# %% API definition
    
def API_exit_angle_division_line_detection(image):
    global angle, error_id
    
    error_id = 0
    angle = 0
    
    try:
        image_clone = image.copy()
        
        cropped_img = crop_img(image)
        gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        
        # Detect line angle
        thresh_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 199, 5)

        erode = cv2.erode(thresh_img, np.ones((5, 5), np.uint8)) 
        
        #cv2.imshow('Erode', erode)
        
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        closing = cv2.morphologyEx(erode, cv2.MORPH_CLOSE, kernel1)
    
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel2)

        contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                                      
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt = contours[max_index]
        
        # Bounding quadrangle
        hull = cv2.convexHull(cnt)
        
        # Approx
        approximations = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), closed = True)
                                   
        edge_image = cv2.Canny(opening, 50, 100, apertureSize=3)
    
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
                        
            x = time.time()
            
            cv2.imwrite(f'./storage1/EXIT_ANGLE_Straight_Lines_{x}.jpg', crop_img(image_clone))
            
            angle = np.pi/2 - theta
            angle = np.degrees(angle)
        except:
            pass
            
    except:
        if len(lines) == 0:
            error_id = 1004
    
    return angle, error_id


def API_exit_angle_2_points(image, focus_length, angle):    
    error_id = 0
    
    p1 = []
    p2 = []
    
    try:
        # Get image dimensions
        height, width = image.shape[:2]

        # Calculate the center of the image
        center_x = width // 2
        center_y = height // 2
        
        image_clone = image.copy()
        
        cropped_img = crop_img(image)
        gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

        # Detect 2 points
        _, thresh_img = cv2.threshold(gray_img, 90, 255, cv2.THRESH_BINARY)
        
        erode = cv2.erode(thresh_img, np.ones((5, 5), np.uint8)) 
                
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        closing = cv2.morphologyEx(erode, cv2.MORPH_CLOSE, kernel1)
    
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel2)
                                  
        edge_image = cv2.Canny(opening, 50, 100, apertureSize=3)
        
        # Calculate the angle in radians
        angle = np.radians(angle) # -3.76
        
        # Calculate the startpoint of the line
        start_x = int(center_x - np.cos(angle) * 400)
        start_y = int(center_y + np.sin(angle) * 400)
        
        # Calculate the endpoint of the line
        end_x = int(center_x + np.cos(angle) * 400)
        end_y = int(center_y - np.sin(angle) * 400)
        
        # Draw the line on the image
        m = (end_y - start_y) / (end_x - start_x)
        b = start_y - m * start_x
        
        for x in range(center_x + 50, image.shape[1], 5):
            
            y = int(m * x + b)

            total_intersect = []
            for y_intersect in range(0, cropped_img.shape[0]):
                if edge_image[y_intersect, x] > 0:
                    total_intersect.append([x, y_intersect])
                            
            if len(total_intersect) > 1:
                kmeans = KMeans(n_clusters=2, random_state=0).fit(total_intersect)
            else:
                kmeans = KMeans(n_clusters=1, random_state=0).fit(total_intersect)
                
            kmeans.labels_
            points = kmeans.cluster_centers_
            points = points.astype(int)
            points = points[points[:, 1].argsort()]

            
            if len(points) > 1:
                intersect_p1 = points[0]
                intersect_p2 = points[1]

                d1 = np.linalg.norm(np.array(intersect_p1) - np.array([x, y]))
                d2 = np.linalg.norm(np.array(intersect_p2) - np.array([x, y]))

            else:
                intersect_p1 = points[0]
                intersect_p2 = [-1, -1]

                d1 = np.linalg.norm(np.array(intersect_p1) - np.array([x, y]))
                d2 = 0.0
            
            if (int(d1) > focus_length * 2.5):    
                midpoint_p1 = np.mean([intersect_p1, [x, y]], axis=0).astype(int)
            else:
                midpoint_p1 = np.array([-1, -1])
                
            if (int(d2) > focus_length * 2.5):    
                midpoint_p2 = np.mean([intersect_p2, [x, y]], axis=0).astype(int)
            else:
                midpoint_p2 = np.array([-1, -1])
                
            cv2.circle(cropped_img, intersect_p1, 2, (175, 150, 255), -1)
            cv2.circle(cropped_img, intersect_p2, 2, (175, 150, 255), -1)
            cv2.circle(cropped_img, (x, y), 2, (175, 150, 255), -1)
            cv2.circle(cropped_img, midpoint_p1, 2, (0, 0, 255), -1)
            cv2.circle(cropped_img, midpoint_p2, 2, (0, 255, 0), -1)
            
            if len(p1) == 0:
                if np.array_equal(midpoint_p1, np.array([-1, -1])) == False:
                    p1.append(midpoint_p1)
                    
            if len(p2) == 0:
                if np.array_equal(midpoint_p2, np.array([-1, -1])) == False:
                    p2.append(midpoint_p2)
                    
            if len(p1)*len(p2) != 0:
                break
            
        cv2.circle(cropped_img, p1[0], focus_length, (255, 0, 0))
        cv2.circle(cropped_img, p2[0], focus_length, (255, 0, 0))
    
    except:
        pass
    
    x = time.time()
    cv2.imwrite(f'./storage1/EXIT_ANGLE_2_points_{x}.jpg', cropped_img)
    
    return p1, p2, error_id


def API_exit_angle_4_points(image, focus_length, angle):
    error_id = 0
    p1 = []
    p2 = []
    p3 = []
    p4 = []
    
    try:
        image_clone = image.copy()
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Get image dimensions
        height, width = image.shape[:2]

        # Calculate the center of the image
        center_x = width // 2
        center_y = height // 2 
               
        # Calculate the angle in radians
        angle = np.radians(angle)
        
        # Calculate the startpoint of the line
        start_x = int(center_x - np.cos(angle) * 400)
        start_y = int(center_y + np.sin(angle) * 400)
        
        # Calculate the endpoint of the line
        end_x = int(center_x + np.cos(angle) * 400)
        end_y = int(center_y - np.sin(angle) * 400)
        
        cv2.line(image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 1)
        
        # Apply adaptive threshold
        #thresh_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 199, 2) # Need to use second_largest contour
        _, thresh_img = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23))
        closing = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel1)
        
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel2)
        
        edge_img = cv2.Canny(opening, 50, 100, apertureSize=3)
        
        #cv2.imshow('Adaptive Gaussian', thresh_img)
        #cv2.imshow('Opening', opening)
        
        contours, hierarchy = cv2.findContours(edge_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # Find the index of the contour with the second largest length
        lengths = [len(c) for c in contours]
        largest = sorted(range(len(lengths)), key=lambda i: lengths[i])[-1] # if need the second_largest, use -2 instead of -1
        
        edge_img = np.zeros((height, width, 1), dtype = "uint8")
        
        cv2.drawContours(edge_img, contours, largest, (255, 255, 255), 1)
        
        edge_img_no_line = edge_img.copy()
        
        cv2.line(edge_img, (start_x, start_y), (end_x, end_y), (255, 255, 255), 1)
        
        # Compute line equation coefficients using two given points
        m = (end_y - start_y) / (end_x - start_x)
        b = start_y - m * start_x
        for x in range(center_x, image.shape[1], 5):
            total_intersect = []
            y = int(m * x + b)
            try:
                # Check if the vertical line intersects with edges in the binary image
                for y_intersect in range(0, image.shape[0]):
                    if edge_img[y_intersect, x] > 0:
                        total_intersect.append([x, y_intersect])
            except:
                pass
                        
            if len(total_intersect) >= 4:
                kmeans = KMeans(n_clusters=3, random_state=0).fit(total_intersect)
            
                kmeans.labels_
                points = kmeans.cluster_centers_
                points = points.astype(int)
                points = points[points[:, 1].argsort()]
                
                if len(p1) == 0:
                    p1.append(points[0] + [0, 20])
                if len(p2) == 0:
                    p2.append(points[1] - [0, 20])
                if len(p3) == 0:
                    p3.append(points[1] + [0, 20])
                if len(p4) == 0:
                    p4.append(points[2] - [20, 20])
                    
                cv2.circle(image, p1[0], 2, (0, 0, 255), -1)
                cv2.circle(image, p2[0], 2, (0, 0, 255), -1)
                cv2.circle(image, p3[0], 2, (0, 255, 0), -1)
                cv2.circle(image, p4[0], 2, (0, 255, 0), -1)
                
                if len(p1)*len(p2)*len(p3)*len(p4) != 0:
                    break
            
            cv2.circle(image, (x, y), 2, (175, 150, 255), -1)  
            
    except:
        pass
    x = time.time()
    cv2.imwrite(f'./storage1/EXIT_ANGLE_4_points_{x}.jpg', image)
    
    return p1, p2, p3, p4, error_id

def get_angle(img):
    cv2.imwrite(f"./storage1/angle_{STATIC.index}.jpg", img)
    STATIC.index += 1
    angle, error_id = API_exit_angle_division_line_detection(img)
    return angle, error_id

def get_2_points(img, data):
    cv2.imwrite(f"./storage1/2_point_{STATIC.index}.jpg", img)
    STATIC.index += 1
    focus_width = data["focusWidth"]
    focus_length = data["focusLength"]
    px2mm = data["px2mm"]
    print("2 point px2mm : ", px2mm)
    angle = data["angle"]
    length = min(focus_width, focus_length)
    x_center = img.shape[1] // 2
    y_center = img.shape[0] // 2
    p1, p2, error_id = API_exit_angle_2_points(img, length, angle)
    p1_mm = []
    p2_mm = []
    
    if error_id == 0:
        p1 = p1[0].tolist()
        p2 = p2[0].tolist()
        p1_mm = [round((p1[0] - x_center) * px2mm, 3), round((p1[1] - y_center) * px2mm, 3)]
        p2_mm = [round((p2[0] - x_center) * px2mm, 3), round((p2[1] - y_center) * px2mm, 3)]
        print("p1 : ", p1_mm)
        print("p2 : ", p2_mm)
    return p1_mm, p2_mm, error_id

def get_4_points(img, data):
    cv2.imwrite(f"./storage1/4_point_{STATIC.index}.jpg", img)
    STATIC.index += 1
    focus_width = data["focusWidth"]
    focus_length = data["focusLength"]
    px2mm = data["px2mm"]
    print("4 point px2mm : ", px2mm)
    angle = data["angle"]
    length = min(focus_width, focus_length)
    x_center = img.shape[1] // 2
    y_center = img.shape[0] // 2
    p1, p2, p3, p4, error_id = API_exit_angle_4_points(img, length, angle)
    p1_mm = []
    p2_mm = []
    p3_mm = []
    p4_mm = []
    if error_id == 0:
        p1 = p1[0].tolist()
        p2 = p2[0].tolist()
        
        p3 = p3[0].tolist()
        p4 = p4[0].tolist()
        
        p1_mm = [round((p1[0] - x_center) * px2mm, 3), round((p1[1] - y_center) * px2mm, 3)]
        p2_mm = [round((p2[0] - x_center) * px2mm, 3), round((p2[1] - y_center) * px2mm, 3)]
        p3_mm = [round((p3[0] - x_center) * px2mm, 3), round((p3[1] - y_center) * px2mm, 3)]
        p4_mm = [round((p4[0] - x_center) * px2mm, 3), round((p4[1] - y_center) * px2mm, 3)]
    
        print("p1 : ", p1_mm)
        print("p2 : ", p2_mm)
        print("p3 : ", p3_mm)
        print("p4 : ", p4_mm)
    return p1_mm, p2_mm, p3_mm, p4_mm, error_id
