import cv2
import numpy as np
import os
import math

def detect_nose_phase(img):
    h, w, c = img.shape
    roi = img
    gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    _, binary = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=lambda x: cv2.contourArea(x))
    hull = cv2.convexHull(cnt, False)
    hull_reshape = np.reshape(hull, (-1, 2))
    
    # compute x+y 
    hull_plus = np.sum(hull_reshape, axis=1)
    # compute -x+y (measure diameter)
    hull_minus = np.add(np.sum(np.multiply(hull_reshape, [-1, 1]), axis=1), w)
    
    max_plus = np.amax(hull_plus)
    max_minus = np.amax(hull_minus)
    
    thres_minus = max_minus - (w *0.01 + h *0.01)
    
    idx_minus = [index for index, value in enumerate(hull_minus) if value >= thres_minus]
    
    x_min = 1000
    for it in idx_minus:
        if x_min > hull_reshape[it][0]:
            x_min = hull_reshape[it][0]
            res_min = hull_reshape[it]
        
        cv2.circle(roi, (hull_reshape[it][0], hull_reshape[it][1]), 1, (0, 0, 255), -1)
        
        if hull_reshape[it][0] + hull_reshape[it][1] == max_plus: # max_minus ??
            cv2.circle(roi, (hull_reshape[it][0], hull_reshape[it][1]), 1, (0, 255, 255), -1)

    point = (res_min[0], res_min[1])
    
    return point

def distance(center, point):
    x1, y1 = center
    x2, y2 = point
    return x2 - x1, y2 - y1

def measure_nose_phase(img):
    center = (int(img.shape[1] / 2), int(img.shape[0] / 2))
    point = detect_nose_phase(img)
    x_distance, y_distance = distance(center, point)
    return x_distance, y_distance
    
    