#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Author       : HUYNH Vinh-Nam
# Email        : namhv.ictlab@gmail.com 
# Created Date : 10-February-2023
# Description  : 
"""
    
"""
#----------------------------------------------------------------------------
import cv2

from imhist import imhist

import numpy as np
import matplotlib.pyplot as plt

import os
import time

# %% Auxiliary functions

def NMHE(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    m, n = gray_image.shape

    histo = imhist(gray_image)/(m*n)
    L = 256
    histo[histo>=(1/L)] = 1/L
    
    u1 = np.ones(L)/L
    moe = 1 - np.sum(u1-histo)
    
    F = np.abs(gray_image[:,3:n].astype(np.uint16) - gray_image[:,1:n-2].astype(np.uint16))
    
    r, c = np.nonzero(F>6)
    h5 = np.ravel_multi_index([r, c], dims=[m, n])
    
    d = gray_image.flatten()[h5]
   
    h22, _ = np.histogram(np.double(d), bins=256)
    h22 = h22/len(r)
    
    hmod = (1-moe)*h22 + (moe*np.transpose(u1))
    hmod1 = hmod
    hmod2 = np.cumsum(hmod1)
    hmod3 = (hmod1/max(hmod2))
    hmod4 = (np.cumsum(hmod3)*255).astype(np.uint8)

    equalized_image = hmod4[gray_image + 1]
    
    return equalized_image


def edge_detection(image):
    # Edge detection on mask 
    t_lower = 50                                     
    t_upper = 200                                    
    edge_img = cv2.Canny(image, t_lower, t_upper)
    
    # Show the edge
    #cv2.imshow('Edge detection', edge_img)
    
    return edge_img


def extract_contour(image, clone):
    # Find contours
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(clone, contours, -1, (0, 255, 0), 1) # Should be 'image' instead of 'clone' image

    # Get the largest contour
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]
    
    return cnt


def approx_contour(image, cnt):
    # Bounding quadrangle
    hull = cv2.convexHull(cnt)
    
    # Approx
    approximations = cv2.approxPolyDP(hull, 0.002 * cv2.arcLength(hull, True), closed = True)
    cv2.drawContours(image, [approximations], 0, (255, 0, 255), 2)
    
    return approximations


# %% API definition
    
def API_web_diameter_measurement(img):
    global dist, error_id
    
    error_id = 0
    
    cX = img.shape[1]//2
    cY = img.shape[0]//2
    
    clone = img.copy()
    clone_gray = NMHE(img)
    
    z = clone_gray.reshape((-1,3))
    
    # Convert to np.float32
    z = np.float32(z)
    
    # Define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3
    ret,label,center=cv2.kmeans(z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((clone_gray.shape))
    
    # Display the image
    cv2.imshow('Image with K=3',res2)
    
    lower = 0
    upper = 70
    mask = cv2.inRange(res2, lower, upper)
    
    #cv2.imshow('Mask', mask)
    
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel1)
    
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel2)
    
    # Make border for close contour at the image edge
    start_point = (0, 0) 
    end_point = (opening.shape[1], opening.shape[0])
    color = (0, 0, 0)
    thickness = 2
    
    opening = cv2.rectangle(opening, start_point, end_point, color, thickness)
    
    edge = edge_detection(opening)
    
    #cv2.imshow('Edge',edge)
    
    cnt = extract_contour(edge, clone)
    approximations = approx_contour(clone, cnt)
    #cv2.imshow('Approximation', clone)
    
    # compute shortest distance of center point from the contour
    dist = cv2.pointPolygonTest(cnt,(cX, cY),True)
    # print(f'Shortest distance of center point:', dist)
    
    cv2.circle(img, (cX, cY), int(np.abs(dist)), (0, 0, 255), 2)
    cv2.imshow("Shapes", img)
    
    x = time.time()
    cv2.imwrite(f'./storage1/Detect_web_{x}.jpg', img)
    return dist, 0, 0, error_id

    
# %% Main function

if __name__ == "__main__":
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass
