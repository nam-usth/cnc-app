import cv2
import imutils

import numpy as np

import os
import time

# %% Auxiliary functions

def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()

def moving_average(L, n=5):
    # n is the number of entries in a single window
    ret = np.cumsum(L, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def non_increasing(L):
    return np.all(np.around(np.diff(moving_average(np.array(L))), decimals=6)<=0)

def find_border(imgray):
    _, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_cont = 0
    count = []
    img_seg = np.zeros((imgray.shape[0], imgray.shape[1], 3))
    for _, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > max_cont:
            count = contour
            max_cont = area
    
    cv2.fillPoly(img_seg, [count], (255, 255, 255))

    img_erosion = cv2.dilate(img_seg, np.ones((9,9),np.uint8), iterations=1)

    img_dilation = cv2.dilate(img_seg, np.ones((11,11),np.uint8), iterations=1)

    subtracted = img_dilation - img_erosion

    subtracted = np.array(subtracted, dtype=np.uint8)

    return subtracted

def distant(points, x, el):
    for i in points:
        if abs(x - i) < el:
            return False
    return True

def smoothImage(img):
    blur = cv2.GaussianBlur(img, (0,0), sigmaX=5, sigmaY=5, borderType = cv2.BORDER_DEFAULT)
    result = skimage.exposure.rescale_intensity(blur, in_range=(200,255), out_range=(0,255))
    result = np.array(result, dtype=np.uint8)
    return result

def get_corner(image):
    pass

def compute_distord_matrix(image):
    pass