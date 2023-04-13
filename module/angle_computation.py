import cv2
import imutils

import glob

import numpy as np
import math

import os
import time
#from utils import variance_of_laplacian, moving_average, non_increasing

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
    
    x1 = w//4
    y1 = h//4
    x2 = 3*w//4
    y2 = 3*h//4

    crop_image = image[y1:y2, x1:x2]
    return crop_image


def angle_computation_reset():
    global angle, error_id
    
    files = glob.glob('./storage/*.jpg')
    for f in files:
        os.remove(f)


def equation_plane(x1, y1, z1, x2, y2, z2, x3, y3, z3):
     
    a1 = x2 - x1
    b1 = y2 - y1
    c1 = z2 - z1
    a2 = x3 - x1
    b2 = y3 - y1
    c2 = z3 - z1
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    d = (- a * x1 - b * y1 - c * z1)
    return a, b, c, d
 
    
def distance(a1, b1, c1, a2, b2, c2):
     
    d = ( a1 * a2 + b1 * b2 + c1 * c2 )
    e1 = math.sqrt( a1 * a1 + b1 * b1 + c1 * c1)
    e2 = math.sqrt( a2 * a2 + b2 * b2 + c2 * c2)
    try:
        d = d / (e1 * e2)
    except:
        d = 0
    A = math.degrees(math.acos(d))
    return A


# %% API definition
    
def API_angle_computation(x11, y11, z11, x21, y21, z21, x31, y31, z31, x12, y12, z12, x22, y22, z22, x32, y32, z32):
    global angle, error_id
    
    error_id = 0
    
    a1, b1, c1, d1 = equation_plane(x11, y11, z11, x21, y21, z21, x31, y31, z31)
    a2, b2, c2, d2 = equation_plane(x12, y12, z12, x22, y22, z22, x32, y32, z32)
    
    angle = distance(a1, b1, c1, a2, b2, c2)
    
    return angle, error_id

# %% Main function

if __name__ == "__main__":
    angle, error_id = API_angle_computation(1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 1, 0, 0, 0, 1)
    print("Deg: ", angle)
    pass