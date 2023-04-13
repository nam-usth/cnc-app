import cv2
import imutils

import numpy

import os
import time

# %% Initial values

status_flag = 0

# %% Auxiliary functions

def get_corner(image):
    pass

def compute_distord_matrix(image):
    pass

def reset():    
    status_flag = 0
    
# %% API definition

def API_calibration(focus_width, focus_length, image):
    global status_flag

    error_id = 0
    
    try:
        #if compute_distord_matrix(image):
        status_flag = 1
    except:
        error_id = 1001
    
    return bool(status_flag), error_id

# %% Main function

if __name__ == "__main__":
    
    # For testing ONLY
    '''
    image = cv2.imread('D:/Working/KhoanCNC/Nam_work/experiment/frame/phase_frame32.jpg')
    API_calibration(300, 300, image, 32)
    '''
        
    pass