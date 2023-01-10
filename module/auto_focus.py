import cv2
import imutils

import numpy

import os
import time

# %% Initial values

frame_id_max = -1
val_max = -1
fm_list = [] # frame_max list

THRESHOLD = 350

# %% Auxiliary functions

def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()

def non_increasing(L):
    return all(L[i] >= L[i+1] for i in range(len(L)-1))

def API_auto_focus(crop_width, crop_height, image, frame_id):
    global val_max, frame_id_max
    
    x1 = image.shape[1]//2 - crop_width//2
    y1 = image.shape[0]//2 - crop_height//2
    x2 = image.shape[1]//2 + crop_width//2
    y2 = image.shape[0]//2 + crop_height//2

    crop_image = image[y1:y2, x1:x2]
    
    cv2.imshow('Cropped', crop_image)
    
    gray_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray_image)
    fm_list.append(fm)

    if fm > val_max:
        val_max = fm
        frame_id_max = frame_id
        
    if fm < THRESHOLD:
        move_flag = 1

    if non_increasing(fm_list[-3:]):
        move_flag = 0

    print(fm)

    return frame_id_max, move_flag

# %% Main function

if __name__ == "__main__":
    image = cv2.imread('D:/Working/KhoanCNC/Nam_work/experiment/frame/phase_frame32.jpg')
    API_auto_focus(300, 300, image, 32)
    cv2.waitKey(0)
    
    pass