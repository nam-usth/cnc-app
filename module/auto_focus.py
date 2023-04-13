import cv2
import imutils

import glob

import numpy as np

import os
import time

# %% Initial values

image_output_id = -1
val_max = -1
search_flag = 0
sharpness = 0
sharpness_list = [] # frame_max list

THRESHOLD = 350

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

def auto_focus_reset():
    global image_output_id, val_max, search_flag, sharpness, sharpness_list, total_frame, error_id
    global mse_list
    global prev_image
   
    image_output_id = -1
    val_max = -1
    search_flag = 0
    sharpness = 0
    sharpness_list = [] # frame_max list
    total_frame = 0
    mse_list = []
   
    error_id = 0
    prev_image = None
   
    files = glob.glob('./storage/*.jpg')
    for f in files:
        os.remove(f)

# %% API definition

def API_auto_focus(focus_width, focus_length, image):
    global image_output_id, val_max, search_flag, sharpness, sharpness_list, total_frame, error_id
    
    try:            
        x1 = image.shape[1]//2 - focus_width//2
        y1 = image.shape[0]//2 - focus_length//2
        x2 = image.shape[1]//2 + focus_width//2
        y2 = image.shape[0]//2 + focus_length//2
    
        crop_image = image[y1:y2, x1:x2]
        
        gray_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
        sharpness = variance_of_laplacian(gray_image)
        sharpness_list.append(sharpness)

        if sharpness > val_max:
            val_max = sharpness
            
        image_output_id = sharpness_list.index(val_max)
                
        search_flag = 1

        total_frame += 1
        
         # For DEBUGGING purpose ONLY
        cv2.imwrite('./storage/AUTO-FOCUS_CAPTURE_Frame_' + str(total_frame) + '.jpg', image)
       
        #print("List: ", sharpness_list, "\nSharp image found: ", bool(search_flag), "\nThe sharpest image ID: ", image_output_id, "\nTotal frames: ", total_frame)

    except:
        error_id = 1001

        if search_flag:
            error_id = 0

    #print("List: ", sharpness_list, "\nSharp image found: ", bool(search_flag), "\nThe sharpest image ID: ", image_output_id)
    
    return sharpness, bool(search_flag), image_output_id, total_frame, error_id

# %% Main function

if __name__ == "__main__":
    
    # For testing ONLY
    '''
    image = cv2.imread('D:/Working/KhoanCNC/Nam_work/experiment/frame/phase_frame32.jpg')
    API_auto_focus(300, 300, image, 32)
    '''
        
    pass