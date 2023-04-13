import cv2
import imutils

import glob

import numpy as np

import os
import time
import string, random

# %% Initial values

detect_flag = 0
index_frame = 0

# %% API definition

def API_object_detection(image):
    global detect_flag
    global index_frame
    
    error_id = 0
    name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
    index_frame += 1

    try:
        h, w, c = image.shape
        threshold = w*h / 20 # One-twentieth image
    
        roi = image
        gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),0)
        _, binary = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
        if len(contours) == 0:
            detect_flag = 0
        else:
            cnt = max(contours, key=lambda x: cv2.contourArea(x))
            
            if cv2.contourArea(cnt) > threshold:
                detect_flag = 1
            else:
                detect_flag = 0

    except:
        error_id = 1004 # No image
    cv2.imwrite(f"storage1/{name}_{index_frame}_{detect_flag}.jpg", image)
    return detect_flag, error_id


# %% Main function

if __name__ == "__main__":
    # For testing ONLY
    '''
    folder_path = "D:/Working/KhoanCNC/Nam_work/experiment/"
    image_name = "SLOW_Frame_18_phase_search.jpg"
    img_path = os.path.join(folder_path, image_name)

    img = cv2.imread(img_path)

    image_name_out = image_name[:-4] + "_out.jpg"
    img_path_out = os.path.join(folder_path, "out", image_name_out)

    h, w, c = image.shape

    threshold = w*h / 8

    roi = img
    # roi = img[int(h/3):int(2*h/3),int(w/3):int(2*w/3)]
    gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    _, binary = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=lambda x: cv2.contourArea(x))

    if cv2.contourArea(cnt) > threshold:
        cv2.putText(img, "Object found", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 6)
    else:
        cv2.putText(img, "Ojbect NOT found", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 6)

    cv2.imwrite(img_path_out, img)
    '''
        
    pass