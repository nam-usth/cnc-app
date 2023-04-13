import cv2
import imutils

import numpy as np

import os
import time

# %% Initial values

val_max = -1
search_flg = 0
sharpness = 0
sharpness_list = [] # frame_max list
peak_list = []

THRESHOLD = 275

stop_flg = 0
[difference_Y, difference_Z] = [100, 100]

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

def reset():
    val_max = -1
    search_flg = 0
    sharpness = 0
    sharpness_list = [] # frame_max list
    peak_list = []
    total_frame = 0

    return val_max, search_flg, sharpness, sharpness_list, peak_list, total_frame

# %% Image processing functions

def get_focus_zone(image, focus_width=400, focus_length=400):
    x1 = image.shape[1]//2 - focus_width//2
    y1 = image.shape[0]//2 - focus_length//2
    x2 = image.shape[1]//2 + focus_width//2
    y2 = image.shape[0]//2 + focus_length//2

    crop_image = image[y1:y2, x1:x2]
    return crop_image


def get_blade_hsv(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
       
    #lower = (85, 25, 50)
    #upper = (135, 240, 200)
    
    lower = (0, 25, 75)
    upper = (200, 240, 260)
    
    mask = cv2.inRange(image_hsv, lower, upper)
    
    #cv2.imshow('Mask (HSV)', mask)
    
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel1)

    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel2)

    # Make border for close contour at the image edge
    start_point = (0, 0) 
    end_point = (opening.shape[1], opening.shape[0])
    color = (0, 0, 0)
    thickness = 2

    opening = cv2.rectangle(opening, start_point, end_point, color, thickness)

    mask = opening
    
    blade_image = cv2.bitwise_and(image, image, mask=mask)
    blade_image[np.where((blade_image==[0,0,0]).all(axis=2))] = [255,255,255]

    #cv2.imshow('Blade (HSV)', blade_image)
    return mask, blade_image, image


def sharpen(image):    
    kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
    image_sharp = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)

    return image_sharp


def edge_detection(image):
    # Edge detection on mask 
    t_lower = 50                                     
    t_upper = 200                                    
    edge_image = cv2.Canny(image, t_lower, t_upper)
    
    # Show the edge
    #cv2.imshow('Edge detection', edge_image)
    
    return edge_image


def approx_contour(edge_image, blade_image):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated_img = cv2.dilate(edge_image, kernel)
    contours, hierarchy = cv2.findContours(dilated_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get the largest contour
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]
    
    # Bounding quadrangle
    hull = cv2.convexHull(cnt)
    
    # Approx
    approximations = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), closed = True)
    #cv2.drawContours(blade_image, [approximations], 0, (255, 0, 255), 2)
    
    approximations = approximations.flatten()
    
    return approximations

# %% API definition

# Legacy
def API_nose_phase_measurement_streaming(focus_width, focus_length, source):    
    global frame_id, val_max, search_flg, sharpness_list, total_frame
    
    error_id = 0
    
    frame_id, val_max, search_flg, sharpness, sharpness_list, peak_list, total_frame = reset()
    
    while True:
        success, image = source.read()   
        if not success:
            break
        else:
            if search_flg:
                break
                
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
                        
                if sharpness >= THRESHOLD:
                    mask, blade_image, original = get_blade_hsv(image)
                    
                    # Edge detection on mask 
                    edge_image = edge_detection(blade_image)
                        
                    # Find contours
                    approximations = approx_contour(edge_image, blade_image)
                    peak_list.append(approximations[2:4])
                    
                    # Draw the phase result
                    blade_image = cv2.circle(original, (approximations[2:4]), radius=1, color=(0, 255, 0), thickness=-1)
                
                if len(sharpness_list) < 15:
                    search_flg = 0
                else:
                    if non_increasing(sharpness_list[-10:]):
                        search_flg = 1
                    else:
                        search_flg = 0
                
                total_frame += 1
                
                # For DEBUGGING purpose ONLY
                #cv2.imwrite('Frame_' + str(total_frame) + '_phase_search.jpg', original)
                
                #print("Peak list: ", peak_list)
                        
            except:
                error_id = 1001

    return peak_list, error_id


# Flagging and checking condition
def nose_phase_measurement_stop_streaming():
    global stop_flg
    stop_flg = 1 


# Remastered
def API_nose_phase_measurement_slow_streaming(source, stop_flg):    
    global val_max, search_flg, sharpness_list, nFrame, total_frame
    global image_temp
    global peak_point
    peak_point = [-1, -1] # By default
    
    error_id = 0

    if stop_flg != 1:
        val_max, search_flg, sharpness, sharpness_list, peak_list, total_frame = reset()
    
    while not stop_flg:
        success, image = source.read()   
        
        if not success:
            break
        
        try:  
            center = [image.shape[1]//2, image.shape[0]//2]

            mask, blade_image, original = get_blade_hsv(image)
            
            # Edge detection on mask 
            edge_image = edge_detection(blade_image)
                
            # Find contours
            approximations = approx_contour(edge_image, blade_image)
            n = len(approximations)

            if approximations is not None:
                [difference_Y, difference_Z] = np.subtract(approximations[n-6:n-4], center) # Counter-clock wise position of Nose phase

                # For DEBUGGING purpose ONLY
                #print("Peak found in focus zone (Slow): ", approximations[n-6:n-4])
                
                horizontal_coord = approximations[n-6:n-4][0]
                
                peak_list.append(horizontal_coord)
                
                if (val_max < horizontal_coord):
                    val_max = horizontal_coord
                    nFrame = total_frame
                    peak_point = approximations[n-6:n-4]
                    
                    if np.linalg.norm([difference_Y, difference_Z]) <= 3:
                        [difference_Y, difference_Z] = [0, 0]
                        
                        search_flg = 1
                        
            total_frame += 1      
            
            # For DEBUGGING purpose ONLY
            #cv2.imwrite('./storage/Frame_' + str(total_frame) + '_phase_search.jpg', image)

            if peak_point == [-1, -1]:
                error_id = 1001                    
            
        except:
            error_id = 1001  

    try:
        return search_flg, nFrame, total_frame, peak_point[0], peak_point[1], error_id
    except:
        return search_flg, 0, 0, peak_point[0], peak_point[1], error_id


def API_nose_phase_measurement_fast_streaming(source, stop_flg):    
    global val_max, search_flg, sharpness_list, nFrame, total_frame
    global image_temp
    global difference_Y, difference_Z
    
    error_id = 0
    
    if stop_flg != 1:
        val_max, search_flg, sharpness, sharpness_list, peak_list, total_frame = reset()
    
    while not stop_flg:
        success, image = source.read()   
        
        if not success:
            break
        
        try:  
            crop_image = get_focus_zone(image)
            
            gray_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
            sharpness = variance_of_laplacian(gray_image)
            sharpness_list.append(sharpness)
            
            if (val_max < sharpness):
                val_max = sharpness
                nFrame = total_frame
                image_temp = image
            
            total_frame += 1
            
            # For DEBUGGING purpose ONLY
            #print("Difference of Nose Phase till the camera center in focus zone (Fast): ", [difference_Y, difference_Z])

            # For DEBUGGING purpose ONLY
            cv2.imwrite('./storage/Frame_' + str(total_frame) + '_phase_search.jpg', image)

        except:
            error_id = 1001

    try:
        print(sharpness_list)
        
        center = [image_temp.shape[1]//2, image_temp.shape[0]//2]
        
        mask, blade_image, original = get_blade_hsv(image_temp)
        
        # Edge detection on mask 
        edge_image = edge_detection(blade_image)
            
        # Find contours
        approximations = approx_contour(edge_image, blade_image)
        n = len(approximations)
        
        if approximations is not None:
            [difference_Y, difference_Z] = np.subtract(approximations[n-6:n-4], center) # Counter-clock wise position of Nose phase
            
        if np.linalg.norm([difference_Y, difference_Z]) <= 100:
            [difference_Y, difference_Z] = [0, 0]
        
            search_flg = 1

    except:
        error_id = 1001                

    try:
        return search_flg, nFrame, total_frame, difference_Y, difference_Z, error_id
    except:
        return search_flg, 0, 0, difference_Y, difference_Z, error_id

# %% Main function

if __name__ == "__main__":
    
    # For testing ONLY
    '''
    video_path = 'D:/Working/KhoanCNC/Nam_work/experiment/phase.m4v'
    
    source = cv2.VideoCapture(video_path)
    
    print(API_nose_phase_measurement_fast_streaming(source, 0))
    print(API_nose_phase_measurement_slow_streaming(source))

    source.release()
    '''

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    pass