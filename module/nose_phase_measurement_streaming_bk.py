import cv2
import imutils

import numpy as np

import operator
import os
import time

# %% Initial values

val_max = -1
search_flg = 0
sharpness = 0
sharpness_list = [] # frame_max list
peak_list = []
diff_list = []
mse_list = []

THRESHOLD = 70

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

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def index(lst, value):
    return operator.indexOf(lst, value)

def rindex(lst, value):
    return len(lst) - operator.indexOf(reversed(lst), value) - 1

def nose_phase_measurement_reset():
    global val_max, search_flg, sharpness, sharpness_list, peak_list, nFrame, total_frame, error_id
    global mse_list
    global prev_image
    
    val_max = -1
    search_flg = 0
    sharpness = 0
    sharpness_list = [] # frame_max list
    peak_list = [] 
    total_frame = 0
    mse_list = []
    
    error_id = 0
    prev_image = None
    
    
# %% Image processing functions

def get_focus_zone(image, focus_width=200, focus_length=200):
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
    
    lower = (0, 25, 40)
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


def remove_black_background(image):
    image_background = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
       
    lower = (0, 0, 0)
    upper = (180, 255, 30)
    
    mask = cv2.inRange(image_background, lower, upper)
    
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1)

    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2)

    # Make border for close contour at the image edge
    start_point = (0, 0) 
    end_point = (closing.shape[1], closing.shape[0])
    color = (255, 255, 255)
    thickness = 2

    closing = cv2.rectangle(closing, start_point, end_point, color, thickness)

    mask = closing
    mask = cv2.bitwise_not(mask)
    
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
    cv2.drawContours(blade_image, [approximations], 0, (255, 0, 255), 2)
    
    approximations = approximations.flatten()
    
    return approximations

# %% API definition

def API_nose_phase_measurement_final_streaming(image): # A.k.a super slow step
    global val_max, search_flg, sharpness_list, nFrame, total_frame, error_id
    global image_temp, prev_image
    global difference_Y, difference_Z, temp_Y, temp_Z
    global peak_list, peak_point
    
    move_flg = True
    search_flg = 1
    
    try:  
        image_clone = image.copy()
        
        crop_image = get_focus_zone(image)
        
        gray_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
        sharpness = variance_of_laplacian(gray_image)
        sharpness_list.append(sharpness)
        
        alpha = 0.5
            
        center = [crop_image.shape[1]//2, crop_image.shape[0]//2]
        
        mask_1, blade_image_1, original = get_blade_hsv(crop_image)
        mask_2, blade_image_2, original = remove_black_background(crop_image)
        
        blade_image = cv2.addWeighted(blade_image_1, alpha , blade_image_2, 1-alpha, 0)
        
        # Edge detection on mask 
        edge_image = edge_detection(blade_image)
            
        # Find contours
        approximations = approx_contour(edge_image, blade_image)

        # Find the closest point to center
        approximations = np.reshape(approximations, (-1, 2))
        dist = np.linalg.norm(np.subtract(approximations, center), axis=1)
        idx = np.unravel_index(np.argmin(dist), dist.shape)

        if approximations is not None:
            [difference_Y, difference_Z] = np.subtract(approximations[idx], center)

            # For DEBUGGING purpose ONLY
            #print("Peak found in focus zone (Slow): ", approximations[idx])
            
            # Draw the phase result
            blade_image = cv2.circle(original, (approximations[idx]), radius=1, color=(0, 255, 0), thickness=-1)
            
            cv2.putText(blade_image, "[{}, {}]".format(approximations[idx][0], approximations[idx][1]), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 6)
            
            horizontal_coord = approximations[idx][0]
            
            peak_list.append(horizontal_coord)
            
            '''
            if non_increasing(peak_list[-2:]):
                move_flg = 0
            '''
            if len(peak_list) > 1:
                if peak_list[len(peak_list) - 1] < peak_list[len(peak_list) - 2]:
                    move_flg = 0
            
            '''
            if np.linalg.norm([difference_Y, difference_Z]) <= 20: # 10
                temp_Y, temp_Z = 0, 0
                
            else:
                temp_Y, temp_Z = difference_Y, difference_Z
            '''
        
        # Result for demo ONLY 
        temp_Y, temp_Z = 0, 0 
            
        total_frame += 1
        
        # For DEBUGGING purpose ONLY
        cv2.imwrite('./storage/SUPER_SLOW_Frame_' + str(total_frame) + '_phase_search.jpg', image_clone)
        cv2.imwrite('./storage/point/SUPER_SLOW_Frame_' + str(total_frame) + '_phase_search_point.jpg', blade_image)
    
    except:
        pass
        
    return search_flg, temp_Y, temp_Z, move_flg, error_id


def API_nose_phase_measurement_slow_streaming(image):    
    global val_max, search_flg, sharpness_list, nFrame, total_frame, error_id
    global image_temp, prev_image
    global difference_Y, difference_Z, temp_Y, temp_Z
    global peak_list, peak_point

    try:  
        
        h, w = image.shape[0], image.shape[1]
        
        if prev_image is None:
            prev_image = image
            
        mse_list.append(mse(image[0:h,0:w//2], prev_image[0:h,0:w//2]))
        
        #mse_array = np.asarray(mse_list).reshape(-1, 1)

        '''
        kmeans = KMeans(n_clusters=2).fit(mse_array)
        M = kmeans.predict(mse_array)
        D = np.abs(np.diff(M))
        
        start_dup = index(D, 1) - 1
        end_dup = rindex(D, 1) + 1
        '''
        
        start_dup = len(mse_list)-1
        
        for i in range(3, len(mse_list)):
            mean = 0.1
            if i == 3:
                mean = mse_list[2]
            else:
                mean = (mse_list[i-1] + mse_list[i-2])/2
            if (mse_list[i] > mean*1.35) or (mse_list[i] < mean*0.65):
                start_dup = i
                break
            
        end_dup = 0
        
        for i in range(len(mse_list)-3, 0, -1):
            mean = (mse_list[i+1] + mse_list[i+2])/2
            if (mse_list[i] > mean*1.35) or (mse_list[i] < mean*0.65):
                end_dup = i
                break
        
        #print('Start Dup: ', start_dup, ' - End Dup: ', end_dup)
        
        image_clone = image.copy()
        
        crop_image = get_focus_zone(image)
        
        gray_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
        sharpness = variance_of_laplacian(gray_image)
        sharpness_list.append(sharpness)
        
        if sharpness >= THRESHOLD:
            alpha = 0.5
            
            center = [crop_image.shape[1]//2, crop_image.shape[0]//2]
            
            mask_1, blade_image_1, original = get_blade_hsv(crop_image)
            mask_2, blade_image_2, original = remove_black_background(crop_image)
            
            blade_image = cv2.addWeighted(blade_image_1, alpha , blade_image_2, 1-alpha, 0)
            
            # Edge detection on mask 
            edge_image = edge_detection(blade_image)
                
            # Find contours
            approximations = approx_contour(edge_image, blade_image)

            # Find the closest point to center
            approximations = np.reshape(approximations, (-1, 2))
            dist = np.linalg.norm(np.subtract(approximations, center), axis=1)
            idx = np.unravel_index(np.argmin(dist), dist.shape)

            if approximations is not None:
                [difference_Y, difference_Z] = np.subtract(approximations[idx], center)

                # For DEBUGGING purpose ONLY
                #print("Peak found in focus zone (Slow): ", approximations[idx])
                
                # Draw the phase result
                blade_image = cv2.circle(original, (approximations[idx]), radius=1, color=(0, 255, 0), thickness=-1)
                
                cv2.putText(blade_image, "[{}, {}]".format(approximations[idx][0], approximations[idx][1]), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 6)
                
                horizontal_coord = approximations[idx][0]
                
                peak_list.append(horizontal_coord)
                
                if (val_max < horizontal_coord):
                    val_max = horizontal_coord
                    nFrame = total_frame
                    peak_point = approximations[idx]
                    
                    '''
                    if np.linalg.norm([difference_Y, difference_Z]) <= 20: # 10
                        temp_Y, temp_Z = 0, 0
                        search_flg = 1
                    else:
                        temp_Y, temp_Z = difference_Y, difference_Z
                    '''
                    
        total_frame += 1
        
        prev_image = image
        
        # For DEBUGGING purpose ONLY
        cv2.imwrite('./storage/SLOW_Frame_' + str(total_frame) + '_phase_search.jpg', image_clone)
        cv2.imwrite('./storage/point/SLOW_Frame_' + str(total_frame) + '_phase_search_point.jpg', blade_image)
    
    except:
        pass
    
    # Result for demo ONLY 
    temp_Y, temp_Z = 0, 0
    search_flg = 1

    try:
        frame_idx = nFrame - (start_dup - 1)
        if frame_idx < 0:
            frame_idx = 0
            
        if len(peak_list) == 0:
            error_id = 1004
        
        #print('Start Dup: ', start_dup, ' - End Dup: ', end_dup)
        #print('nFrame: ', frame_idx)
        
        return search_flg, frame_idx, (end_dup + 1) - start_dup, temp_Y, temp_Z, error_id
    except:
        return search_flg, 0, 0, 0, 0, 1009


def API_nose_phase_measurement_fast_streaming(image):    
    global val_max, search_flg, sharpness_list, nFrame, total_frame, error_id
    global mse_list
    global image_temp, prev_image
    global difference_Y, difference_Z
                
    try:
        if prev_image is None:
            prev_image = image
        
        mse_list.append(mse(image, prev_image))
        
        crop_image = get_focus_zone(image)
        
        gray_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
        sharpness = variance_of_laplacian(gray_image)
        sharpness_list.append(sharpness)
        
        if (val_max < sharpness):
            val_max = sharpness
            nFrame = total_frame
            image_temp = image
        
        total_frame += 1
        
        prev_image = image
        
        # For DEBUGGING purpose ONLY
        #print("Difference of Nose Phase till the camera center in focus zone (Fast): ", [difference_Y, difference_Z])

        # For DEBUGGING purpose ONLY
        cv2.imwrite('./storage/FAST_Frame_' + str(total_frame) + '_phase_search.jpg', image)

    except:
        error_id = 1001

    try:
        #mse_array = np.asarray(mse_list).reshape(-1, 1)
        
        '''
        kmeans = KMeans(n_clusters=2).fit(mse_array)
        M = kmeans.predict(mse_array)
        D = np.abs(np.diff(M))
        
        start_dup = index(D, 1) - 1
        end_dup = rindex(D, 1) + 1
        '''
        
        start_dup = len(mse_list)-1
        
        for i in range(3, len(mse_list)):
            mean = 0.1
            if i == 3:
                mean = mse_list[2]
            else:
                mean = (mse_list[i-1] + mse_list[i-2])/2
            if (mse_list[i] > mean*1.8) or (mse_list[i] < mean*0.4):
                start_dup = i
                break
            
        end_dup = 0
        
        for i in range(len(mse_list)-3, 0, -1):
            mean = (mse_list[i+1] + mse_list[i+2])/2
            if (mse_list[i] > mean*1.8) or (mse_list[i] < mean*0.4):
                end_dup = i
                break
            
        alpha = 0.5
        
        center = [image_temp.shape[1]//2, image_temp.shape[0]//2]
        
        mask_1, blade_image_1, original = get_blade_hsv(image_temp)
        mask_2, blade_image_2, original = remove_black_background(image_temp)
        
        blade_image = cv2.addWeighted(blade_image_1, alpha , blade_image_2, 1-alpha, 0)
        
        # Edge detection on mask 
        edge_image = edge_detection(blade_image)
            
        # Find contours
        approximations = approx_contour(edge_image, blade_image)
        
        # Find the closest point to center
        approximations = np.reshape(approximations, (-1, 2))
        dist = np.linalg.norm(np.subtract(approximations, center), axis=1)
        idx = np.unravel_index(np.argmin(dist), dist.shape)

        if approximations is not None:
            [difference_Y, difference_Z] = np.subtract(approximations[idx], center)
        
        '''
        if np.linalg.norm([difference_Y, difference_Z]) <= 100:
            [difference_Y, difference_Z] = [0, 0]
        
            search_flg = 1
        '''

        # Result for demo ONLY        
        [difference_Y, difference_Z] = [0, 0]

        search_flg = 1
        
    except:
        error_id = 1001                

    try:
        frame_idx = nFrame - (start_dup - 1)
        if frame_idx < 0:
            frame_idx = 0
        
        return search_flg, frame_idx, (end_dup + 1) - start_dup, difference_Y, difference_Z, error_id
    except:
        return search_flg, 0, 0, difference_Y, difference_Z, error_id
    

# %% Main function

if __name__ == "__main__":
    
    # For testing ONLY
    '''
    video_path = 'F:/phase-slow-search-test.mp4'
    
    source = cv2.VideoCapture(video_path)
    
    #print(API_nose_phase_measurement_fast_streaming(source, 0))
    print(API_nose_phase_measurement_slow_streaming_debug(source))

    source.release()
    '''
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    pass