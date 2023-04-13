import cv2
import imutils

import glob

import numpy as np

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

def auto_focus_streaming_reset():
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

def crop_img(image, focus_width, focus_length):
    x1 = image.shape[1]//2 - focus_width//2
    y1 = image.shape[0]//2 - focus_length//2
    x2 = image.shape[1]//2 + focus_width//2
    y2 = image.shape[0]//2 + focus_length//2

    crop_image = image[y1:y2, x1:x2]
    return crop_image

def calculate_sharpness(crop_image):
    gray_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
    sharpness = variance_of_laplacian(gray_image)
    return sharpness

def add_mse_sharpness(image, focus_width, focus_length):
    global mse_list
    global prev_image
    image_org = image.copy()
    
    image = crop_img(image, focus_width, focus_length)

    if prev_image is None:
        mse_value = -1
    else:
        mse_value = mse(prev_image, image_org)
    prev_image = image_org

    mse_list.append({'mse': mse_value, 'img' : image_org, 'sh' : calculate_sharpness(image), 'time' : time.time()})
    
def clearn_duplicate():
    global mse_list
    index_first = -1
    index_last = -1
    time_crop = -1
    for index, item in enumerate(mse_list):
        print("mse : ", item["mse"])
        time_draw = 0 
        if time_crop == -1:
            time_draw = 0
            time_crop = item["time"]
        else:
            time_draw = item["time"] - time_crop
        cv2.imwrite(f"storage/{index}_{int(time_draw * 1000)}.jpg", item["img"])
        
    for index, item in enumerate(mse_list):
        if item["mse"] == -1:
            continue
        if index + 1 == len(mse_list):
            break
        item_next = mse_list[index + 1]
        ratio = item_next["mse"] / item["mse"]
        print("ratio 1 : ", ratio)
        if ratio > 1.25 or ratio < 0.75:
            index_first = index
            break
    if index_first == -1:
        # error ko thay doi
        return False
    for index in range(len(mse_list), 0, -1):
        index -= 1
        if index - 1 < 0:
            break
        item = mse_list[index]
        item_pre = mse_list[index - 1]
        ratio = item_pre["mse"] / item["mse"]
        print("ratio 2 : ", ratio)
        if ratio > 1.25 or ratio < 0.75:
            index_last = index - 1
            break
    print("index_first : ", index_first)
    print("index_last : ", index_last)
    mse_list = mse_list[index_first:index_last]
    return True

def find_max_sharpness():
    global mse_list
    index_max = -1
    max_sharpness = -1
    print("mse_list : ", len(mse_list))
    for index, item in enumerate(mse_list):
        print(index, item["sh"])
        if item["sh"] > max_sharpness:
            max_sharpness = item["sh"]
            index_max = index
    print("index_max :", index_max)
    if len(mse_list) > 0:
        time_frame_max = mse_list[index_max]['time'] - mse_list[0]['time']
        time_total = mse_list[-1]['time'] - mse_list[0]['time']
        return int(time_frame_max * 1000), int(time_total * 1000)
    return -1, -1
# %% API definition
    
def API_auto_focus_streaming(focus_width, focus_length, image):
    global image_output_id, val_max, search_flag, sharpness_list, total_frame, error_id
    global mse_list, sharpness
    global prev_image
    
    '''
    if search_flag:
        if val_max < THRESHOLD:
            search_flag = 0
        else:
            pass
    '''
    '''
    try:      
        if prev_image is None:
            prev_image = image
            
        mse_list.append(mse(image, prev_image))
            
        start_dup = len(mse_list)-1
        
        
        for i in range(3, len(mse_list)):
            mean = 0.1
            if i == 3:
                mean = mse_list[2]
            else:
                mean = (mse_list[i-1] + mse_list[i-2])/2
            if (mse_list[i] > mean*1.15) or (mse_list[i] < mean*0.85):
                start_dup = i
                break

        print(start_dup, total_frame)
        
        
        x1 = image.shape[1]//2 - focus_width//2
        y1 = image.shape[0]//2 - focus_length//2
        x2 = image.shape[1]//2 + focus_width//2
        y2 = image.shape[0]//2 + focus_length//2
    
        crop_image = image[y1:y2, x1:x2]
        
        gray_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
        sharpness = variance_of_laplacian(gray_image)
        
        #if (total_frame != start_dup):
        
        sharpness_list.append(sharpness)

        if sharpness > val_max:
            val_max = sharpness
            
        image_output_id = sharpness_list.index(val_max)
                
        if len(sharpness_list) < 15:
            search_flag = 0
        else:
            if non_increasing(sharpness_list[-3:]):
                search_flag = 1
            else:
                search_flag = 0
        
        search_flag = 1
        
        
        total_frame += 1
        
        # For DEBUGGING purpose ONLY
        cv2.imwrite('./storage/AUTO-FOCUS_Frame_' + str(total_frame) + '.jpg', image)
        
        prev_image = image
        
        #print("List: ", sharpness_list, "\nSharp image found: ", bool(search_flag), "\nThe sharpest image ID: ", image_output_id, "\nTotal frames: ", total_frame)
            
    except:
        error_id = 1001

    return sharpness, bool(search_flag), image_output_id, total_frame, error_id, sharpness_list
'''
# sharpness, status. None, False, 150, True

# %% Main function
'''
global mse_list
global prev_image
mse_list = []
prev_image = None
'''

if __name__ == "__main__":
    '''
    for i in range(14):
        
        path_1 = f"images/AUTO-FOCUS_Frame_{i + 1}.jpg"
        #path_2 = f"images/AUTO-FOCUS_Frame_{i + 2}.jpg"
        
        img_1 = cv2.imread(path_1)
        
        check_duplicate_sharpness(img_1, 100, 100)
        #img_2 = cv2.imread(path_2)
        
        #if img_1 is None or img_2 is None:
        #    continue
        #img_1 = crop(img_1, 100, 100)
        #img_2 = crop(img_2, 100, 100)
        #m_1_2 = mse(img_1, img_2)
        #print(f"{i + 1} vs {i + 2} : {m_1_2}")
    print("list mse : ", mse_list)
    new_mse_list = clearn_duplicate()
    print("\n")
    print("new_mse_list : ", new_mse_list)
    '''
    # For testing ONLY
    '''
    source = cv2.VideoCapture('D:/Working/KhoanCNC/Nam_work/experiment/phase.m4v')
    API_auto_focus_streaming(300, 300, source)
    '''
        
    pass