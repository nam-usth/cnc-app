import cv2
import os
import numpy as np
import skimage.exposure
from module.auto_focus import variance_of_laplacian

LEFT = "left"
RIGHT = "right"

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

def API_get_distance(img, width, height, thr_sharpness=700, direction="left"):
    kernel = np.ones((2,2),np.uint8)

    width_img = img.shape[1]
    height_img = img.shape[0]

    x1 = int((width_img - width) / 2)
    y1 = int((height_img - height) / 2)

    img = img[y1: y1 + height, x1: x1 + width]

    score_sharpness = variance_of_laplacian(img, (200, 200))

    if score_sharpness < thr_sharpness:
        return -2

    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    (_, im_bw) = cv2.threshold(im_gray, 40, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    opening = cv2.morphologyEx(im_bw, cv2.MORPH_OPEN, kernel)

    opening = smoothImage(opening)

    border = find_border(opening)

    border = cv2.cvtColor(border, cv2.COLOR_BGR2GRAY)

    img_vis = np.zeros_like(img)

    cv2.line(img_vis, (0, int(height / 2)), (width, int(height / 2)), (255, 255, 255), 2) 

    img_vis = cv2.cvtColor(img_vis, cv2.COLOR_BGR2GRAY)

    img_bwa = cv2.bitwise_and(img_vis, border)

    mask = img_bwa == 255

    data = np.where(mask == True)[1]

    points = []

    for item in data:
        if distant(points, item, 10):
            points.append(item - 2)
    min_distance = 9999
    for x in points:
        if direction == LEFT:
            if min_distance > x:
                min_distance = x
        else:
            if min_distance > (width - x):
                min_distance = (width - x)
    
    if min_distance == 9999:
        return -1
    
    return min_distance - int(width / 2)

