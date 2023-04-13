import cv2
import numpy as np
from tools.Static import STATIC
from tools.Static import *

def get_position(im_bw, postion, direction):
    pxes = im_bw[:,postion]
    mask = np.where(pxes == 255)
    if not len(mask[0]):
        return -1
    if direction == 0:
        return mask[0][0]
    return mask[0][-1]

def get_edge_points(img, direction, px2mm):
    list_points = []
    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (_, im_bw) = cv2.threshold(im_gray, 70, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    x_center = img.shape[1] // 2
    for index in range(STATIC.num_step):
        px = int(index * STATIC.step / px2mm)
        if index > 0 and (index * STATIC.step / px2mm) % int(index * STATIC.step / px2mm) > 0.5:
            px += 1
        dis = get_position(im_bw, x_center + px, direction)
        if dis == -1:
            continue
        y = img.shape[0] // 2 - dis + STATIC.space
        list_points.append([index * STATIC.step, y])
    distance = get_position(im_bw, x_center, direction)
    
    po = distance
    
    distance = img.shape[0] // 2 - distance
    STATIC.space += distance
    if abs(distance * px2mm) < STATIC.thr:
        cv2.circle(img, (img.shape[1] // 2, img.shape[0] // 2), 4, (0,0,255), 5)
        cv2.circle(img, (img.shape[1] // 2, po), 4, (0,255,0), 5)
        cv2.line(img, (img.shape[1] // 2, img.shape[0] // 2), (img.shape[1] // 2, po), (255, 0, 0), 4) 
        cv2.imwrite(f"storage/{STATIC.index}.jpg", img)
        STATIC.index += 1
        return 0, list_points
    return distance * px2mm, list_points

def get_step():
    step = []
    for _ in range(STATIC.num_step):
        if not len(STATIC.list_points):
            break
        point = STATIC.list_points[0]
        step.append(point)
        del STATIC.list_points[0]
    return step

def smoot_points():
    last_points = []
    response_points = []
    while True:
        points = get_step()
        points = np.array(points)
        if not len(points):
            # push to response list
            [response_points.append([x, y]) for (x, y) in last_points]
            break
        if len(last_points):
            el = (points[:, 1][0] - last_points[:, 1][-1]) / len(last_points)
            for index in range(1, len(last_points), 1):
                last_points[:,1][index] += el * index
        # push to response list
        [response_points.append([x, y]) for (x, y) in last_points]
        last_points = points
    response_points = np.array(response_points)
    response_points = np.round(response_points, 3)
    return response_points.tolist()
    
 
def calculator_edge(data, img, direction):
    cv2.imwrite(f"./storage/edge_{STATIC.index}.jpg", img)
    STATIC.index += 1
    stopFlg = data["stopFlg"]
    positionXY = data["positionXY"]
    positionZ = data["positionZ"]
    px2mm = data["px2mm"]
    print("px2mm : ", px2mm)
    distance, list_points = get_edge_points(img, direction, px2mm)
    points = np.array(list_points)
    
    if len(points):
        points[:,0] = points[:,0] + positionZ
        points[:,1] = points[:,1] * px2mm + positionXY
        points = np.round(points, 3)
        
    if not distance:
        [STATIC.list_points.append(point.tolist()) for point in points]

    l_points = []
    if stopFlg == STOP_EDGE_DETECTION and not distance:
        STATIC.space = 0
        l_points = smoot_points()
        STATIC.list_last_points = []
        STATIC.list_points = []
        
    response = {
        'distance' : distance,
        'moveFocusFlg' : 0,
        'edgePositionArr' : l_points,
        'errorID' : 0
    }
    return response