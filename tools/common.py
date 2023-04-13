import os
import numpy as np
import cv2
import time

def check_camera_status(session, cam_id):
    try:
        res = session.get(f"http://localhost:3000/camera_status/{cam_id}")
        if res.status_code == 200:
            data = res.json()
            if data["status"] == 1:
                return True
    except:
        pass
    return False

def change_resolution(session, width, height):
    try:
        res = session.get(f"http://localhost:3000/change_resolution_api/{width}/{height}")
        if res.status_code == 200:
            data = res.json()
            if data['status'] == 0:
                return True
    except:
        pass
    return False

def decode_img(bytes_):
    if len(bytes_) == 0:
        return None
    np_array = np.frombuffer(bytes_, np.uint8)     
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return img

def get_frame(session):
    time_wait = 5
    while time_wait > 0:
        try: 
            res = session.get("http://localhost:3000/frame")
            if res.status_code == 200:
                img = res.content
                img = decode_img(img)
                if img is not None:
                    return img
        except:
            pass
        time_wait -= 1
        time.sleep(0.01)
    return None
                
