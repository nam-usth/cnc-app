from tools.emit_log import emit_camera_status
from tools.Static import STATIC
from tools.common import get_frame, check_camera_status
from module.auto_focus import API_auto_focus, auto_focus_reset
import time
from module.auto_focus_streaming import API_auto_focus_streaming, auto_focus_streaming_reset, add_mse_sharpness, clearn_duplicate, find_max_sharpness
from module.object_detection import API_object_detection
from module.nose_phase_measurement_streaming import API_nose_phase_measurement_fast_streaming, API_nose_phase_measurement_slow_streaming, \
                API_nose_phase_measurement_final_streaming, nose_phase_measurement_reset
from module.helix_angle_measurement import API_helix_angle_measurement, helix_angle_measurement_reset
from module.axial_angle_measurement import API_axial_angle_measurement, axial_angle_measurement_reset
from module.pocket_angle_measurement import API_pocket_angle_measurement, pocket_angle_measurement_reset
import numpy as np

# AUTO FOCUS
# Updated on 16/3/2023
# -- Auto Focus with Image Capturing (not Streaming)
def thread_auto_focus_capture(socketio, cam_id, focus_width, focus_length):
    if not emit_camera_status(STATIC.session, socketio, cam_id):
        return
    img = get_frame(STATIC.session)
    if img is None:
        socketio.emit("auto_focus", {
            "sharpness" : 0,
            "searchFlg" : 0,
            "imageOutputID" : 0,
            "totalFrame": 0,
            "errorID" : 1004,
        })
        return
    API_auto_focus_output = API_auto_focus(img, focus_width, focus_length)
        
    socketio.emit('auto_focus',
    {
        "sharpness" : API_auto_focus_output[0],
        "searchFlg" : API_auto_focus_output[1],
        "imageOutputID" : API_auto_focus_output[2],
        "totalFrame": API_auto_focus_output[3],
        "errorID" : API_auto_focus_output[4],
    })

def thread_auto_focus(socketio, cam_id, focus_width, focus_length):
    if not emit_camera_status(STATIC.session, socketio, cam_id):
        return
    STATIC.stop_flg = 0
    auto_focus_streaming_reset()
    socketio.emit('auto_focus_streaming_start', {'readyFlg': 1})
    while not STATIC.stop_flg:
        current_time = time.time()
        img = get_frame(STATIC.session)
        if img is None:
            time.sleep(0.01)
            continue
        if STATIC.last_time is None:
            STATIC.last_time = current_time
        duration = current_time - STATIC.last_time
        add_mse_sharpness(img, focus_width, focus_length)
        STATIC.last_time = current_time

# OBJECT DETECTION
# Updated on 16/3/2023
# -- Object Detection with Image Capturing
def thread_object_detection(socketio, cam_id):
    img = get_frame(STATIC.session)
    if img is None:
        if not check_camera_status(STATIC.session, cam_id):
            socketio.emit("object_detection", {"errorID" : 1004})
        else:
            socketio.emit("object_detection", {"errorID" : 0, "detectFlg" : 0})
        return
    API_object_detection_output = API_object_detection(img)
    
    socketio.emit('object_detection',     
    {
        "detectFlg" : API_object_detection_output[0],
        "errorID" : API_object_detection_output[1],
    })
    

# NOSE PHASE MEASUREMENT
# [FAST]
def thread_nose_phase_measurement_fast(socketio, cam_id):
    # global session
    if not emit_camera_status(STATIC.session, socketio, cam_id):
        return
    
    # global stop_flg
    STATIC.stop_flg = 0
    
    nose_phase_measurement_reset()
    
    socketio.emit('nose_phase_measurement_fast_streaming_start', {'readyFlg': 1})
    
    while not STATIC.stop_flg:
        img = get_frame(STATIC.session)
        if img is None:
            time.sleep(0.01)
            continue
        STATIC.API_nose_phase_measurement_fast_streaming_output = API_nose_phase_measurement_fast_streaming(img)
        time.sleep(0.04)

# [SLOW]
def thread_nose_phase_measurement_slow(socketio, cam_id):
    if not emit_camera_status(STATIC.session, socketio, cam_id):
        return
    STATIC.stop_flg = 0
    nose_phase_measurement_reset()
    socketio.emit('nose_phase_measurement_slow_streaming_start', {'readyFlg': 1})
    while not STATIC.stop_flg:
        img = get_frame(STATIC.session)
        if img is None:
            time.sleep(0.01)
            continue
        STATIC.API_nose_phase_measurement_slow_streaming_output = API_nose_phase_measurement_slow_streaming(img)
        time.sleep(0.04)
        
# [FINAL - SUPER SLOW]
def thread_nose_phase_measurement_final(socketio, cam_id):
    if not emit_camera_status(STATIC.session, socketio, cam_id):
        return
    
    img = get_frame(STATIC.session)
    if img is None:
        socketio.emit('nose_phase_measurement_final_streaming',     
        {
            "searchFlg" : 0,
            "differenceY" : 0,
            "differenceZ" : 0,
            "moveFlg" : 0,
            "errorID" : 1004,
        })
        return 
        
    
    API_nose_phase_measurement_final_streaming_output = API_nose_phase_measurement_final_streaming(img)
        
    socketio.emit('nose_phase_measurement_final_streaming',     
    {
        "searchFlg" : API_nose_phase_measurement_final_streaming_output[0],
        "differenceY" : 0,
        "differenceZ" : 0,
        "moveFlg" : int(API_nose_phase_measurement_final_streaming_output[3]),
        "errorID" : API_nose_phase_measurement_final_streaming_output[4],
    })
    
# HELIX ANGLE
# Updated on 27/3/2023
# -- Helix Angle measurement with Image Capturing (not Streaming)    
def thread_helix_angle_measurement(socketio, cam_id):
    if not emit_camera_status(STATIC.session, socketio, cam_id):
        return
    
    img = get_frame(STATIC.session)
    if img is None:
        socketio.emit('helix_angle_measurement',     
        {
            "helixAngle" : 0,
            "moveFocusFlag": 0,
            "moveFindHelix": 0,
            "errorID" : 1004,
        })
        return 
    
    API_helix_angle_measurement_output = API_helix_angle_measurement(img)
        
    socketio.emit('helix_angle_measurement',     
    {
        "helixAngle" : API_helix_angle_measurement_output[0] * 180/np.pi,
        "moveFocusFlag": API_helix_angle_measurement_output[1],
        "moveFindHelix": API_helix_angle_measurement_output[2],    
        "errorID" : API_helix_angle_measurement_output[3],
    })
    
# RELIEF/EXIT ANGLE 
# Updated on 30/3/2023
def thread_exit_angle_measurement_step1(socketio, cam_id, focus_width, focus_length):
    socketio.emit('thread_exit_angle_measurement_step1')

def thread_exit_angle_measurement_step2(socketio, cam_id, focus_width, focus_length):
    socketio.emit('thread_exit_angle_measurement_step2')

def thread_exit_angle_measurement_step3(socketio, cam_id, focus_width, focus_length):
    socketio.emit('thread_exit_angle_measurement_step3')
    
# GASH + POCKET ANGLE
# Updated on 31/3/2023
# -- Gash + Pocket Angle measurement with Image Capturing (not Streaming)
# -- Gash is Axial and vice versa
def thread_axial_angle_measurement(socketio, cam_id, focus_width, focus_length):
    if not emit_camera_status(STATIC.session, socketio, cam_id):
        return
    
    img = get_frame(STATIC.session)
    if img is None:
        socketio.emit('axial_angle_measurement',     
        {
            "axialAngle" : 0,
            "moveFocusFlg": 0,
            "errorID" : 1004,
        })
        return 
    
    API_axial_angle_measurement_output = API_axial_angle_measurement(img)
        
    socketio.emit('axial_angle_measurement',     
    {
        "axialAngle" : API_axial_angle_measurement_output[0] * 180/np.pi,
        "moveFocusFlg" : API_axial_angle_measurement_output[1],
        "errorID" : API_axial_angle_measurement_output[2],
    })
    
def thread_pocket_angle_measurement(socketio, cam_id, focus_width, focus_length):
    if not emit_camera_status(STATIC.session, socketio, cam_id):
        return
    
    img = get_frame(STATIC.session)
    if img is None:
        socketio.emit('pocket_angle_measurement',     
        {
            "pocketAngle" : 0,
            "moveFocusFlg": 0,
            "errorID" : 1004,
        })
        return 
    
    API_pocket_angle_measurement_output = API_pocket_angle_measurement(img)
        
    socketio.emit('pocket_angle_measurement',     
    {
        "pocketAngle" : API_pocket_angle_measurement_output[0] * 180/np.pi,
        "moveFocusFlg": API_pocket_angle_measurement_output[1],
        "errorID" : API_pocket_angle_measurement_output[2],
    })
