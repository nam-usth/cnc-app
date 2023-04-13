from engineio.async_drivers import gevent
from flask import Flask, request, jsonify, Response
from flask_socketio import SocketIO
import pandas as pd
import numpy as np
import os
import socketio
import json
import ast
from tools.emit_log import emit_camera_status
from module.auto_focus import auto_focus_reset
from module.auto_focus_streaming import clearn_duplicate, find_max_sharpness
from module.nose_phase_measurement_streaming import nose_phase_measurement_reset
from tools.common import check_camera_status
from module.object_detection import API_object_detection
import time
from threading import Thread
from module.nose_phase_detection import measure_nose_phase
from tools.common import get_frame, asys_get_frame
from module.edge_detection import get_edge_points, calculator_edge
import cv2
from module.exit_angle_measurement import get_angle, get_2_points, get_4_points
from module.angle_computation import API_angle_computation
from module.web_diameter_measurement import API_web_diameter_measurement
from module.rake_angle_measurement import API_rake_angle_measurement_step1, API_rake_angle_measurement_step2
from module.hermite import API_hermite

# %% Auxiliary functions

from tools.Static import STATIC
from threads.thread_camera import thread_auto_focus_capture, thread_auto_focus, \
                        thread_object_detection, thread_nose_phase_measurement_fast, \
                        thread_nose_phase_measurement_slow, thread_nose_phase_measurement_final, \
                        thread_helix_angle_measurement, thread_axial_angle_measurement, thread_pocket_angle_measurement

# %% Application setting

app = Flask(__name__)
                
app.secret_key = 'You Will Never Guess'

socketio = SocketIO(app, logger=True, cors_allowed_origins='*')

#last_time = 

# %% SocketIO events

@socketio.on('connect')
def test_connect():
    socketio.emit('client_connect', {'data': 'Client connected'})
    # Start camera services
    try:
        res = STATIC.session.get("http://localhost:3000/start/0") # Start camera 0 by default
        if res.status_code == 200:
            socketio.emit('client_connect', {'data': 'Initialize camera service successful'})
        else:
            socketio.emit('client_connect', {'data': 'Failed to initialize camera service! Attempting to re-initialize...'})        
    except:
        pass
    
@socketio.on('disconnect')
def test_disconnect():
    socketio.emit('client_disconnect', {'data': 'Client disconnected'})
    # Start camera services
    try:
        res = STATIC.session.get("http://localhost:3000/stop") # Start camera 0 by default
        if res.status_code == 200:
            socketio.emit('client_disconnect', {'data': 'Camera service has stopped successfully'})
        else:
            socketio.emit('client_disconnect', {'data': 'Failed to stop camera service!'})
    except:
        pass


@socketio.on('auto_focus')
def auto_focus_capture(data):
    cam_id = int(data['camID'])
    focus_length = int(data['focusLength'])
    focus_width = int(data['focusWidth'])
    t = Thread(target=thread_auto_focus_capture, args=(socketio, cam_id, focus_width, focus_length), daemon=True)
    t.start()

@socketio.on('auto_focus_reset')
def auto_focus_reset(data):
    auto_focus_reset()
    
@socketio.on('auto_focus_streaming_start')
def auto_focus_streaming_start(data):
    cam_id = int(data['camID'])
    focus_length = int(data['focusLength'])
    focus_width = int(data['focusWidth'])
    t = Thread(target=thread_auto_focus, args=(socketio, cam_id, focus_width, focus_length), daemon=True)
    t.start()
    
@socketio.on('auto_focus_streaming_low_stop')
def auto_focus_streaming_low_stop(data):
    STATIC.stop_flg = 1
    imageOutputID, totalFrame = find_max_sharpness()
    socketio.emit('auto_focus_streaming_low_stop',
    {
        "camID" : 0,
        "sharpness" : 0,
        "searchFlg" : 1,
        "imageOutputID" : imageOutputID,
        "totalFrame": totalFrame,
        "FPS": 30,
        "errorID" : 0 if totalFrame != -1 else 1004,
    })

@socketio.on('auto_focus_streaming_stop')
def auto_focus_streaming_stop(data):
    STATIC.stop_flg = 1
    status = clearn_duplicate()
    imageOutputID = -1
    totalFrame = -1
    if status:
        imageOutputID, totalFrame = find_max_sharpness()
    socketio.emit('auto_focus_streaming_stop',
    {
        "camID" : 0,
        "sharpness" : 0,
        "searchFlg" : int(status),
        "imageOutputID" : imageOutputID,
        "totalFrame": totalFrame,
        "FPS": 30,
        "errorID" : 0 if totalFrame != -1 else 1004,
    })

@socketio.on('object_detection')
def object_detection(data):
    cam_id = int(data['camID'])
    t = Thread(target=thread_object_detection, args=(socketio, cam_id,), daemon=True)
    t.start()

        
@socketio.on('nose_phase_measurement_fast_streaming_start')
def nose_phase_measurement_fast_streaming_start(data):
    cam_id = int(data['camID'])

    t = Thread(target=thread_nose_phase_measurement_fast, args=(socketio, cam_id,), daemon=True)
    t.start()


@socketio.on('nose_phase_measurement_fast_streaming')
def nose_phase_measurement_fast_streaming_stop(data):
    STATIC.stop_flg = 1
    try:
        socketio.emit('nose_phase_measurement_fast_streaming', 
        {
            "searchFlg" : STATIC.API_nose_phase_measurement_fast_streaming_output[0],
            "nFrame" : STATIC.API_nose_phase_measurement_fast_streaming_output[1],
            "totalFrame" : STATIC.API_nose_phase_measurement_fast_streaming_output[2],
            "differenceY" : 0,
            "differenceZ" : 0,
            "errorID" : STATIC.API_nose_phase_measurement_fast_streaming_output[5],
        })
    except:
        socketio.emit('nose_phase_measurement_fast_streaming', 
        {})
        


        
@socketio.on('nose_phase_measurement_slow_streaming_start')
def nose_phase_measurement_slow_streaming_start(data):
    cam_id = int(data['camID'])
    t = Thread(target=thread_nose_phase_measurement_slow, args=(socketio, cam_id,), daemon=True)
    t.start()


@socketio.on('nose_phase_measurement_slow_streaming')
def nose_phase_measurement_slow_streaming_stop(data):
    STATIC.stop_flg = 1
    try:
        socketio.emit('nose_phase_measurement_slow_streaming', 
        {
            "searchFlg" : STATIC.API_nose_phase_measurement_slow_streaming_output[0],
            "nFrame" : STATIC.API_nose_phase_measurement_slow_streaming_output[1],
            "totalFrame" : STATIC.API_nose_phase_measurement_slow_streaming_output[2],
            "differenceY" : 0,
            "differenceZ" : 0,
            "errorID" : STATIC.API_nose_phase_measurement_slow_streaming_output[5],
        })
    except:
        socketio.emit('nose_phase_measurement_slow_streaming', 
        {})
    nose_phase_measurement_reset()


@socketio.on("measure_diameter")
def measure_nose_phase_api(data):
    time_out = 5
    while time_out:
        img = get_frame(STATIC.session)
        if img is not None:
            break
        time_out -= 1
        time.sleep(0.05)
    if img is None:
        response = {'distanceZ' : 0, 'errorID' : 1004}
        socketio.emit("measure_diameter", str(response))
        return
    x_dis, _ = measure_nose_phase(img)
    response = {'distanceZ' : x_dis, 'errorID' : 0}
    socketio.emit('measure_diameter', str(response))
    
@socketio.on("measure_liphight")
def measure_liphight_api(data):
    time_out = 5
    while time_out:
        img = get_frame(STATIC.session)
        if img is not None:
            break
        time_out -= 1
        time.sleep(0.05)
    if img is None:
        response = {'lipHight' : 0, 'errorID' : 1004}
        socketio.emit("measure_liphight", str(response))
        return
    _, y_dis = measure_nose_phase(img)
    response = {'lipHight' : y_dis, 'errorID' : 0}
    socketio.emit("measure_liphight", str(response))

@socketio.on('nose_phase_measurement_final_streaming')
def nose_phase_measurement_final(data):
    cam_id = int(data['camID'])

    t = Thread(target=thread_nose_phase_measurement_final, args=(socketio, cam_id,), daemon=True)
    t.start()

@socketio.on('straight_edge_detection')
def straight_edge_detection(data):
    time.sleep(0.6)
    img = asys_get_frame(STATIC.session)
    if img is None:
        socketio.emit("straight_edge_detection", {"errorID" : 1004})
        return
    direction = 0
    response = calculator_edge(data, img, direction)
    socketio.emit('straight_edge_detection', str(response))
    
@socketio.on("side_edge_detection")
def side_edge_detection(data):
    time.sleep(0.6)
    img = asys_get_frame(STATIC.session)
    if img is None:
        socketio.emit("side_edge_detection", {"errorID" : 1004})
        return
    direction = 1
    response = calculator_edge(data, img, direction)
    socketio.emit('side_edge_detection', str(response))

@socketio.on("helix_angle_measurement")
def helix_angle_measurement(data):
    cam_id = int(data['camID'])
    t = Thread(target=thread_helix_angle_measurement, args=(socketio, cam_id,), daemon=True)
    t.start()
    
@socketio.on("reliefAngle_measurement_step1")
def reliefAngle_measurement_step1(data):
    time.sleep(0.07)
    img = asys_get_frame(STATIC.session)
    if img is None:
        socketio.emit("reliefAngle_measurement_step1", {"errorID" : 1004})
        return
    angle, error_id = get_angle(img)
    socketio.emit("reliefAngle_measurement_step1", {"angle" : angle, 'errorID' : error_id})
    
@socketio.on("reliefAngle_measurement_step2")
def reliefAngle_measurement_step2(data):
    time.sleep(0.07)
    img = asys_get_frame(STATIC.session)
    if img is None:
        socketio.emit("reliefAngle_measurement_step2", {"errorID" : 1004})
        return
    p1, p2, error_id = get_2_points(img, data)
    socketio.emit("reliefAngle_measurement_step2", {
        "Y11" : -p1[1],
        "Z11" : p1[0],
        "Y21" : -p2[1],
        "Z21" : p2[0],
        "errorID" : error_id
    })
    
@socketio.on("reliefAngle_measurement_step3")
def reliefAngle_measurement_step3(data):
    time.sleep(0.07)
    img = asys_get_frame(STATIC.session)
    if img is None:
        socketio.emit("reliefAngle_measurement_step3", {"errorID" : 1004})
        return
    p1, p2, p3, p4, error_id = get_4_points(img, data)
    socketio.emit("reliefAngle_measurement_step3", {
        "Y12" : -p1[1],
        "Z12" : p1[0],
        "Y13" : -p2[1],
        "Z13" : p2[0],
        "Y22" : -p3[1],
        "Z22" : p3[0],
        "Y23" : -p4[1],
        "Z23" : p4[0],
        "errorID" : error_id
    })

@socketio.on('reliefAngle_measurement_step4')
def exit_angle_measurement_step4(data):
    try: 
        data = json.loads(data) 
    except: pass
    focus_point_arr1 = data['focusPointArr1']
    focus_point_arr2 = data['focusPointArr2']
    
    # Uncomment these 2 lines below if parse 'focusPointArr1' and 'focusPointArr2' in string
    '''
    focus_point_arr1 = ast.literal_eval(focus_point_arr1)
    focus_point_arr2 = ast.literal_eval(focus_point_arr2)
    '''
    
    print(focus_point_arr1, focus_point_arr2)
    
    flat_point_arr1 = [item for sublist in focus_point_arr1 for item in sublist]
    flat_point_arr2 = [item for sublist in focus_point_arr2 for item in sublist]
    
    print(flat_point_arr1, flat_point_arr2)
    
    # TO DO - just call function to compute
    output_relief, error = API_angle_computation(
        flat_point_arr1[0], flat_point_arr1[1], flat_point_arr1[2], \
        flat_point_arr1[3], flat_point_arr1[4], flat_point_arr1[5], \
        flat_point_arr1[6], flat_point_arr1[7], flat_point_arr1[8], \
        0, 0, 0, 0, 1, 0, 0, 0, 1)
        
    output_relief_2, error_2 = API_angle_computation(
        flat_point_arr2[0], flat_point_arr2[1], flat_point_arr2[2], \
        flat_point_arr2[3], flat_point_arr2[4], flat_point_arr2[5], \
        flat_point_arr2[6], flat_point_arr2[7], flat_point_arr2[8], \
        0, 0, 0, 0, 1, 0, 0, 0, 1)
  
    socketio.emit('reliefAngle_measurement_step4', 
    {
        "relief" : output_relief,
        "relief_2" : output_relief_2,
        "errorID" : max(error, error_2),
    })

# GASH
@socketio.on('axial_angle_measurement')
def axial_angle_measurement(data):
    cam_id = int(data['camID'])
    focus_length = int(data['focusLength'])
    focus_width = int(data['focusWidth'])
    t = Thread(target=thread_axial_angle_measurement, args=(socketio, cam_id, focus_width, focus_length), daemon=True)
    t.start()
    
# POCKET
@socketio.on('pocket_angle_measurement')
def pocket_angle_measurement(data):
    cam_id = int(data['camID'])
    focus_length = int(data['focusLength'])
    focus_width = int(data['focusWidth'])
    t = Thread(target=thread_pocket_angle_measurement, args=(socketio, cam_id, focus_width, focus_length), daemon=True)
    t.start()

# WEB DIAMETER MEASUREMENT
def thread_web_diameter_measurement(socketio, cam_id, focus_width, px2mm):
    if not emit_camera_status(STATIC.session, socketio, cam_id):
        return
    
    img = get_frame(STATIC.session)
    if img is None:
        socketio.emit('web_diameter_measurement',     
        {
            "webDiameter" : 0,
            "distance": 0,
            "moveFocusFlg": 0,
            "errorID" : 1004,
        })
        return 

    API_web_diameter_measurement_output = API_web_diameter_measurement(img)
        
    socketio.emit('web_diameter_measurement',     
    {
        "webDiameter" : API_web_diameter_measurement_output[0] * px2mm,
        "distance": API_web_diameter_measurement_output[1] * px2mm,
        "moveFocusFlg": API_web_diameter_measurement_output[2],
        "errorID" : API_web_diameter_measurement_output[3],
    })
    
@socketio.on('web_diameter_measurement')
def web_measurement(data):
    cam_id = int(data['camID'])
    focus_length = int(data['focusLength'])
    focus_width = int(data['focusWidth'])
    px2mm = float(data['px2mm'])
    t = Thread(target=thread_web_diameter_measurement, args=(socketio, cam_id, focus_width, px2mm), daemon=True)
    t.start()
    
# THINNING ANGLE
def thread_rakeAngle_measurement_step1(socketio, cam_id, length, px2mm):
    if not emit_camera_status(STATIC.session, socketio, cam_id):
        return
    
    img = get_frame(STATIC.session)
    if img is None:
        socketio.emit('rakeAngle_measurement_step1',     
        {
            "angle": 0,
            "moveFocusFlg": 0,
            "errorID" : 1004,
        })
        return 
    
    angle, moveFocusFlg, error_id = API_rake_angle_measurement_step1(img, length, px2mm)
        
    socketio.emit('rakeAngle_measurement_step1',     
    {
        "angle": 90 - np.degrees(angle),
        "moveFocusFlg": moveFocusFlg,
        "errorID" : error_id
    })

@socketio.on("rakeAngle_measurement_step1")
def rakeAngle_measurement_step1(data):
    cam_id = int(data['camID'])
    focus_length = int(data['focusLength'])
    focus_width = int(data['focusWidth'])
    px2mm = float(data['px2mm'])
    
    length = min(focus_width, focus_length)
    
    t = Thread(target=thread_rakeAngle_measurement_step1, args=(socketio, cam_id, length, px2mm), daemon=True)
    t.start()
    
def thread_rakeAngle_measurement_step2(socketio, cam_id, length, px2mm):
    if not emit_camera_status(STATIC.session, socketio, cam_id):
        return
    
    img = get_frame(STATIC.session)
    if img is None:
        print("bbbbbbb")
        socketio.emit('rakeAngle_measurement_step2',     
        {
            "Y1" : 0,
            "Z1" : 0,
            "Y2" : 0,
            "Z2" : 0,
            "Y3" : 0,
            "Z3" : 0,
            "detectFlg": 0,
            "errorID" : 1004,
        })
        return 
    
    p1, p2, p3, detectFlg, error_id = API_rake_angle_measurement_step2(img, length, px2mm)
        
    socketio.emit('rakeAngle_measurement_step2', 
    {
        "Y1" : -p1[1],
        "Z1" : p1[0],
        "Y2" : -p2[1],
        "Z2" : p2[0],
        "Y3" : -p3[1],
        "Z3" : p3[0],
        "detectFlg" : detectFlg,
        "errorID" : error_id
    })

@socketio.on("rakeAngle_measurement_step2")
def rakeAngle_measurement_step2(data):
    cam_id = int(data['camID'])
    focus_length = int(data['focusLength'])
    focus_width = int(data['focusWidth'])
    px2mm = float(data['px2mm'])
    
    length = min(focus_width, focus_length)
    
    t = Thread(target=thread_rakeAngle_measurement_step2, args=(socketio, cam_id, length, px2mm), daemon=True)
    t.start()
    
@socketio.on("rakeAngle_measurement_step3")
def rakeAngle_measurement_step3(data):
    try: 
        data = json.loads(data) 
    except: 
        pass
    focus_point_arr = data['focusPointArr']
    
    # Uncomment the line below if parse 'focusPointArr' in string
    '''
    focus_point_arr = ast.literal_eval(focus_point_arr)
    '''
    
    flat_point_arr = [item for sublist in focus_point_arr for item in sublist]
       
    # TO DO - just call function to compute
    output_rake, error = API_angle_computation(
        flat_point_arr[0], flat_point_arr[1], flat_point_arr[2], \
        flat_point_arr[3], flat_point_arr[4], flat_point_arr[5], \
        flat_point_arr[6], flat_point_arr[7], flat_point_arr[8], \
        0, 0, 0, 0, 1, 0, 0, 0, 1)
  
    socketio.emit('rakeAngle_measurement_step3', 
    {
        "rakeAngle" : output_rake,
        "errorID" : max(0, error),
    })

# Hermite smooth curve
def thread_hermite(points, filename):
    error_id = API_hermite(points, filename)
    
    # Write a new csv file for GH5
    
    if error_id == 0:        
        socketio.emit('edgeDetection_fn', 
        {
            "fnFlg" : 1,
            "errorID" : error_id,
        })
    else:
        socketio.emit('edgeDetection_fn', 
        {
            "fnFlg" : 0,
            "errorID" : error_id,
        })
        
@socketio.on("edgeDetection_fn")
def edge_detection_fn(data):
    csv_path = data['pathEdge']
    points = pd.read_csv(csv_path)

    head, tail = os.path.split(csv_path)
    filename = os.path.splitext(tail)[0]

    t = Thread(target=thread_hermite, args=(points, filename), daemon=True)
    t.start()

# %% API routes

@app.route('/')
def hello_world():
    return 'CNC application API server'
    
# %% Main function
if __name__ == '__main__':
    print("Start [Flask] server for the CNC at localhost:8000")    
    socketio.run(app, host='127.0.0.1', port=8000)