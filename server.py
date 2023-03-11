from engineio.async_drivers import gevent
from flask import Flask, request, jsonify, Response
from flask_socketio import SocketIO

import cv2
import humps

import logging

from module.auto_focus_streaming import API_auto_focus_streaming, auto_focus_streaming_reset
from module.nose_phase_measurement_streaming import API_nose_phase_measurement_fast_streaming, API_nose_phase_measurement_slow_streaming, API_nose_phase_measurement_final_streaming, nose_phase_measurement_reset

import numpy as np
import os

import sys
import time
import threading
from threading import Thread
import requests

# %% Auxiliary functions



# %% Application setting

app = Flask(__name__)
                
app.secret_key = 'You Will Never Guess'

socketio = SocketIO(app, logger=True, cors_allowed_origins='*')

duration = 0

global session
session = requests.Session()

# %% SocketIO events

@socketio.on('connect')
def test_connect():
    socketio.emit('client_connect', {'data': 'Client connected'})
    
    '''
    # Start camera services
    global session
    try:
        while True:
            res = session.get("http://localhost:3000/start/0") # Start camera 0 by default
            if res.status_code == 200:
                socketio.emit('client_connect', {'data': 'Initialize camera service successful'})
                break
            else:
                socketio.emit('client_connect', {'data': 'Failed to initialize camera service! Attempting to re-initialize...'})        
    except:
        pass
    '''
    
    
@socketio.on('disconnect')
def test_disconnect():
    socketio.emit('client_disconnect', {'data': 'Client disconnected'})


# AUTO FOCUS
def thread_auto_focus(cam_id, focus_width, focus_length):
    
    global session
    status_run_camera = True
    try:
        res = session.get(f"http://localhost:3000/camera_status/{cam_id}")
        if res.status_code == 200:
            data = res.json()
            if data["status"] == 1:    
                socketio.emit('auto_focus_streaming_start', {'readyFlg': 1})
            else:
                status_run_camera = False
                socketio.emit('auto_focus_streaming_start', {'readyFlg': -1})
        else:
            status_run_camera = False
            socketio.emit('auto_focus_streaming_start', {'readyFlg': -1})
            
    except:
        status_run_camera = False
        socketio.emit('auto_focus_streaming_start', {'readyFlg': -1})
        
    if not status_run_camera:
        return
    
    global stop_flg
    stop_flg = 0
    
    auto_focus_streaming_reset()

    socketio.emit('auto_focus_streaming_start', {'readyFlg': 1})
    
    while not stop_flg:
        res = session.get("http://localhost:3000/frame")

        if res.status_code == 200:
            img = res.json()["img"]
            if len(img) == 0:
                continue
            img = bytes(img)
            np_array = np.frombuffer(img, np.uint8)
            img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
            
            global API_auto_focus_streaming_output
            API_auto_focus_streaming_output = API_auto_focus_streaming(focus_width, focus_length, img)
        
        time.sleep(0.04)

    
@socketio.on('auto_focus_streaming_start')
def auto_focus_streaming_start(data):
    
    global cam_id
    global fps
    fps = 30 # Hard-coded for testing ONLY
    
    cam_id = int(data['camID'])
    focus_length = int(data['focusLength'])
    focus_width = int(data['focusWidth'])

    t = Thread(target=thread_auto_focus, args=(cam_id, focus_width, focus_length), daemon=True)
    t.start()
       

@socketio.on('auto_focus_streaming_stop')
def auto_focus_streaming_stop(data):
        
    global stop_flg
    stop_flg = 1

    #socketio.emit('auto_focus_streaming_stop', {'stopFlg': 1})
    try:
        socketio.emit('auto_focus_streaming_stop',
        {
            "camID" : cam_id,
            "sharpness" : API_auto_focus_streaming_output[0],
            "searchFlg" : API_auto_focus_streaming_output[1],
            "imageOutputID" : API_auto_focus_streaming_output[2],
            "totalFrame": API_auto_focus_streaming_output[3],
            "FPS": fps,
            "errorID" : API_auto_focus_streaming_output[4],
        })
    except:
        socketio.emit('auto_focus_streaming_stop',
        {})


# NOSE PHASE MEASUREMENT
# [FAST]
def thread_nose_phase_measurement_fast(cam_id):
    
    global session
    status_run_camera = True
    try:
        res = session.get(f"http://localhost:3000/camera_status/{cam_id}")
        if res.status_code == 200:
            data = res.json()
            if data["status"] == 1:    
                socketio.emit('nose_phase_measurement_fast_streaming_start', {'readyFlg': 1})
            else:
                status_run_camera = False
                socketio.emit('nose_phase_measurement_fast_streaming_start', {'readyFlg': -1})
        else:
            status_run_camera = False
            socketio.emit('nose_phase_measurement_fast_streaming_start', {'readyFlg': -1})
            
    except:
        status_run_camera = False
        socketio.emit('nose_phase_measurement_fast_streaming_start', {'readyFlg': -1})
        
    if not status_run_camera:
        return
    
    global stop_flg
    stop_flg = 0
    
    nose_phase_measurement_reset()
    
    socketio.emit('nose_phase_measurement_fast_streaming_start', {'readyFlg': 1})
    
    while not stop_flg:
        res = session.get("http://localhost:3000/frame")

        if res.status_code == 200:
            img = res.json()["img"]
            if len(img) == 0:
                continue
            img = bytes(img)
            np_array = np.frombuffer(img, np.uint8)
            img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
            
            global API_nose_phase_measurement_fast_streaming_output
            API_nose_phase_measurement_fast_streaming_output = API_nose_phase_measurement_fast_streaming(img)
        
        time.sleep(0.04)
        
        
@socketio.on('nose_phase_measurement_fast_streaming_start')
def nose_phase_measurement_fast_streaming_start(data):
    global cam_id
    global fps
    fps = 30 # Hard-coded for testing ONLY
    
    cam_id = int(data['camID'])

    t = Thread(target=thread_nose_phase_measurement_fast, args=(cam_id,), daemon=True)
    t.start()


@socketio.on('nose_phase_measurement_fast_streaming')
def nose_phase_measurement_fast_streaming_stop(data):

    global stop_flg
    stop_flg = 1

    #socketio.emit('nose_phase_measurement_fast_streaming', {'stopFlg': 1})
    try:
        socketio.emit('nose_phase_measurement_fast_streaming', 
        {
            "searchFlg" : API_nose_phase_measurement_fast_streaming_output[0],
            "nFrame" : API_nose_phase_measurement_fast_streaming_output[1],
            "totalFrame" : API_nose_phase_measurement_fast_streaming_output[2],
            "differenceY" : int(API_nose_phase_measurement_fast_streaming_output[3]),
            "differenceZ" : int(API_nose_phase_measurement_fast_streaming_output[4]),
            "errorID" : API_nose_phase_measurement_fast_streaming_output[5],
        })
    except:
        socketio.emit('nose_phase_measurement_fast_streaming', 
        {})
        

# [SLOW]
def thread_nose_phase_measurement_slow(cam_id):
    
    global session
    status_run_camera = True
    try:
        res = session.get(f"http://localhost:3000/camera_status/{cam_id}")
        if res.status_code == 200:
            data = res.json()
            if data["status"] == 1:    
                socketio.emit('nose_phase_measurement_slow_streaming_start', {'readyFlg': 1})
            else:
                status_run_camera = False
                socketio.emit('nose_phase_measurement_slow_streaming_start', {'readyFlg': -1})
        else:
            status_run_camera = False
            socketio.emit('nose_phase_measurement_slow_streaming_start', {'readyFlg': -1})
            
    except:
        status_run_camera = False
        socketio.emit('nose_phase_measurement_slow_streaming_start', {'readyFlg': -1})
        
    if not status_run_camera:
        return
    
    global stop_flg
    stop_flg = 0
    
    nose_phase_measurement_reset()
    
    socketio.emit('nose_phase_measurement_slow_streaming_start', {'readyFlg': 1})
    
    while not stop_flg:
        res = session.get("http://localhost:3000/frame")

        if res.status_code == 200:
            img = res.json()["img"]
            if len(img) == 0:
                continue
            img = bytes(img)
            np_array = np.frombuffer(img, np.uint8)
            img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
            
            global API_nose_phase_measurement_slow_streaming_output
            API_nose_phase_measurement_slow_streaming_output = API_nose_phase_measurement_slow_streaming(img)
        
        time.sleep(0.04)
        
        
@socketio.on('nose_phase_measurement_slow_streaming_start')
def nose_phase_measurement_slow_streaming_start(data):
    global cam_id
    global fps
    fps = 30 # Hard-coded for testing ONLY
    
    cam_id = int(data['camID'])

    t = Thread(target=thread_nose_phase_measurement_slow, args=(cam_id,), daemon=True)
    t.start()


@socketio.on('nose_phase_measurement_slow_streaming')
def nose_phase_measurement_slow_streaming_stop(data):

    global stop_flg
    stop_flg = 1

    #socketio.emit('nose_phase_measurement_slow_streaming', {'stopFlg': 1})
    try:
        socketio.emit('nose_phase_measurement_slow_streaming', 
        {
            "searchFlg" : API_nose_phase_measurement_slow_streaming_output[0],
            "nFrame" : API_nose_phase_measurement_slow_streaming_output[1],
            "totalFrame" : API_nose_phase_measurement_slow_streaming_output[2],
            "differenceY" : int(API_nose_phase_measurement_slow_streaming_output[3]),
            "differenceZ" : int(API_nose_phase_measurement_slow_streaming_output[4]),
            "errorID" : API_nose_phase_measurement_slow_streaming_output[5],
        })
    except:
        socketio.emit('nose_phase_measurement_slow_streaming', 
        {})
        
    nose_phase_measurement_reset()


# [FINAL - SUPER SLOW]
def thread_nose_phase_measurement_final(cam_id):
    
    global session
    status_run_camera = True
    try:
        res = session.get(f"http://localhost:3000/camera_status/{cam_id}")
        if res.status_code == 200:
            data = res.json()
            if data["status"] == 1:    
                socketio.emit('nose_phase_measurement_final_streaming', {'readyFlg': 1})
            else:
                status_run_camera = False
                socketio.emit('nose_phase_measurement_final_streaming', {'readyFlg': -1})
        else:
            status_run_camera = False
            socketio.emit('nose_phase_measurement_final_streaming', {'readyFlg': -1})
            
    except:
        status_run_camera = False
        socketio.emit('nose_phase_measurement_final_streaming', {'readyFlg': -1})
        
    if not status_run_camera:
        return
    
    res = session.get("http://localhost:3000/frame")

    if res.status_code == 200:
        img = res.json()["img"]
        if len(img) == 0:
            pass
        
        img = bytes(img)
        np_array = np.frombuffer(img, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        API_nose_phase_measurement_final_streaming_output = API_nose_phase_measurement_final_streaming(img)
        
    socketio.emit('nose_phase_measurement_final_streaming',     
    {
        "searchFlg" : API_nose_phase_measurement_final_streaming_output[0],
        "differenceY" : int(API_nose_phase_measurement_final_streaming_output[1]),
        "differenceZ" : int(API_nose_phase_measurement_final_streaming_output[2]),
        "moveFlg" : int(API_nose_phase_measurement_final_streaming_output[3]),
        "errorID" : API_nose_phase_measurement_final_streaming_output[4],
    })


@socketio.on('nose_phase_measurement_final_streaming')
def nose_phase_measurement_final(data):
    global cam_id
    global fps
    fps = 30 # Hard-coded for testing ONLY
    
    cam_id = int(data['camID'])

    t = Thread(target=thread_nose_phase_measurement_final, args=(cam_id,), daemon=True)
    t.start()


# %% API routes

@app.route('/')
def hello_world():
    return 'CNC application API server'
    
    
# %% Main function

if __name__ == '__main__':
    print("Start [Flask] server for the CNC at localhost:8000")    
    socketio.run(app, host='127.0.0.1', port=8000)