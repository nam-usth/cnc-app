import cv2
from flask import Flask, request, jsonify, Response
import humps
import numpy as np
import os

import threading
from threading import Thread

import werkzeug
from werkzeug.debug import DebuggedApplication

from gevent.pywsgi import WSGIServer
import time
from datetime import datetime

# %% Application setting

app = Flask(__name__)
                
app.secret_key = 'You Will Never Guess'

global frame
global stop
global camid
global status_camera
frame = None
stop = False
status_camera = False
camid = None

def run_in_background():
    global frame
    global stop
    global camid
    global status_camera
    
    while True:
        if stop or camid == None:
            time.sleep(0.5)
            status_camera = False
            continue
        source = cv2.VideoCapture(int(camid))
        if not source.isOpened():
            time.sleep(0.5)
            status_camera = False
            continue
        status_camera = True
        while True:
           isFrame, frame = source.read()
           if not isFrame or stop:
               status_camera = False
               break
        source.release()
        

@app.route('/frame', methods=['GET'])
def get_frame():
    global frame
    if frame is None:
        return jsonify({
            "img" : ""
        })
    byteimg = cv2.imencode('.jpg', frame)[1].tostring()
    byteimg = str(byteimg)
    frame = None
    return jsonify({
        "img" : byteimg
        })

@app.route('/stop', methods=['GET'])
def stop_cam():
    global stop
    global status_camera
    global camid
    stop = True
    time_wait = 20
    status = None
    camid = None
    
    f = open("camera_time_measure.txt", "a+")
    f.write("TURN OFF - Start " + str(status_camera) + " - " + str(datetime.now()) + "\n")
    f.close() 
    
    while True:
        if not status_camera:
            status = True
            break
        time.sleep(0.1)
        time_wait -= 1
        if time_wait <= 0:
            break
        
    f = open("camera_time_measure.txt", "a+")
    f.write("TURN OFF - End " + str(status_camera) + " - " + str(datetime.now()) + "\n")
    f.close()     
    
    return jsonify({
            "status" : 1 if status else -1
        })

@app.route("/camera_status/<camid_>", methods=["GET"])
def camera_status(camid_):
    global camid
    global status_camera
    if camid != int(camid_):
        return jsonify({
            "status" : -1
            })
    return jsonify({
        "status" : 1 if status_camera else -1
        })

@app.route('/start/<camid_>', methods=['GET'])
def start_cam(camid_):
    global stop
    global status_camera
    global camid
    
    f = open("camera_time_measure.txt", "a+")
    f.write("TURN ON - Start " + str(status_camera) + " - " + str(datetime.now()) + "\n")
    f.close()  
    
    if status_camera:
        if camid == int(camid_):
            return jsonify({
                "status" : -1,
                "mess" : f"Camera {camid} is running"
                })
        else:
            return jsonify({
                "status" : -1,
                "mess" : f"Camera {camid} is running. You must turn off before run cam {camid_}"
                })
    
    stop = False
    camid = int(camid_)
    time_wait = 50
    
    status = None
    while True:
        if status_camera:
            status = True
            break
        time.sleep(0.1)
        time_wait -= 1
        if time_wait <= 0:
            break
        
    f = open("camera_time_measure.txt", "a+")
    f.write("TURN ON - End " + str(status_camera) + " - " + str(datetime.now()) + "\n")
    f.close()    
        
    return jsonify({
            "status" : 1 if status else -1
        })

@app.route('/hello', methods=['GET'])
def hello():
    return jsonify({
        "img" : "hello"
        })

# %% Main function
if __name__ == '__main__':
    # Convert all JSON keys format to snake_case
    t = Thread(target=run_in_background, args=())
    t.start()
    print("start server at port : ", 3000)
    app = DebuggedApplication(app)
    http_server = WSGIServer(("127.0.0.1", 3000), app)
    http_server.serve_forever()
    t.join()