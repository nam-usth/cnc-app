from flask import Flask, request, jsonify, Response
from gevent.pywsgi import WSGIServer
import werkzeug

import cv2
from module.auto_focus import API_auto_focus
import numpy as np
import os

app = Flask(__name__)
            
app.secret_key = 'You Will Never Guess'
 
@app.route('/')
def hello_world():
    return 'Hello World'

@app.route('/api/auto_focus', methods=['POST'])
def auto_focus():
    width = int(request.json['width'])
    height = int(request.json['height'])   
    img_path = request.json['image']
    frame_id = int(request.json['frame_id']) 
    
    img = cv2.imread(img_path)
    return jsonify({
        "frame_id_max" : API_auto_focus(width, height, img, frame_id)[0],
        "move" : API_auto_focus(width, height, img, frame_id)[1], 
    })

if __name__ == '__main__':
    print("Start server for the CNC at localhost:8000")
    http_server = WSGIServer(("127.0.0.1", 8000), app)
    http_server.serve_forever()