import cv2
import imutils

import numpy

import os
import time

# %% Initial values

# %% Auxiliary functions

# %% API definition

def API_video_streaming(video):   
    while True:
        success, image = video.read()   
        if not success:
            break
        else:
            # cv2.imshow('Video frame', image)
            
            frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.equalizeHist(frame_gray)
    
            ret, jpeg = cv2.imencode('.jpg', image)           

            frame = jpeg.tobytes()
            
            # For testing ONLY
            '''
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            '''
            
            '''
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            '''
            
            yield image

# %% Main function

if __name__ == "__main__":
    
    # For testing ONLY
    '''
    video_path = 'D:/Working/KhoanCNC/Nam_work/experiment/auto-focus-video/04.wmv'
    video = cv2.VideoCapture(video_path)
    
    API_video_streaming(video)
    
    video.release()
    cv2.destroyAllWindows()
    '''
        
    pass