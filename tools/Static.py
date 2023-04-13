import requests

class STATIC:
    session = requests.Session()
    current_time = None
    last_time = None
    stop_flg = 0
    API_nose_phase_measurement_slow_streaming_output = [0, 1, 2, 3, 4, 5]
    API_nose_phase_measurement_fast_streaming_output = [0, 1, 2, 3, 4, 5]

    #static for edge detection
    step = 0.01
    num_step = 50
    px2mm = 0.00118764845
    space = 0
    list_points = []
    list_last_points = []    
    thr = 0.01
    #
    index = 0
