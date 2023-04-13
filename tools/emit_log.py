from tools.common import check_camera_status

def emit_camera_status(session, socketio, cam_id):
    if check_camera_status(session, cam_id):
        socketio.emit("camera_ready", {'readyFlg': 1})
        return True
    socketio.emit("camera_ready", {'readyFlg': 0})
    return False