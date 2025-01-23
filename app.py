# app.py
from flask import Flask, render_template, jsonify, Response
import cv2
import mediapipe as mp
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np
import screen_brightness_control as sbc
import pyautogui
from threading import Thread, Event
import json

app = Flask(__name__)

class GestureControl:
    def __init__(self):
        self.is_running = Event()
        self.capture_thread = None
        self.cap = None
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mpDraw = mp.solutions.drawing_utils
        
        # Setup audio control
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = cast(interface, POINTER(IAudioEndpointVolume))
        self.volMin, self.volMax = self.volume.GetVolumeRange()[:2]
        
        self.brightnessMin, self.brightnessMax = 0, 100
        self.screenWidth, self.screenHeight = pyautogui.size()
        
        self.currentVolume = 0
        self.currentBrightness = 0
        self.status = {
            'volume': 0,
            'brightness': 0,
            'mouse_position': {'x': 0, 'y': 0}
        }

    def start(self):
        if not self.is_running.is_set():
            self.cap = cv2.VideoCapture(0)
            self.is_running.set()
            self.capture_thread = Thread(target=self.run)
            self.capture_thread.start()
            return True
        return False

    def stop(self):
        if self.is_running.is_set():
            self.is_running.clear()
            if self.cap:
                self.cap.release()
            if self.capture_thread:
                self.capture_thread.join()
            cv2.destroyAllWindows()
            return True
        return False

    def get_frame(self):
        success, img = self.cap.read()
        if success:
            img = cv2.flip(img, 1)
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(imgRGB)

            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    self.mpDraw.draw_landmarks(img, hand_landmarks, self.mpHands.HAND_CONNECTIONS)
                    
                    # Process gestures
                    handedness = results.multi_handedness[idx].classification[0].label
                    if handedness == "Left":
                        self.process_brightness(hand_landmarks, img)
                    else:
                        self.process_volume(hand_landmarks, img)

            ret, jpeg = cv2.imencode('.jpg', img)
            return jpeg.tobytes()
        return None

    def process_volume(self, hand_landmarks, img):
        x1, y1 = hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y
        x2, y2 = hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y
        length = hypot(x2 - x1, y2 - y1)
        vol = np.interp(length, [0.1, 0.5], [self.volMin, self.volMax])
        self.volume.SetMasterVolumeLevel(vol, None)
        volPer = int(np.interp(vol, [self.volMin, self.volMax], [0, 100]))
        self.status['volume'] = volPer
        cv2.putText(img, f"Volume: {volPer}%", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def process_brightness(self, hand_landmarks, img):
        x1, y1 = hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y
        x2, y2 = hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y
        length = hypot(x2 - x1, y2 - y1)
        brightness = int(np.interp(length, [0.1, 0.5], [self.brightnessMin, self.brightnessMax]))
        sbc.set_brightness(brightness)
        self.status['brightness'] = brightness
        cv2.putText(img, f"Brightness: {brightness}%", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    def run(self):
        while self.is_running.is_set():
            frame = self.get_frame()
            if frame is None:
                break

# Initialize the gesture control system
gesture_control = GestureControl()

def gen_frames():
    while True:
        if gesture_control.is_running.is_set():
            frame = gesture_control.get_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start')
def start():
    success = gesture_control.start()
    return jsonify({'success': success})

@app.route('/stop')
def stop():
    success = gesture_control.stop()
    return jsonify({'success': success})

@app.route('/status')
def status():
    return jsonify(gesture_control.status)

if __name__ == '__main__':
    app.run(debug=True)