from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import time
import math
import numpy as np
import threading
import json

app = Flask(__name__)

class PushUpCounter:
    def __init__(self):
        self.mode = False
        self.smooth = True
        self.detectionCon = 0.8
        self.trackingCon = 0.5
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            smooth_landmarks=self.smooth,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackingCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.count = 0
        self.dir = 0
        self.bar = 0
        self.per = 0
        self.is_running = False
        self.cap = None
        self.start_time = None
        self.session_time = 0

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if draw and self.results.pose_landmarks:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img
    
    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                self.lmList.append([id, cx, cy])
                if draw: 
                    cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        return self.lmList
    
    def findAngle(self, img, p1, p2, p3, draw=True):
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        x3, y3 = self.lmList[p3][1], self.lmList[p3][2]
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        if angle > 180: 
            angle = 360 - angle
        elif angle < 0: 
            angle = -angle
        if draw:
            cv2.circle(img, (x1, y1), 10, (64, 127, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (64, 127, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 10, (64, 127, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (64, 127, 255))
            cv2.circle(img, (x2, y2), 15, (64, 127, 255))
            cv2.circle(img, (x3, y3), 15, (64, 127, 255))
            cv2.line(img, (x1, y1), (x2, y2), (255, 127, 64), 3)
            cv2.line(img, (x2, y2), (x3, y3), (255, 127, 64), 3)
        return angle

    def start_session(self):
        self.is_running = True
        self.count = 0
        self.dir = 0
        self.start_time = time.time()
        self.cap = cv2.VideoCapture(0)
        return True

    def stop_session(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        return True

    def get_stats(self):
        if self.start_time:
            self.session_time = time.time() - self.start_time
        return {
            'count': int(self.count),
            'percentage': int(self.per),
            'session_time': int(self.session_time),
            'is_running': self.is_running
        }

# Global counter instance
counter = PushUpCounter()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start')
def start_counter():
    success = counter.start_session()
    return jsonify({'success': success, 'message': 'Push-up counter started'})

@app.route('/stop')
def stop_counter():
    success = counter.stop_session()
    return jsonify({'success': success, 'message': 'Push-up counter stopped'})

@app.route('/stats')
def get_stats():
    return jsonify(counter.get_stats())

def generate_frames():
    pTime = 0
    
    while counter.is_running and counter.cap:
        success, img = counter.cap.read()
        if not success:
            break
        
        img = cv2.resize(img, (640, 480))
        img = counter.findPose(img, draw=False)
        lmList = counter.findPosition(img, draw=False)

        if len(lmList):
            if (lmList[31][2] + 50 > lmList[29][2] and lmList[32][2] + 50 > lmList[30][2]):
                angle = counter.findAngle(img, 11, 13, 15)
                counter.findAngle(img, 12, 14, 16)
                counter.findAngle(img, 27, 29, 31)
                counter.findAngle(img, 28, 30, 32)
                
                counter.per = -1.25 * angle + 212.5
                counter.per = (0 if counter.per < 0 else 100 if counter.per > 100 else counter.per)
                counter.bar = np.interp(counter.per, (0, 100), (400, 50))
                
                if counter.per >= 95:
                    if counter.dir == 0:
                        counter.count += 0.5
                        counter.dir = 1
                elif counter.per <= 5:
                    if counter.dir == 1:
                        counter.count += 0.5
                        counter.dir = 0

                img = cv2.flip(img, 1)
                
                # Add visual feedback
                if (counter.per >= 95 or counter.per <= 5):
                    cv2.putText(img, f'{int(counter.per)}%', (550, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(img, f'{int(counter.per)}%', (550, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                # Progress bar
                cv2.rectangle(img, (550, 70), (600, 400), (0, 0, 255), 2)
                cv2.rectangle(img, (550, int(counter.bar)), (600, 400), (0, 255, 0), cv2.FILLED)
        else:
            img = cv2.flip(img, 1)
            cv2.putText(img, 'Take your position', (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        # Add count display
        cv2.rectangle(img, (10, 10), (200, 60), (0, 0, 0), cv2.FILLED)
        cv2.putText(img, f'Count: {int(counter.count)}', (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
        # FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime) if pTime != 0 else 0
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)