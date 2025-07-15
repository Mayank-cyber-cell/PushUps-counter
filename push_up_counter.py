import cv2
import mediapipe as mp
import time
import math
import numpy as np

class poseDetector():
    
    def __init__(self, mode=False, smooth=True, detectionCon = 0.5, trackingCon=0.5):
        self.mode = mode
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode = self.mode,
            smooth_landmarks = self.smooth,
            min_detection_confidence = self.detectionCon,
            min_tracking_confidence = self.trackingCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if draw and self.results.pose_landmarks:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img
    
    def findPosition(self, img, draw=True):
        self.lmList = []
        if  self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                self.lmList.append([id, cx, cy])
                if draw: cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        return self.lmList
    
    def findAngle(self, img, p1, p2, p3, draw=True):
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        x3, y3 = self.lmList[p3][1], self.lmList[p3][2]
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        if angle > 180: angle = 360 - angle
        elif angle < 0: angle = -angle
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



cap = cv2.VideoCapture(0)
pTime = 0
detector = poseDetector(detectionCon=0.8)
count, dir, bar, per = 0, 0, 0, 0

while True:
    success, img = cap.read()
    img = cv2.resize(img, (1366, 780))
    img = detector.findPose(img, draw=False)
    lmList = detector.findPosition(img, draw=False)

    if (len(lmList)):
        if (lmList[31][2] + 50 > lmList[29][2] and lmList[32][2] + 50 > lmList[30][2]):
            angle = detector.findAngle(img, 11, 13, 15)
            detector.findAngle(img, 12, 14, 16)
            detector.findAngle(img, 27, 29, 31)
            detector.findAngle(img, 28, 30, 32)
            
            per = -1.25 * angle + 212.5
            per = (0 if per < 0 else 100 if per > 100 else per)
            bar = np.interp(per, (0, 100), (650, 100))
            
            if per >= 95:
                if dir == 0:
                    count += 0.5
                    dir = 1
            elif per <= 5:
                if dir == 1:
                    count += 0.5
                    dir = 0

            img = cv2.flip(img, 1)
            
            if (per >= 95 or per <= 5):
                cv2.putText(img, f'{int(per)}%', (1200, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            else:
                cv2.putText(img, f'{int(per)}%', (1200, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
            cv2.rectangle(img, (1200, 100), (1275, 650), (0, 0, 255), 3)
            cv2.rectangle(img, (1200, int(bar)), (1275, 650), (0, 0, 255), cv2.FILLED)
    else:
        img = cv2.flip(img, 1)
        cv2.rectangle(img, (430, 740), (1335, 620), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, 'Take your position.', (440, 710), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
        
    cv2.rectangle(img, (12, 6), (425, 100), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'count: {int(count)}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
            
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 730), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)
    if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
        break
cap.release()
cv2.destroyAllWindows()
