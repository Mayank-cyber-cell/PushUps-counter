import streamlit as st
import cv2
import numpy as np
from push_up_counter import poseDetector

st.title("💪 Push-Up Counter Web App")
st.markdown("Allow camera, stand in frame and start doing pushups!")

FRAME_WINDOW = st.image([])
run = st.checkbox('Start Camera')

cap = cv2.VideoCapture(0)
detector = poseDetector(detectionCon=0.8)
count, dir = 0, 0

while run:
    success, img = cap.read()
    if not success:
        st.warning("Cannot access camera.")
        break

    img = cv2.resize(img, (640, 480))
    img = detector.findPose(img, draw=False)
    lmList = detector.findPosition(img, draw=False)

    if lmList:
        if lmList[31][2] + 50 > lmList[29][2] and lmList[32][2] + 50 > lmList[30][2]:
            angle = detector.findAngle(img, 11, 13, 15)
            per = -1.25 * angle + 212.5
            per = max(0, min(100, per))

            if per >= 95 and dir == 0:
                count += 0.5
                dir = 1
            if per <= 5 and dir == 1:
                count += 0.5
                dir = 0

            cv2.putText(img, f'{int(per)}%', (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    img = cv2.flip(img, 1)
    cv2.putText(img, f'Count: {int(count)}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    FRAME_WINDOW.image(img, channels="BGR")

else:
    cap.release()
