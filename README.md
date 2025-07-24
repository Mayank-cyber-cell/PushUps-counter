# Push-Up Counter with Pose Detection

This project uses **MediaPipe** and **OpenCV** to build an intelligent push-up counter that uses **pose estimation** to track body joints and count reps in real time. It's an excellent example of how computer vision can be used in fitness and health applications.

---

## 📦 File Included

### "push_up_counter.py"
This single file handles everything:
- Captures webcam feed
- Detects human pose
- Monitors joint angles
- Converts elbow angle to progress percentage
- Counts reps
- Visualizes feedback

---

## 🧠 How the Project Works (Detailed Code Walkthrough)

### 1. Import Libraries
```python
import cv2
import numpy as np
import time
import pose as pem
```
- `cv2`: OpenCV for image processing and webcam access
- `numpy`: Used for interpolating bar height
- `time`: To measure FPS
- `pose`: A helper module (`pose.py`) that detects body pose and calculates angles

### 2. Setup Video and Variables
```python
cap = cv2.VideoCapture(0)
pTime = 0

detector = pem.poseDetector(detectionCon=0.8)
count, dir, bar, per = 0, 0, 0, 0
```
- Open the webcam
- Initialize the pose detector with higher accuracy
- Initialize the push-up rep counter (`count`), movement direction flag (`dir`), and progress values (`per`, `bar`)

### 3. Start Webcam Loop
```python
while True:
    success, img = cap.read()
    img = cv2.resize(img, (1366, 780))
```
- Read and resize each frame to a fixed resolution

### 4. Pose Estimation and Landmark Extraction
```python
    img = detector.findPose(img, draw=False)
    lmList = detector.findPosition(img, draw=False)
```
- Detect body pose
- Extract landmarks (joint positions) as a list

### 5. Check Body Position (Legs Extended)
```python
    if (lmList[31][2] + 50 > lmList[29][2] and lmList[32][2] + 50 > lmList[30][2]):
```
- Ensure ankles are below knees (to confirm proper push-up position)

### 6. Calculate Angles
```python
        angle = detector.findAngle(img, 11, 13, 15)  # Left arm
        detector.findAngle(img, 12, 14, 16)           # Right arm
        detector.findAngle(img, 27, 29, 31)           # Left leg
        detector.findAngle(img, 28, 30, 32)           # Right leg
```
- Calculate elbow and knee angles

### 7. Convert Angle to Percentage
```python
        per = -1.25 * angle + 212.5
        per = (0 if per < 0 else 100 if per > 100 else per)
```
- Maps elbow angle to push-up progress percentage (0% up, 100% down)

### 8. Map Percentage to Bar Height
```python
        bar = np.interp(per, (0, 100), (650, 100))
```
- Converts progress to bar height on screen

### 9. Count Push-Ups
```python
        if per >= 95:
            if dir == 0:
                count += 0.5
                dir = 1
        elif per <= 5:
            if dir == 1:
                count += 0.5
                dir = 0
```
- Toggle `dir` to ensure a full push-up cycle is counted
- Adds 0.5 when going down and 0.5 when coming back up

### 10. Visual Feedback (Flipped Image)
```python
        img = cv2.flip(img, 1)
```
- Flip image like a mirror for natural orientation

### 11. Show Percentage and Bar
```python
        if (per == 100 or per == 0):
            cv2.putText(... green)
        else:
            cv2.putText(... red)
        cv2.rectangle(... bar background)
        cv2.rectangle(... bar fill)
```
- Show percentage and fill the bar according to progress

### 12. Else: Prompt to Get in Position
```python
    else:
        img = cv2.flip(img, 1)
        cv2.rectangle(... 'Take your position')
```
- Display a message if the user is not in proper pose

### 13. Display Count and FPS
```python
    cv2.rectangle(... count)
    cv2.putText(... count)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(... fps)
```
- Display total count and frame rate

### 14. Show Image
```python
    cv2.imshow("Image", img)
    cv2.waitKey(1)
```
- Display the final result with all visuals

---

## ✅ Project Summary
| Feature | Description |
|--------|-------------|
| Library | MediaPipe, OpenCV, NumPy |
| Input | Live webcam video feed |
| Output | Pose landmarks, bar chart, rep count |
| ML Model | MediaPipe Pose Estimation |
| Applications | Fitness, Rehab, Posture Correction |

---

## 🚀 Future Ideas
- Add sound feedback for each rep
- Save history of workout sessions
- Detect poor form (e.g., bent knees)
- Export stats (CSV, Graphs)

----



