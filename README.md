# Focus_Detection_System

This is the library for Real-Time Mood Detector with Eye Tracking for Focus Monitoring.
![Demo](assets/demo.gif)]

## Description
The system utilizes a standard computer webcam and integrates three open-source software tools: OpenCV for face detection, DeepFace for emotion analysis, and GazeTracking for eye movement and gaze direction measurement. The webcam captures a video stream of the driver’s face and eyes. OpenCV’s Haar cascade classifier locates the face, DeepFace’s pre-trained model classifies emotional states (happy, sad, angry, etc.), and GazeTracking determines gaze direction and fixation. By fusing gaze metrics and emotion data, the system classifies the driver’s state as “focused,” “distracted,” or “sleepy.” If distraction or drowsiness persists for a preset threshold (e.g., two seconds), the system issues a real-time audio and visual alert to prompt corrective action.


## For Pip install
Install these dependencies (NumPy, OpenCV, Dlib):

```shell
pip install -r requirements.txt
```

## System Demo
Run the demo:

```shell
python focus_detection_system.py
```
