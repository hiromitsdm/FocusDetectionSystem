# Focus Detection System

This is the library for Real-Time Mood Detector with Eye Tracking for Focus Monitoring.
![Demo](https://youtube.com/shorts/A5zI8s_M-k4?feature=share)

<div style="padding:120.13% 0 0 0;position:relative;"><iframe src="https://player.vimeo.com/video/1084127194?title=0&amp;byline=0&amp;portrait=0&amp;badge=0&amp;autopause=0&amp;player_id=0&amp;app_id=58479" frameborder="0" allow="autoplay; fullscreen; picture-in-picture; clipboard-write; encrypted-media" style="position:absolute;top:0;left:0;width:100%;height:100%;" title="demo"></iframe></div><script src="https://player.vimeo.com/api/player.js"></script>

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
