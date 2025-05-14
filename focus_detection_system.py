import cv2
from deepface import DeepFace
from gaze_tracking import GazeTracking
import time
import subprocess

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
gaze = GazeTracking()

# Configuration
TRACKING_THRESHOLD = 50
MIN_FACE_SIZE = 100
MOOD_DURATION = 3
ALERT_INTERVAL = 5

def draw_fancy_box(frame, x, y, w, h):
    corner_length = min(w,h) // 4
    cv2.rectangle(frame, (x, y), (x+w, y+h), (45, 255, 255), 2)
    corners = [
        ((x, y), (x+corner_length, y), (x, y+corner_length)),
        ((x+w, y), (x+w-corner_length, y), (x+w, y+corner_length)),
        ((x, y+h), (x+corner_length, y+h), (x, y+h-corner_length)),
        ((x+w, y+h), (x+w-corner_length, y+h), (x+w, y+h-corner_length))
    ]
    for main, h_line, v_line in corners:
        cv2.line(frame, main, h_line, (0, 255, 0), 3)
        cv2.line(frame, main, v_line, (0, 255, 0), 3)
        
def analyze_emotion(face_roi):
    try:
        result = DeepFace.analyze(face_roi, actions=['emotion'], 
                                enforce_detection=False, silent=True)
        return result[0]['dominant_emotion'].capitalize()
    except Exception:
        return "Unknown"

def get_gaze_direction(gaze):
    directions = []
    if gaze.is_left():
        directions.append("Left")
    if gaze.is_right():
        directions.append("Right")
    if gaze.is_down():
        directions.append("Down")
    if gaze.is_blinking():
        directions.append("Blink")
    return directions[0] if directions else "Center"

def main():
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    face_tracker = {}
    next_face_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Gaze tracking
        gaze.refresh(frame)
        gaze_available = gaze.pupil_left_coords() and gaze.pupil_right_coords()
        gaze_status = "Track" if gaze_available else "Lost"
        gaze_direction = get_gaze_direction(gaze)
        
        # Face detection
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE))
        
        # Update face tracker
        current_ids = []
        for (x, y, w, h) in faces:
            best_match = None
            min_distance = float('inf')
            
            for fid, data in face_tracker.items():
                fx, fy, fw, fh = data['pos']
                distance = ((x - fx)**2 + (y - fy)**2)**0.5
                size_diff = abs(w - fw) + abs(h - fh)
                
                if distance < TRACKING_THRESHOLD and size_diff < 100:
                    if distance < min_distance:
                        min_distance = distance
                        best_match = fid
            
            if best_match is not None:
                current_ids.append(best_match)
                face_tracker[best_match]['pos'] = (x, y, w, h)
            else:
                current_ids.append(next_face_id)
                face_tracker[next_face_id] = {
                    'pos': (x, y, w, h),
                    'mood_start': time.time(),
                    'current_mood': 'Focus',
                    'last_mood': 'Focus'
                }
                next_face_id += 1

        # Process each tracked face
        for fid in list(face_tracker.keys()):
            if fid not in current_ids:
                del face_tracker[fid]
                continue
                
            data = face_tracker[fid]
            x, y, w, h = data['pos']
            
            # Emotion analysis
            face_roi = frame[y:y+h, x:x+w]
            emotion = analyze_emotion(face_roi)
            
            # Determine temporary mood with emotion priority
            if not gaze_available:
                    temp_mood = "Sleepy"
            else: 
                if emotion.lower() in ['angry']:
                    temp_mood = "Distracted"
                elif gaze_direction in ("Down", "Blink"):
                    temp_mood = "Distracted"
                else:
                    temp_mood = "Focus"

            # Mood persistence logic
            if temp_mood != data['current_mood']:
                data['current_mood'] = temp_mood
                data['mood_start'] = time.time()
            
            # Calculate final mood
            mood_duration = time.time() - data['mood_start']
            if mood_duration >= MOOD_DURATION:
                final_mood = temp_mood
            else:
                final_mood = "Focus"
            
            # Continuous Voice alerts
            current_time = time.time()
            alert_cooldown = current_time - data.get('last_alert_time', 0)

            if final_mood in ["Sleepy", "Distracted"]:
                if alert_cooldown >= ALERT_INTERVAL:
                    if final_mood == "Sleepy":
                        subprocess.Popen(['say', 'Please wake up'])
                    else:
                        subprocess.Popen(['say', 'You\'re distracted'])
                    data['last_alert_time'] = current_time
            else:
                # Reset alert timer when focused
                if 'last_alert_time' in data:
                    del data['last_alert_time']
            
            # Display elements
            draw_fancy_box(display_frame, x, y, w, h)

            font = cv2.FONT_HERSHEY_SIMPLEX
            line_height = 25
            text_lines = [
                f"Gaze: {gaze_status}",
                f"Direction: {gaze_direction}",
                f"Emotion: {emotion}",
                f"Temp Mood: {temp_mood}",
                f"Final Mood: {final_mood}"
            ]
            # Define a BGR color for each line
            colors = [
                (255, 0, 0),     # Blue for "Gaze"
                (0, 255, 0),     # Green for "Direction"
                (0, 0, 255),     # Red for "Emotion"
                (0, 200, 255),   # Orange for "Temp Mood"
                (255, 255, 255)  # White for "Final Mood"
            ]

            start_y = max(y - (len(text_lines) * line_height) - 10, 0)
            for i, text in enumerate(text_lines):
                cv2.putText(display_frame, text, (x, start_y + i*line_height),
                           font, 0.7, colors[i], 2, cv2.LINE_AA)

        if gaze_available:
            left_pupil = gaze.pupil_left_coords()  
            right_pupil = gaze.pupil_right_coords()
            for pupil in [left_pupil, right_pupil]:
                if pupil is not None:
                     x, y = pupil
                     cv2.line(display_frame, (x-5, y), (x+5, y), (0, 255, 0), 2)
                     cv2.line(display_frame, (x, y-5), (x, y+5), (0, 255, 0), 2)

        cv2.imshow('Emotion & Attention Monitor', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
