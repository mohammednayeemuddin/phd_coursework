"""Eye Blink Rate Detector - ~40 lines core logic"""
import cv2
import mediapipe as mp
import numpy as np

def get_ear(landmarks, eye_indices, w, h):
    """Eye Aspect Ratio - key metric for blink detection"""
    pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in eye_indices])
    # Vertical distances
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    # Horizontal distance
    h1 = np.linalg.norm(pts[0] - pts[3])
    return (v1 + v2) / (2.0 * h1)

def count_blinks(video_path, threshold=0.2):
    """Count blinks in a video"""
    # Eye landmark indices for MediaPipe Face Mesh
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    
    mp_face = mp.solutions.face_mesh
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    blinks, frames, prev_closed = 0, 0, False
    
    with mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True) as face:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames += 1
            h, w = frame.shape[:2]
            results = face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                ear = (get_ear(lm, LEFT_EYE, w, h) + get_ear(lm, RIGHT_EYE, w, h)) / 2
                
                closed = ear < threshold
                if closed and not prev_closed:
                    blinks += 1
                prev_closed = closed
    
    cap.release()
    duration = frames / fps if fps > 0 else 1
    print(f"Blinks: {blinks} | Duration: {duration:.1f}s | Rate: {blinks/duration:.2f} blinks/sec")
    return blinks, duration, blinks/duration

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        count_blinks(sys.argv[1])
    else:
        print("Usage: python blink_detector.py <video_path>")
