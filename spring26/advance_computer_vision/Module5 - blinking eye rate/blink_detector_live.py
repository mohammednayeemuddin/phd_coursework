"""Real-time Eye Blink Rate Detector - Live Camera"""
import cv2
import mediapipe as mp
import numpy as np
import time

# Eye landmark indices for MediaPipe Face Mesh
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def get_ear(landmarks, eye_indices, w, h):
    """Eye Aspect Ratio - drops when eye closes"""
    pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in eye_indices])
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    h1 = np.linalg.norm(pts[0] - pts[3])
    return (v1 + v2) / (2.0 * h1)

def main():
    mp_face = mp.solutions.face_mesh
    cap = cv2.VideoCapture(0)
    
    blinks = 0
    prev_closed = False
    start_time = time.time()
    threshold = 0.2  # EAR threshold for blink
    
    print("Starting camera... Press 'q' to quit")
    
    with mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True) as face:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)  # Mirror
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face.process(rgb)
            
            ear = 0
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                ear = (get_ear(lm, LEFT_EYE, w, h) + get_ear(lm, RIGHT_EYE, w, h)) / 2
                
                # Detect blink (EAR drops below threshold)
                closed = ear < threshold
                if closed and not prev_closed:
                    blinks += 1
                prev_closed = closed
                
                # Draw eye landmarks
                for idx in LEFT_EYE + RIGHT_EYE:
                    x, y = int(lm[idx].x * w), int(lm[idx].y * h)
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            
            # Stats overlay
            elapsed = time.time() - start_time
            rate = blinks / elapsed if elapsed > 0 else 0
            
            cv2.putText(frame, f"Blinks: {blinks}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Time: {elapsed:.1f}s", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Rate: {rate:.2f} blinks/sec", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.imshow("Blink Detector - Press Q to quit", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final stats
    total_time = time.time() - start_time
    print(f"\n=== Final Results ===")
    print(f"Total blinks: {blinks}")
    print(f"Duration: {total_time:.1f} seconds")
    print(f"Blink rate: {blinks/total_time:.3f} blinks/sec")
    print(f"Blink rate: {blinks/total_time * 60:.1f} blinks/min")

if __name__ == "__main__":
    main()
