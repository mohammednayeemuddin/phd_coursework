"""Eye Blink Rate Detector - Video File Version"""
import cv2
import mediapipe as mp
import numpy as np
import sys

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

def analyze_video(video_path, show_preview=True):
    mp_face = mp.solutions.face_mesh
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video '{video_path}'")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Video: {video_path}")
    print(f"FPS: {fps:.1f} | Frames: {total_frames} | Duration: {duration:.1f}s")
    print("Processing... Press 'q' to skip preview\n")
    
    blinks = 0
    prev_closed = False
    threshold = 0.2
    frame_count = 0
    
    with mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True) as face:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face.process(rgb)
            
            ear = 0
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                ear = (get_ear(lm, LEFT_EYE, w, h) + get_ear(lm, RIGHT_EYE, w, h)) / 2
                
                closed = ear < threshold
                if closed and not prev_closed:
                    blinks += 1
                prev_closed = closed
                
                # Draw eye landmarks
                if show_preview:
                    for idx in LEFT_EYE + RIGHT_EYE:
                        x, y = int(lm[idx].x * w), int(lm[idx].y * h)
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            
            if show_preview:
                # Progress & stats overlay
                progress = frame_count / total_frames * 100
                current_time = frame_count / fps
                rate = blinks / current_time if current_time > 0 else 0
                
                cv2.putText(frame, f"Blinks: {blinks}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Time: {current_time:.1f}s / {duration:.1f}s", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Rate: {rate:.2f} blinks/sec", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Progress: {progress:.0f}%", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                cv2.imshow("Blink Detector - Press Q to skip", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    show_preview = False
                    cv2.destroyAllWindows()
                    print("Preview skipped, continuing analysis...")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final results
    rate_per_sec = blinks / duration if duration > 0 else 0
    rate_per_min = rate_per_sec * 60
    
    print(f"\n{'='*40}")
    print(f"RESULTS: {video_path}")
    print(f"{'='*40}")
    print(f"Total blinks:    {blinks}")
    print(f"Duration:        {duration:.1f} seconds")
    print(f"Blink rate:      {rate_per_sec:.3f} blinks/sec")
    print(f"Blink rate:      {rate_per_min:.1f} blinks/min")
    print(f"{'='*40}\n")
    
    return {
        'video': video_path,
        'blinks': blinks,
        'duration': duration,
        'rate_per_sec': rate_per_sec,
        'rate_per_min': rate_per_min
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python blink_detector_video.py <video_path> [video_path2 ...]")
        print("\nExample:")
        print("  python blink_detector_video.py movie_watching.mp4")
        print("  python blink_detector_video.py movie.mp4 reading.mp4")
        sys.exit(1)
    
    results = []
    for video in sys.argv[1:]:
        result = analyze_video(video)
        if result:
            results.append(result)
    
    # Comparison if multiple videos
    if len(results) > 1:
        print("\n" + "="*50)
        print("COMPARISON")
        print("="*50)
        for r in results:
            print(f"{r['video']}: {r['rate_per_min']:.1f} blinks/min")
        print("="*50)
