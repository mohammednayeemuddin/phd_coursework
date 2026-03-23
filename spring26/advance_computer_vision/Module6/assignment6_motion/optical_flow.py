"""
CSc 8830: Computer Vision - Assignment 6
Part A: Optical Flow Computation and Visualization

README:
    This script generates two synthetic videos with motion, computes
    Lucas-Kanade sparse optical flow on each, and saves the visualization
    as output videos. It also validates bilinear interpolation and
    tracking against actual pixel locations.

Usage:
    python optical_flow.py

Dependencies:
    pip install opencv-python-headless numpy matplotlib scipy

Output:
    - video1_raw.avi         : Synthetic video 1 (translating circle)
    - video2_raw.avi         : Synthetic video 2 (rotating shapes)
    - video1_flow.avi        : Optical flow visualization video 1
    - video2_flow.avi        : Optical flow visualization video 2
    - flow_frames_v1.png     : Frame comparison for video 1
    - flow_frames_v2.png     : Frame comparison for video 2
    - bilinear_validation.png: Bilinear interpolation validation
    - tracking_validation.png: Tracking validation results

References:
    - Lucas, B. D., & Kanade, T. (1981). An iterative image registration technique.
    - Bradski, G. (2000). The OpenCV Library. Dr. Dobb's Journal.
    - Szeliski, R. (2010). Computer Vision: Algorithms and Applications.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ─────────────────────────────────────────────
# 1. SYNTHETIC VIDEO GENERATION
# ─────────────────────────────────────────────

def generate_video1(path, n_frames=90, w=640, h=480):
    """
    Video 1: Translating bright circle on dark background.
    Motion: constant rightward + slight downward drift.
    """
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path, fourcc, 30.0, (w, h))
    cx, cy = 80, 240
    for i in range(n_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Moving circle
        x = int(cx + i * 5.5)
        y = int(cy + i * 1.2)
        cv2.circle(frame, (x % w, y), 40, (200, 200, 50), -1)
        # Static reference rectangle
        cv2.rectangle(frame, (20, 20), (80, 60), (100, 100, 200), -1)
        # Add mild Gaussian noise
        noise = np.random.normal(0, 5, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        out.write(frame)
    out.release()
    print(f"[✓] Video 1 saved: {path}")


def generate_video2(path, n_frames=90, w=640, h=480):
    """
    Video 2: Two objects — rotating square + bouncing ball.
    Demonstrates rotational and oscillating motion.
    """
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path, fourcc, 30.0, (w, h))
    cx, cy = 320, 240
    for i in range(n_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Rotating square (draw as polygon)
        angle = np.radians(i * 4)
        size = 60
        pts = np.array([
            [cx + size * np.cos(angle + k * np.pi / 2),
             cy + size * np.sin(angle + k * np.pi / 2)]
            for k in range(4)], dtype=np.int32)
        cv2.fillPoly(frame, [pts], (50, 180, 220))
        # Bouncing ball
        bx = int(100 + 200 * abs(np.sin(i * 0.1)))
        by = int(380 - 100 * abs(np.sin(i * 0.2)))
        cv2.circle(frame, (bx, by), 25, (220, 80, 80), -1)
        noise = np.random.normal(0, 4, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        out.write(frame)
    out.release()
    print(f"[✓] Video 2 saved: {path}")


# ─────────────────────────────────────────────
# 2. OPTICAL FLOW COMPUTATION (Lucas-Kanade)
# ─────────────────────────────────────────────

def compute_optical_flow(input_path, output_path, label="Video"):
    """
    Compute sparse Lucas-Kanade optical flow and render as overlay.
    Returns frame pair (f0, f1) and tracked points for validation.
    """
    cap = cv2.VideoCapture(input_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Shi-Tomasi corner detection params
    feature_params = dict(maxCorners=120, qualityLevel=0.3,
                          minDistance=7, blockSize=7)
    # Lucas-Kanade optical flow params
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                10, 0.03))

    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Store first frame pair for validation
    frame0_gray = old_gray.copy()
    frame1_gray = None
    p0_initial = p0.copy() if p0 is not None else None
    p1_tracked = None

    mask = np.zeros_like(old_frame)
    colors = np.random.randint(0, 255, (200, 3))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if p0 is not None and len(p0) > 0:
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, p0, None, **lk_params)
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # Save first pair tracking result
            if frame_idx == 0:
                frame1_gray = frame_gray.copy()
                p1_tracked = p1.copy()

            # Draw tracks
            for j, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel().astype(int)
                c, d = old.ravel().astype(int)
                mask = cv2.line(mask, (a, b), (c, d),
                                colors[j % 200].tolist(), 2)
                frame = cv2.circle(frame, (a, b), 4,
                                   colors[j % 200].tolist(), -1)

            img = cv2.add(frame, mask)

            # Update
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
        else:
            img = frame
            old_gray = frame_gray.copy()
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
            mask = np.zeros_like(frame)

        # Label overlay
        cv2.putText(img, f"{label} | Frame {frame_idx+1}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        out.write(img)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"[✓] Optical flow video saved: {output_path}")
    return frame0_gray, frame1_gray, p0_initial, p1_tracked


# ─────────────────────────────────────────────
# 3. BILINEAR INTERPOLATION VALIDATION
# ─────────────────────────────────────────────

def bilinear_interpolate(img, x, y):
    """
    Bilinear interpolation at sub-pixel (x, y).
    I(x,y) = (1-a)(1-b)*I(x0,y0) + a(1-b)*I(x1,y0)
            + (1-a)*b*I(x0,y1) + a*b*I(x1,y1)
    where a = x-x0, b = y-y0
    """
    x0, y0 = int(x), int(y)
    x1, y1 = min(x0 + 1, img.shape[1] - 1), min(y0 + 1, img.shape[0] - 1)
    a = x - x0   # fractional part x
    b = y - y0   # fractional part y

    val = ((1 - a) * (1 - b) * img[y0, x0] +
            a       * (1 - b) * img[y0, x1] +
           (1 - a) *  b       * img[y1, x0] +
            a       *  b       * img[y1, x1])
    return val


def validate_bilinear(frame0, out_path):
    """Compare bilinear interpolation vs OpenCV remap on a grid of sub-pixels."""
    h, w = frame0.shape
    test_x = np.array([50.3, 100.7, 200.25, 300.9, 400.1])
    test_y = np.array([60.8, 120.4, 180.75, 250.5, 360.2])

    manual_vals = [bilinear_interpolate(frame0, x, y)
                   for x, y in zip(test_x, test_y)]

    # OpenCV reference
    map_x = test_x.astype(np.float32).reshape(1, -1)
    map_y = test_y.astype(np.float32).reshape(1, -1)
    cv_vals = cv2.remap(frame0.astype(np.float32), map_x, map_y,
                        cv2.INTER_LINEAR).flatten()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot comparison
    pts = range(len(test_x))
    axes[0].plot(pts, manual_vals, 'bo-', label='Manual Bilinear', linewidth=2)
    axes[0].plot(pts, cv_vals, 'r^--', label='OpenCV INTER_LINEAR', linewidth=2)
    axes[0].set_title('Bilinear Interpolation Validation\n(Manual vs OpenCV)', fontsize=13)
    axes[0].set_xlabel('Test Point Index')
    axes[0].set_ylabel('Interpolated Intensity')
    axes[0].legend()
    axes[0].grid(True, alpha=0.4)

    labels = [f"({x:.1f},{y:.1f})" for x, y in zip(test_x, test_y)]
    errors = [abs(m - c) for m, c in zip(manual_vals, cv_vals)]
    axes[1].bar(pts, errors, color='steelblue', alpha=0.8)
    axes[1].set_xticks(pts)
    axes[1].set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    axes[1].set_title('Absolute Error |Manual − OpenCV|', fontsize=13)
    axes[1].set_ylabel('Error (intensity units)')
    axes[1].grid(True, alpha=0.4, axis='y')

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[✓] Bilinear validation plot saved: {out_path}")
    return list(zip(test_x, test_y, manual_vals, cv_vals, errors))


# ─────────────────────────────────────────────
# 4. TRACKING VALIDATION (Theory vs Actual)
# ─────────────────────────────────────────────

def validate_tracking(frame0, frame1, p0, p1_tracked, out_path, label="Video"):
    """
    Compare theoretically predicted position (translate by mean flow)
    vs LK-tracked actual positions for the first few feature points.
    """
    if p0 is None or p1_tracked is None or frame1 is None:
        print(f"[!] Skipping tracking validation for {label} (insufficient data)")
        return []

    N = min(8, len(p0), len(p1_tracked))
    pts0 = p0[:N].reshape(N, 2)
    pts1 = p1_tracked[:N].reshape(N, 2)

    # Theoretical prediction: use the median flow as "average motion"
    flow_vecs = pts1 - pts0
    mean_u = np.median(flow_vecs[:, 0])
    mean_v = np.median(flow_vecs[:, 1])
    pts_predicted = pts0 + np.array([mean_u, mean_v])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Frame 0 with initial points
    axes[0].imshow(frame0, cmap='gray')
    axes[0].scatter(pts0[:, 0], pts0[:, 1], c='yellow', s=60, zorder=5)
    axes[0].set_title(f'{label} — Frame t\nInitial feature points', fontsize=11)
    axes[0].axis('off')

    # Frame 1 with actual vs predicted
    axes[1].imshow(frame1, cmap='gray')
    axes[1].scatter(pts1[:, 0], pts1[:, 1],
                    c='lime', s=70, marker='o', label='LK Tracked', zorder=5)
    axes[1].scatter(pts_predicted[:, 0], pts_predicted[:, 1],
                    c='red', s=70, marker='^', label='Predicted (mean flow)', zorder=5)
    for i in range(N):
        axes[1].plot([pts1[i, 0], pts_predicted[i, 0]],
                     [pts1[i, 1], pts_predicted[i, 1]], 'w-', alpha=0.5)
    axes[1].legend(fontsize=9)
    axes[1].set_title(f'{label} — Frame t+1\nTracked vs Predicted', fontsize=11)
    axes[1].axis('off')

    # Error per point
    errors = np.linalg.norm(pts1 - pts_predicted, axis=1)
    axes[2].bar(range(N), errors, color='steelblue', alpha=0.85)
    axes[2].set_title(f'Tracking Error\n|Tracked − Predicted| (pixels)', fontsize=11)
    axes[2].set_xlabel('Feature Point Index')
    axes[2].set_ylabel('Euclidean Error (px)')
    axes[2].grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'{label} — Tracking Validation: Theory vs LK Result', fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[✓] Tracking validation plot saved: {out_path}")

    # Print table
    print(f"\n{'─'*70}")
    print(f"  {label} Tracking Validation Table")
    print(f"{'─'*70}")
    print(f"  {'Pt':>3} | {'x0':>7} {'y0':>7} | {'x1_lk':>8} {'y1_lk':>8} | {'x1_pred':>8} {'y1_pred':>8} | {'err_px':>7}")
    print(f"{'─'*70}")
    results = []
    for i in range(N):
        row = (pts0[i, 0], pts0[i, 1],
               pts1[i, 0], pts1[i, 1],
               pts_predicted[i, 0], pts_predicted[i, 1],
               errors[i])
        print(f"  {i:>3} | {row[0]:>7.2f} {row[1]:>7.2f} | {row[2]:>8.2f} {row[3]:>8.2f} | {row[4]:>8.2f} {row[5]:>8.2f} | {row[6]:>7.3f}")
        results.append(row)
    print(f"{'─'*70}\n  Mean error: {np.mean(errors):.3f} px\n")
    return results


# ─────────────────────────────────────────────
# 5. FLOW FRAME VISUALIZATION
# ─────────────────────────────────────────────

def save_flow_frame_comparison(video_path, out_path, label="Video"):
    """Extract frame 0 and frame 1, draw dense HSV optical flow between them."""
    cap = cv2.VideoCapture(video_path)
    ret, f0 = cap.read()
    ret, f1 = cap.read()
    cap.release()

    g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(g0, g1, None,
                                         0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros_like(f0)
    hsv[..., 1] = 255
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(cv2.cvtColor(f0, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'{label} — Frame t', fontsize=12)
    axes[0].axis('off')
    axes[1].imshow(cv2.cvtColor(f1, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'{label} — Frame t+1', fontsize=12)
    axes[1].axis('off')
    axes[2].imshow(flow_rgb)
    axes[2].set_title(f'{label} — Dense Optical Flow\n(HSV: hue=direction, brightness=magnitude)',
                      fontsize=11)
    axes[2].axis('off')
    fig.suptitle(f'{label} — Optical Flow Visualization', fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[✓] Flow frame comparison saved: {out_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    print("\n=== PART A: OPTICAL FLOW ===\n")

    # Generate synthetic videos
    generate_video1("output/video1_raw.avi")
    generate_video2("output/video2_raw.avi")

    # Compute optical flow
    f0_v1, f1_v1, p0_v1, p1_v1 = compute_optical_flow(
        "output/video1_raw.avi", "output/video1_flow.avi", label="Video 1")
    f0_v2, f1_v2, p0_v2, p1_v2 = compute_optical_flow(
        "output/video2_raw.avi", "output/video2_flow.avi", label="Video 2")

    # Dense flow frame comparisons
    save_flow_frame_comparison("output/video1_raw.avi",
                                "output/flow_frames_v1.png", "Video 1")
    save_flow_frame_comparison("output/video2_raw.avi",
                                "output/flow_frames_v2.png", "Video 2")

    # Bilinear interpolation validation
    if f0_v1 is not None:
        bilim_results = validate_bilinear(f0_v1,
                                           "output/bilinear_validation.png")

    # Tracking validation
    tracking_v1 = validate_tracking(f0_v1, f1_v1, p0_v1, p1_v1,
                                     "output/tracking_v1.png", "Video 1")
    tracking_v2 = validate_tracking(f0_v2, f1_v2, p0_v2, p1_v2,
                                     "output/tracking_v2.png", "Video 2")

    print("\n[✓] Part A complete. All outputs in ./output/\n")
