# CSc 8830: Computer Vision — Assignment 6
## Optical Flow and Structure from Motion

---

## Overview
This repository contains Python implementations for:
- **Part A**: Lucas-Kanade sparse optical flow, Farneback dense optical flow,
  bilinear interpolation validation, and motion tracking validation
- **Part B**: Structure from Motion (SfM) from 4 synthetic viewpoints —
  DLT triangulation, epipolar geometry, and 3D point reconstruction

---

## Repository Structure
```
├── optical_flow.py          # Part A: optical flow + tracking
├── structure_from_motion.py # Part B: SfM + reconstruction
├── generate_report.py       # Generates the full PDF report
├── output/                  # All generated outputs (created on run)
│   ├── video1_raw.avi
│   ├── video2_raw.avi
│   ├── video1_flow.avi
│   ├── video2_flow.avi
│   ├── flow_frames_v1.png
│   ├── flow_frames_v2.png
│   ├── bilinear_validation.png
│   ├── tracking_v1.png
│   ├── tracking_v2.png
│   ├── sfm_views.png
│   ├── sfm_reconstruction.png
│   ├── sfm_camera_setup.png
│   ├── sfm_epipolar.png
│   └── Assignment6_Report.pdf
└── README.md
```

---

## Dependencies
```bash
pip install opencv-python-headless numpy matplotlib scipy reportlab
```

---

## Usage

### Step 1 — Run Part A (Optical Flow)
```bash
python optical_flow.py
```
**Outputs:**
- `output/video1_raw.avi` — Synthetic video 1 (translating circle)
- `output/video2_raw.avi` — Synthetic video 2 (rotating square + bouncing ball)
- `output/video1_flow.avi` — Sparse LK flow overlay for video 1
- `output/video2_flow.avi` — Sparse LK flow overlay for video 2
- `output/flow_frames_v1.png` — Dense HSV flow visualization (video 1)
- `output/flow_frames_v2.png` — Dense HSV flow visualization (video 2)
- `output/bilinear_validation.png` — Manual vs OpenCV bilinear interpolation
- `output/tracking_v1.png` — Tracking validation for video 1
- `output/tracking_v2.png` — Tracking validation for video 2

### Step 2 — Run Part B (Structure from Motion)
```bash
python structure_from_motion.py
```
**Outputs:**
- `output/sfm_views.png` — Projected object in all 4 camera views
- `output/sfm_reconstruction.png` — 3D reconstruction + error plot
- `output/sfm_camera_setup.png` — Top-down camera arrangement diagram
- `output/sfm_epipolar.png` — Epipolar lines between cameras 1 and 2

### Step 3 — Generate PDF Report
```bash
python generate_report.py
```
**Output:**
- `output/Assignment6_Report.pdf` — Full academic report with all math derivations

---

## Mathematical Concepts Covered

### Part A
| Topic | Location |
|-------|----------|
| Optical Flow Constraint Equation (OFCE) | `optical_flow.py` + Report §A.3 |
| Lucas-Kanade derivation (normal equations) | Report §A.3 |
| Bilinear interpolation (full derivation) | `optical_flow.py` + Report §A.4 |
| Tracking prediction: p' = p + d | Report §A.3 |
| Validation tables (actual vs. predicted pixel) | Report §A.5 |

### Part B
| Topic | Location |
|-------|----------|
| Camera projection matrix P = K[R|t] | `structure_from_motion.py` + Report §B.1 |
| Rotation matrices (yaw/pitch) | Report §B.2 |
| DLT triangulation via SVD | `structure_from_motion.py` + Report §B.2 |
| Essential & Fundamental matrices | Report §B.2 |
| Epipolar constraint x2^T F x1 = 0 | Report §B.2 |

---

## References
1. Lucas & Kanade (1981). An iterative image registration technique.
2. Horn & Schunck (1981). Determining optical flow.
3. Hartley & Zisserman (2003). Multiple View Geometry in Computer Vision.
4. Faugeras (1993). Three-Dimensional Computer Vision.
5. Szeliski (2010). Computer Vision: Algorithms and Applications.
6. Bradski (2000). The OpenCV Library.
7. Shi & Tomasi (1994). Good features to track.
8. Farneback (2003). Two-frame motion estimation.
