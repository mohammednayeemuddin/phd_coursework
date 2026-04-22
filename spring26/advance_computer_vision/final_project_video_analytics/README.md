# Facial Analysis Pipeline

**Multithreaded eye blink rate estimation and face dimension measurement from large video recordings.**

Built for a Computer Vision course assignment — processes 7 GB+ recordings fast by splitting the video into parallel worker chunks, each running its own OpenCV Haar cascade detector. No external model downloads required.

---

## What it does

| Task | Method |
|---|---|
| **Blink rate** (blinks/sec, /min, /hr) | Eye Aspect Ratio proxy via eye bounding-box height ÷ width |
| **Face dimensions** (head→chin, ear→ear) | Haar cascade face bounding box |
| **Eye dimensions** (W × H, left & right) | Eye sub-cascade bounding box inside face ROI |
| **Nose & mouth dimensions** | Estimated from golden-ratio face proportions |
| **Per-minute blink heatmap** | Aggregated from per-frame blink events |
| **Annotated preview clip** | First 60 s with face box, eye regions, EAR readout, blink flash |

---

## Requirements

Python 3.8+ and two pip packages:

```bash
pip install opencv-python numpy
```

> **No MediaPipe, no dlib, no model file downloads.** Everything uses OpenCV's built-in Haar cascade XML files that ship with `opencv-python`.

---

## Quick start

```bash
python3 analyze_video.py --video recording.mp4
```

That's it. The script auto-detects CPU count and sets workers accordingly.

---

## Full usage

```
python3 analyze_video.py  --video    <path>   \
                          --workers  <N>      \
                          --skip     <N>      \
                          --out      <dir>    \
                         [--no-clip]          \
                         [--clip-duration <s>]
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--video` | *(required)* | Path to input video file (MP4, MOV, AVI, MKV …) |
| `--workers` | `min(8, 2×CPU)` | Number of parallel worker threads |
| `--skip` | `3` | Analyse every Nth frame. `3` = 3× faster, fine for blink detection since eyes close for ≥100 ms |
| `--out` | `results/` | Output directory (created if it doesn't exist) |
| `--no-clip` | off | Skip generating the annotated preview clip |
| `--clip-duration` | `60.0` | Length of the annotated preview clip in seconds |

### Recommended settings by file size

| Video size | Resolution | `--workers` | `--skip` | Expected throughput |
|---|---|---|---|---|
| < 1 GB | 1080p | 4 | 2 | ~200 fps |
| 1–4 GB | 1080p / 4K | 8 | 3 | ~150 fps |
| 4–8 GB | 4K (3840×2160) | 12 | 3 | ~150–300 fps |
| > 8 GB | 4K | 16 | 4 | ~300 fps |

> At 4K resolution, frames are automatically downscaled to 960 px wide before cascade detection (~16× fewer pixels to process). Bounding boxes are then scaled back to original coordinates, so all reported measurements are in original pixel units.

---

## Example — the 6.74 GB recording

```bash
python3 analyze_video.py \
  --video  OneDrive_1_22-4-2026/output.mp4 \
  --workers 12 \
  --skip    3 \
  --out     results/
```

Expected console output:

```
01:50:04 │ INFO │ MainThread │ Video    : OneDrive_1_22-4-2026/output.mp4
01:50:04 │ INFO │ MainThread │ Size     : 6.74 GB
01:50:04 │ INFO │ MainThread │ Frames   : 35978  │  FPS: 29.970  │  Duration: 00:20:00
01:50:04 │ INFO │ MainThread │ Res      : 3840 × 2160
01:50:04 │ INFO │ MainThread │ Workers  : 12  │  Skip: every 3 frames
01:50:04 │ INFO │ MainThread │ Chunks   : 12  (~2999 frames each)
         ...
═════════════════════════════════════════════════════════════════
  FACIAL ANALYSIS  —  RESULTS
═════════════════════════════════════════════════════════════════
  Video duration          : 00:20:00
  Frames w/ face          : 8,241 / 11,993  (68.7%)

  ─── BLINK RATE ────────────────────────────────────────
  Total blinks            : 274
  Blinks / second         : 0.02283
  Blinks / minute         : 13.70
  Blinks / hour           : 822.0
  Avg inter-blink gap     : 4.377 s  (σ=2.103)

  ─── FACE DIMENSIONS (pixels, median) ──────────────────
  Face height (head→chin) : 812.0 px
  Face width  (ear→ear)   : 654.0 px
  Left  eye   W × H       : 148.0 × 52.0 px
  Right eye   W × H       : 144.0 × 50.0 px
  Est. nose   W × H       : 229.0 × 244.0 px
  Est. mouth  W × H       : 327.0 × 65.0 px
═════════════════════════════════════════════════════════════════
```

---

## Output files

All three files are written to `--out` (default `results/`):

### `facial_analysis_results.json`

The primary results file. Load this into the HTML dashboard.

```jsonc
{
  "summary": {
    "video_duration_hms": "00:20:00",
    "total_frames_analysed": 11993,
    "frames_with_face": 8241,
    "detection_rate_pct": 68.72
  },
  "blink": {
    "total_blinks": 274,
    "blinks_per_second": 0.02283,
    "blinks_per_minute": 13.70,
    "blinks_per_hour": 822.0,
    "avg_inter_blink_s": 4.377,
    "std_inter_blink_s": 2.103,
    "ear_threshold": 0.18,
    "per_minute_counts": { "0": 12, "1": 15, "2": 11, ... },
    "blink_timestamps_first200": [3.2, 7.6, 12.1, ...]
  },
  "face_dimensions_pixels": {
    "face_height_px":   { "median": 812.0, "mean": 809.3, "std": 22.1 },
    "face_width_px":    { "median": 654.0, "mean": 651.7, "std": 18.4 },
    "left_eye_width":   { "median": 148.0, "mean": 146.2 },
    "left_eye_height":  { "median": 52.0,  "mean": 51.4  },
    "right_eye_width":  { "median": 144.0, "mean": 142.8 },
    "right_eye_height": { "median": 50.0,  "mean": 49.6  },
    "est_nose_width":   { "median": 229.0, "mean": 228.1 },
    "est_nose_height":  { "median": 244.0, "mean": 242.9 },
    "est_mouth_width":  { "median": 327.0, "mean": 325.7 },
    "est_mouth_height": { "median": 65.0,  "mean": 64.8  }
  },
  "ear_timeline": [
    { "t": 0.0, "ear": 0.3812, "blink": false },
    ...
  ],
  "processing": {
    "workers": 12,
    "frame_skip": 3,
    "elapsed_seconds": 87.4,
    "effective_fps": 412.0
  }
}
```

### `frame_data.csv`

One row per analysed frame. Useful for custom analysis in pandas / Excel.

| Column | Description |
|---|---|
| `frame_idx` | Original frame number in the video |
| `timestamp_s` | Time in seconds |
| `face_detected` | `True` / `False` |
| `left_eye_ear` | EAR proxy for left eye |
| `right_eye_ear` | EAR proxy for right eye |
| `avg_ear` | Average of both eyes |
| `left_eye_open` | Whether left eye cascade found a detection |
| `right_eye_open` | Whether right eye cascade found a detection |
| `blink_event` | `True` = a blink started on this frame |
| `face_x/y/w/h` | Face bounding box in original pixel coordinates |
| `left_eye_w/h` | Left eye bbox dimensions (px) |
| `right_eye_w/h` | Right eye bbox dimensions (px) |
| `est_nose_w/h` | Estimated nose dimensions (px) |
| `est_mouth_w/h` | Estimated mouth dimensions (px) |

### `annotated_sample.mp4`

First 60 seconds of the video with overlays:
- **Green box** — face bounding box with pixel dimensions
- **Blue box** — right eye region
- **Orange box** — left eye region
- **EAR readout** — current Eye Aspect Ratio value
- **Blue flash + "BLINK" text** — blink event frames
- **Timestamp** — bottom-left corner

Output is automatically downscaled to 1920×1080 max, regardless of source resolution.

---

## HTML Dashboard

Open `facial_analysis_dashboard.html` in any browser, then drag-and-drop `facial_analysis_results.json` onto it.

**Dashboard panels:**

- KPI cards — total blinks, blinks/sec, blinks/min, blinks/hr, processing speed
- EAR timeline chart — full-session Eye Aspect Ratio with blink threshold line
- Per-minute blink bar chart — distribution over the session
- Face geometry table — all dimensions (median values)
- Feature radar chart — normalised comparison of all face features
- Blink activity heatmap — colour-coded per-minute grid

No server needed — it's a single self-contained HTML file.

---

## How it works

### Threading architecture

```
main()
 ├── Probe video (fps, frame count, resolution)
 ├── Partition frames into N equal chunks by frame index
 ├── Spawn N VideoChunkWorker threads
 │    Each worker:
 │      ├── Opens its own cv2.VideoCapture (thread-safe: separate file handles)
 │      ├── Creates its own CascadeClassifier instances (NOT thread-safe to share)
 │      ├── Seeks to start_frame with CAP_PROP_POS_FRAMES
 │      ├── Reads every `skip` frames → detect face → detect eyes → EAR → blink state machine
 │      └── Posts ChunkResult to a thread-safe queue.Queue
 ├── ProgressPrinter daemon thread — logs % / ETA every 10 s
 └── Collect all ChunkResults → merge in frame order → write JSON + CSV + clip
```

### Blink detection (EAR state machine)

```
EAR_proxy = eye_bbox_height / eye_bbox_width

For each analysed frame:
  eye_closed = (EAR_proxy < 0.18) OR (both eye cascades miss inside face ROI)

  if eye_closed:
      closed_streak += 1
  else:
      if closed_streak >= 2:          # minimum 2 consecutive closed frames
          mark first frame as BLINK
      closed_streak = 0
```

A normal blink lasts ~150–400 ms. At 30 fps with skip=3, each analysed frame represents ~100 ms — so a blink produces 2–4 consecutive closed detections.

### 4K speed optimisation

Running a Haar cascade on a 3840×2160 frame takes ~16× longer than on a 960px-wide frame. The pipeline automatically downscales to `DETECT_MAX_W = 960` px before face detection, then scales bounding boxes back to original coordinates:

```python
scale_down = 960 / 3840 = 0.25
gray_small = cv2.resize(gray, (960, 540))   # detect on this
faces      = face_cascade.detectMultiScale(gray_small, ...)
# scale back:
fx, fy, fw, fh = [int(v / scale_down) for v in raw_bbox]
# eye detection still runs on original-res face ROI for accurate measurements
```

### Face dimension methodology

| Measurement | Method |
|---|---|
| Face height | Haar face bbox height (forehead to chin) |
| Face width | Haar face bbox width (roughly ear to ear) |
| Eye width / height | Eye sub-cascade bbox inside top-60% of face ROI |
| Nose width | 35% of face bbox width (average human proportion) |
| Nose height | 30% of face bbox height |
| Mouth width | 50% of face bbox width |
| Mouth height | 8% of face bbox height |

All values reported as **median** across all detected frames (robust to outliers from false positives).

---

## Tuning parameters

Edit these constants at the top of `analyze_video.py`:

```python
BLINK_EAR_THRESHOLD = 0.18   # lower → fewer blinks detected; raise if missing blinks
BLINK_CONSEC_FRAMES = 2      # raise to 3 if getting false positives from head movement
CASCADE_SCALE       = 1.15   # lower (e.g. 1.05) = slower but catches smaller faces
CASCADE_NEIGHBOURS  = 4      # raise to reduce false positives
MIN_FACE_PX         = 60     # minimum face size in pixels to consider
DETECT_MAX_W        = 960    # downscale width for cascade detection (affects speed)
```

---

## Troubleshooting

**`ValueError: truth value of array is ambiguous`**
You have an older version of the script. The fix is using `if bbox is not None` instead of `if bbox` for numpy arrays returned by `detectMultiScale`. Download the latest version.

**Low face detection rate (< 30%)**
- Subject may be looking away, in low light, or at an angle
- Try lowering `CASCADE_SCALE` to `1.05` and `CASCADE_NEIGHBOURS` to `3`
- Add lighting or ensure the camera is roughly face-level

**Blink count seems too high / too low**
- Too high: raise `BLINK_EAR_THRESHOLD` to `0.22` or `BLINK_CONSEC_FRAMES` to `3`
- Too low: lower `BLINK_EAR_THRESHOLD` to `0.14` or `BLINK_CONSEC_FRAMES` to `1`
- Normal human blink rate is 12–20 blinks/minute at rest; drops during focused study

**Processing is slow**
- Increase `--skip` (3 → 4 or 5) — each +1 reduces work by 25%
- Increase `--workers` up to your physical CPU count
- Add `--no-clip` to skip the annotated clip generation

**`unrecognized arguments: skip 3`**
Missing `--` prefix. Correct syntax: `--skip 3` (two dashes).

---

## Project structure

```
video_analysis_final/
├── analyze_video.py              ← main pipeline script
├── facial_analysis_dashboard.html ← browser dashboard (drag-drop JSON)
├── README.md                     ← this file
└── results/                      ← created on first run
    ├── facial_analysis_results.json
    ├── frame_data.csv
    └── annotated_sample.mp4
```

---

## Assignment context

This tool was built for **Step 2** of the Computer Vision in-class assignment:

> **(A)** Estimate eye blinking rate (blinks per second) during a 4-hour study session recording.
>
> **(B)** Estimate the dimensions of eyes, face (head-to-chin & ear-to-ear), nose, and mouth.

The pipeline processes the full recording without requiring any calibration, external model files, or GPU.

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `opencv-python` | ≥ 4.5 | Video I/O, Haar cascades, image processing |
| `numpy` | ≥ 1.21 | Array operations, statistics |

Standard library only beyond those two: `threading`, `queue`, `csv`, `json`, `argparse`, `dataclasses`, `pathlib`.
