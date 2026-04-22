#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         MULTITHREADED FACIAL ANALYSIS PIPELINE                             ║
║         Eye Blink Rate  +  Face / Feature Dimension Estimation             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Handles 7 GB+ recordings with parallel worker threads.                    ║
║  Self-contained: uses only OpenCV built-in Haar cascades — no downloads.   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Architecture                                                               ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Main thread                                                                ║
║   ├─ Probes video  (fps, total_frames, resolution)                         ║
║   ├─ Splits into N equal frame-range chunks                                ║
║   ├─ Spawns N WorkerThreads — each owns its own cascade objects            ║
║   │    read frame → detect face → detect eyes inside face ROI              ║
║   │    → EAR proxy → blink detector → bbox measurements                   ║
║   ├─ ProgressPrinter thread logs ETA every 10 s                            ║
║   └─ Merge → JSON  +  CSV  +  annotated sample clip                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  EAR Proxy (no 3D landmarks needed)                                        ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  EAR_proxy = eye_bbox_height / eye_bbox_width                              ║
║  Open eye ≈ 0.30-0.50   Closed / missing ≈ 0-0.18                         ║
║  Blink = EAR_proxy < threshold  OR  eye cascade misses for ≥ CONSEC frames ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Usage                                                                      ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  python3 analyze_video.py --video recording.mp4                            ║
║                           --workers 4   (default: 2×CPU, max 8)            ║
║                           --skip 3      (analyse every 3rd frame)          ║
║                           --out results/                                    ║
║                           --no-clip     (skip annotated preview clip)      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import multiprocessing
import os
import queue
import sys
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(threadName)-20s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── tuneable constants ────────────────────────────────────────────────────────
BLINK_EAR_THRESHOLD = 0.18   # EAR proxy below this => eye likely closed
BLINK_CONSEC_FRAMES = 2      # min consecutive closed frames to register a blink
CASCADE_SCALE       = 1.15   # Haar scaleFactor  (lower = slower, more detections)
CASCADE_NEIGHBOURS  = 4      # Haar minNeighbors
MIN_FACE_PX         = 60     # min face bbox side in pixels


# ── data structures ───────────────────────────────────────────────────────────
@dataclass
class FrameResult:
    frame_idx:      int
    timestamp_s:    float
    face_detected:  bool  = False
    # EAR (proxy via bbox aspect ratio)
    left_eye_ear:   float = 0.0
    right_eye_ear:  float = 0.0
    avg_ear:        float = 0.0
    left_eye_open:  bool  = False
    right_eye_open: bool  = False
    blink_event:    bool  = False
    # face bounding box (pixels in original image)
    face_x: int = 0
    face_y: int = 0
    face_w: int = 0   # width  ~= ear-to-ear
    face_h: int = 0   # height ~= forehead-to-chin
    # eye bboxes
    left_eye_w:  float = 0.0
    left_eye_h:  float = 0.0
    right_eye_w: float = 0.0
    right_eye_h: float = 0.0
    # nose & mouth estimated from face proportions
    est_nose_w:   float = 0.0
    est_nose_h:   float = 0.0
    est_mouth_w:  float = 0.0
    est_mouth_h:  float = 0.0


@dataclass
class ChunkResult:
    chunk_id:    int
    start_frame: int
    end_frame:   int
    frames:      List[FrameResult] = field(default_factory=list)
    error:       Optional[str]     = None


# ── helpers ───────────────────────────────────────────────────────────────────
def hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def safe_median(vals):
    v = [x for x in vals if x > 0]
    return float(np.median(v)) if v else 0.0


def safe_mean(vals):
    v = [x for x in vals if x > 0]
    return float(np.mean(v)) if v else 0.0


def safe_std(vals):
    v = [x for x in vals if x > 0]
    return float(np.std(v)) if v else 0.0


def make_cascades():
    """
    Each worker thread must create its OWN cascade instances —
    cv2.CascadeClassifier is NOT thread-safe to share.
    """
    p = cv2.data.haarcascades
    return (
        cv2.CascadeClassifier(p + "haarcascade_frontalface_default.xml"),
        cv2.CascadeClassifier(p + "haarcascade_lefteye_2splits.xml"),
        cv2.CascadeClassifier(p + "haarcascade_righteye_2splits.xml"),
        cv2.CascadeClassifier(p + "haarcascade_eye.xml"),          # fallback
    )


# ── single-frame analysis ─────────────────────────────────────────────────────
# Max width fed to the cascade. At 4K (3840px wide) we downscale 4× to 960px —
# detection is ≈16× faster and accuracy is unchanged for frontal faces.
DETECT_MAX_W = 960

def analyse_frame(gray: np.ndarray, face_c, leye_c, reye_c, eye_c) -> Optional[FrameResult]:
    """
    Detect face + eyes in a grayscale (already equalised) frame.
    Returns a partially-filled FrameResult (frame_idx/timestamp set by caller),
    or None if no face found.
    """
    orig_h, orig_w = gray.shape[:2]

    # ── Downscale for cascade speed (4K → ~960px wide = 16× fewer pixels) ──
    if orig_w > DETECT_MAX_W:
        scale_down = DETECT_MAX_W / orig_w
        det_w = DETECT_MAX_W
        det_h = int(orig_h * scale_down)
        gray_small = cv2.resize(gray, (det_w, det_h), interpolation=cv2.INTER_AREA)
    else:
        scale_down = 1.0
        gray_small = gray

    # 1. Face (on downscaled image)
    faces = face_c.detectMultiScale(
        gray_small,
        scaleFactor=CASCADE_SCALE,
        minNeighbors=CASCADE_NEIGHBOURS,
        minSize=(MIN_FACE_PX, MIN_FACE_PX),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if not len(faces):
        return None

    # Scale bbox back to original pixel coordinates
    sx = 1.0 / scale_down
    fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
    fx  = int(fx * sx);  fy  = int(fy * sx)
    fw  = int(fw * sx);  fh  = int(fh * sx)

    # Eye detection on the original-resolution face ROI (more accurate dims)
    face_roi = gray[fy : fy + fh, fx : fx + fw]

    # 2. Split top 60% of face for eye search
    top_h = max(1, int(fh * 0.60))
    top   = face_roi[:top_h, :]
    mid   = fw // 2

    def best_eye(casc, roi, fb_casc):
        kw = dict(
            scaleFactor  = 1.1,
            minNeighbors = 3,
            minSize      = (max(8, int(fw * 0.10)), max(6, int(fh * 0.06))),
            maxSize      = (int(fw * 0.55), int(top_h * 0.85)),
        )
        eyes = casc.detectMultiScale(roi, **kw)
        if not len(eyes):
            eyes = fb_casc.detectMultiScale(roi, **kw)
        return max(eyes, key=lambda e: e[2] * e[3]) if len(eyes) else None

    r_eye = best_eye(reye_c, top[:, :mid],  eye_c)   # image-left  = person-right
    l_eye = best_eye(leye_c, top[:, mid:],  eye_c)   # image-right = person-left

    def ear(bbox):
        if bbox is None:
            return 0.05          # treat missing eye as closed
        _, _, ew, eh = bbox
        return eh / ew if ew else 0.05

    lear = ear(l_eye)
    rear = ear(r_eye)
    avg  = (lear + rear) / 2.0

    return FrameResult(
        frame_idx      = 0,
        timestamp_s    = 0.0,
        face_detected  = True,
        left_eye_ear   = round(lear, 4),
        right_eye_ear  = round(rear, 4),
        avg_ear        = round(avg,  4),
        left_eye_open  = l_eye is not None,
        right_eye_open = r_eye is not None,
        face_x = fx, face_y = fy, face_w = fw, face_h = fh,
        left_eye_w   = float(l_eye[2]) if l_eye is not None else 0.0,
        left_eye_h   = float(l_eye[3]) if l_eye is not None else 0.0,
        right_eye_w  = float(r_eye[2]) if r_eye is not None else 0.0,
        right_eye_h  = float(r_eye[3]) if r_eye is not None else 0.0,
        # Golden-ratio proportions (avg. human face measurements)
        est_nose_w   = round(fw * 0.35, 1),
        est_nose_h   = round(fh * 0.30, 1),
        est_mouth_w  = round(fw * 0.50, 1),
        est_mouth_h  = round(fh * 0.08, 1),
    )


# ── Worker thread ─────────────────────────────────────────────────────────────
class VideoChunkWorker(threading.Thread):
    """
    Reads its assigned frame range, runs face/eye detection every `skip` frames,
    applies blink state machine, posts ChunkResult to result_queue.
    """

    def __init__(self, chunk_id, video_path, start_frame, end_frame,
                 fps, skip, result_queue, progress_lock, progress_counter):
        super().__init__(name=f"Worker-{chunk_id:02d}", daemon=True)
        self.chunk_id         = chunk_id
        self.video_path       = video_path
        self.start_frame      = start_frame
        self.end_frame        = end_frame
        self.fps              = fps
        self.skip             = skip
        self.result_queue     = result_queue
        self.progress_lock    = progress_lock
        self.progress_counter = progress_counter

    def run(self):
        chunk = ChunkResult(self.chunk_id, self.start_frame, self.end_frame)
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {self.video_path}")

            # Thread-local cascade instances
            face_c, leye_c, reye_c, eye_c = make_cascades()

            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

            closed_streak = 0
            fidx = self.start_frame

            while fidx < self.end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                if (fidx - self.start_frame) % self.skip == 0:
                    ts   = fidx / self.fps
                    gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

                    fr = analyse_frame(gray, face_c, leye_c, reye_c, eye_c)
                    if fr is None:
                        fr = FrameResult(frame_idx=fidx, timestamp_s=ts)
                    else:
                        fr.frame_idx  = fidx
                        fr.timestamp_s = ts

                    # ── Blink state machine ───────────────────────────────────
                    eye_closed = (
                        fr.face_detected and (
                            fr.avg_ear < BLINK_EAR_THRESHOLD
                            or (not fr.left_eye_open and not fr.right_eye_open)
                        )
                    )

                    if eye_closed:
                        closed_streak += 1
                    else:
                        if closed_streak >= BLINK_CONSEC_FRAMES:
                            idx = len(chunk.frames) - closed_streak
                            if 0 <= idx < len(chunk.frames):
                                chunk.frames[idx].blink_event = True
                        closed_streak = 0

                    chunk.frames.append(fr)

                    with self.progress_lock:
                        self.progress_counter[0] += self.skip

                fidx += 1

            # Flush trailing closed streak
            if closed_streak >= BLINK_CONSEC_FRAMES:
                idx = len(chunk.frames) - closed_streak
                if 0 <= idx < len(chunk.frames):
                    chunk.frames[idx].blink_event = True

            cap.release()

        except Exception as exc:
            chunk.error = str(exc)
            log.error("Chunk %02d: %s", self.chunk_id, exc, exc_info=True)

        blinks = sum(1 for f in chunk.frames if f.blink_event)
        faces  = sum(1 for f in chunk.frames if f.face_detected)
        log.info("Chunk %02d done │ frames=%d │ faces=%d │ blinks=%d",
                 self.chunk_id, len(chunk.frames), faces, blinks)
        self.result_queue.put(chunk)


# ── Progress printer thread ───────────────────────────────────────────────────
class ProgressPrinter(threading.Thread):
    def __init__(self, total_frames, counter, lock, done_event):
        super().__init__(name="ProgressPrinter", daemon=True)
        self.total = total_frames
        self.counter = counter
        self.lock = lock
        self.done = done_event

    def run(self):
        t0 = time.time()
        while not self.done.wait(10):
            with self.lock:
                done = self.counter[0]
            elapsed = time.time() - t0
            pct = min(done / max(self.total, 1) * 100, 100)
            fps_eff = done / elapsed if elapsed else 0
            eta = (self.total - done) / fps_eff if fps_eff else 0
            log.info("Progress │ %5.1f%% │ %d/%d frames │ %.1f fps │ ETA %s",
                     pct, done, self.total, fps_eff, hms(eta))


# ── Merge & compute final statistics ─────────────────────────────────────────
def merge_results(chunks, duration_s, fps, skip, n_workers, elapsed_s):
    all_frames: List[FrameResult] = []
    for ch in sorted(chunks, key=lambda c: c.chunk_id):
        all_frames.extend(ch.frames)

    detected     = [f for f in all_frames if f.face_detected]
    blink_frames = [f for f in all_frames if f.blink_event]

    total_blinks  = len(blink_frames)
    rate_ps = total_blinks / duration_s if duration_s else 0
    rate_pm = rate_ps * 60

    blink_ts = [f.timestamp_s for f in blink_frames]
    ibi      = np.diff(blink_ts).tolist() if len(blink_ts) > 1 else []

    per_minute: Dict[int, int] = defaultdict(int)
    for f in blink_frames:
        per_minute[int(f.timestamp_s // 60)] += 1

    # Downsample EAR timeline to ≤3000 pts for JSON portability
    step = max(1, len(all_frames) // 3000)
    ear_timeline = [
        {"t": round(f.timestamp_s, 2), "ear": round(f.avg_ear, 4), "blink": f.blink_event}
        for f in all_frames[::step]
    ]

    def med(a): return round(safe_median([getattr(f, a) for f in detected]), 2)
    def mn(a):  return round(safe_mean  ([getattr(f, a) for f in detected]), 2)
    def sd(a):  return round(safe_std   ([getattr(f, a) for f in detected]), 2)

    stats = {
        "summary": {
            "video_duration_hms":     hms(duration_s),
            "video_duration_seconds": round(duration_s, 1),
            "total_frames_analysed":  len(all_frames),
            "frames_with_face":       len(detected),
            "detection_rate_pct":     round(len(detected) / max(len(all_frames), 1) * 100, 2),
        },
        "blink": {
            "total_blinks":            total_blinks,
            "blinks_per_second":       round(rate_ps, 5),
            "blinks_per_minute":       round(rate_pm, 2),
            "blinks_per_hour":         round(rate_pm * 60, 1),
            "avg_inter_blink_s":       round(float(np.mean(ibi)), 3) if ibi else 0,
            "std_inter_blink_s":       round(float(np.std(ibi)),  3) if ibi else 0,
            "ear_threshold":           BLINK_EAR_THRESHOLD,
            "per_minute_counts":       {str(k): v for k, v in sorted(per_minute.items())},
            "blink_timestamps_first200": blink_ts[:200],
        },
        "face_dimensions_pixels": {
            "face_height_px":   {"median": med("face_h"), "mean": mn("face_h"), "std": sd("face_h")},
            "face_width_px":    {"median": med("face_w"), "mean": mn("face_w"), "std": sd("face_w")},
            "left_eye_width":   {"median": med("left_eye_w"),  "mean": mn("left_eye_w")},
            "left_eye_height":  {"median": med("left_eye_h"),  "mean": mn("left_eye_h")},
            "right_eye_width":  {"median": med("right_eye_w"), "mean": mn("right_eye_w")},
            "right_eye_height": {"median": med("right_eye_h"), "mean": mn("right_eye_h")},
            "est_nose_width":   {"median": med("est_nose_w"),  "mean": mn("est_nose_w")},
            "est_nose_height":  {"median": med("est_nose_h"),  "mean": mn("est_nose_h")},
            "est_mouth_width":  {"median": med("est_mouth_w"), "mean": mn("est_mouth_w")},
            "est_mouth_height": {"median": med("est_mouth_h"), "mean": mn("est_mouth_h")},
            "methodology": (
                "face_height / face_width = Haar cascade bounding box. "
                "Eye dims = eye sub-cascade bbox. "
                "Nose/mouth = golden-ratio proportions from face bbox "
                "(nose_w=35%fw, nose_h=30%fh, mouth_w=50%fw, mouth_h=8%fh)."
            ),
        },
        "ear_timeline": ear_timeline,
        "processing": {
            "workers":         n_workers,
            "frame_skip":      skip,
            "elapsed_seconds": round(elapsed_s, 1),
            "effective_fps":   round(len(all_frames) * skip / elapsed_s, 1) if elapsed_s else 0,
        },
    }
    return stats, all_frames, blink_frames


# ── Save CSV ──────────────────────────────────────────────────────────────────
def save_csv(frames: List[FrameResult], path: str):
    if not frames:
        return
    fields = list(FrameResult.__dataclass_fields__.keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for fr in frames:
            w.writerow(asdict(fr))
    log.info("CSV  → %s  (%d rows)", path, len(frames))


# ── Annotated 60-second preview clip ─────────────────────────────────────────
def save_annotated_clip(video_path, blink_frames, out_path, fps, duration_s=60.0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.warning("Cannot open video for clip — skipped")
        return

    blink_set = {f.frame_idx for f in blink_frames}
    n         = int(fps * duration_s)
    vid_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Downscale clip output to max 1920px wide (saves disk, faster encode)
    out_w, out_h = vid_w, vid_h
    if vid_w > 1920:
        out_w = 1920
        out_h = int(vid_h * 1920 / vid_w)

    writer    = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))

    face_c, leye_c, reye_c, eye_c = make_cascades()

    idx = 0
    while idx < n:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for output if needed
        if (out_w, out_h) != (vid_w, vid_h):
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        fr   = analyse_frame(gray, face_c, leye_c, reye_c, eye_c)

        if fr is not None:
            fx, fy, fw, fh = fr.face_x, fr.face_y, fr.face_w, fr.face_h
            # Face box (green)
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0, 220, 120), 2)
            cv2.putText(frame, f"Face {fw}x{fh}px",
                        (fx, max(fy-8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,220,120), 1)
            # Eye regions (blue / orange)
            mid   = fw // 2
            top_h = int(fh * 0.60)
            cv2.rectangle(frame, (fx, fy+5), (fx+mid, fy+top_h), (100,200,255), 1)
            cv2.rectangle(frame, (fx+mid, fy+5), (fx+fw, fy+top_h), (255,160,50), 1)
            cv2.putText(frame, f"EAR {fr.avg_ear:.3f}",
                        (fx, fy+fh+16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

        if idx in blink_set:
            ov = frame.copy()
            cv2.rectangle(ov, (0, 0), (vid_w, vid_h), (0, 50, 220), -1)
            cv2.addWeighted(ov, 0.22, frame, 0.78, 0, frame)
            cv2.putText(frame, "BLINK", (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.4, (0, 80, 255), 5)

        cv2.putText(frame, hms(idx/fps), (14, vid_h-14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        writer.write(frame)
        idx += 1

    writer.release()
    cap.release()
    log.info("Clip → %s", out_path)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Multithreaded facial analysis — blink rate & face dimensions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--video",         required=True)
    ap.add_argument("--workers",       type=int,   default=0,
                    help="Worker threads (0=auto: 2×CPU, max 8)")
    ap.add_argument("--skip",          type=int,   default=3,
                    help="Analyse every Nth frame (default 3)")
    ap.add_argument("--out",           default="results")
    ap.add_argument("--no-clip",       action="store_true")
    ap.add_argument("--clip-duration", type=float, default=60.0)
    args = ap.parse_args()

    if not os.path.exists(args.video):
        log.error("Video not found: %s", args.video)
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Probe ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        log.error("Cannot open: %s", args.video)
        sys.exit(1)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration_s   = total_frames / fps
    vid_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    size_gb = os.path.getsize(args.video) / 1e9
    log.info("═" * 65)
    log.info("Video    : %s", args.video)
    log.info("Size     : %.2f GB", size_gb)
    log.info("Frames   : %d  │  FPS: %.3f  │  Duration: %s",
             total_frames, fps, hms(duration_s))
    log.info("Res      : %d × %d", vid_w, vid_h)
    log.info("═" * 65)

    # ── Worker count ─────────────────────────────────────────────────────────
    cpu = multiprocessing.cpu_count()
    n_workers = args.workers if args.workers > 0 else min(8, max(4, cpu * 2))
    log.info("Workers  : %d  │  Skip: every %d frames", n_workers, args.skip)

    # ── Partition ────────────────────────────────────────────────────────────
    chunk_size = math.ceil(total_frames / n_workers)
    chunk_defs = []
    for i in range(n_workers):
        s = i * chunk_size
        e = min(s + chunk_size, total_frames)
        if s < total_frames:
            chunk_defs.append((i, s, e))

    log.info("Chunks   : %d  (~%d frames each, ~%d analysed per chunk)",
             len(chunk_defs), chunk_size, chunk_size // args.skip)

    # ── Shared state ─────────────────────────────────────────────────────────
    result_queue     = queue.Queue()
    progress_lock    = threading.Lock()
    progress_counter = [0]
    done_event       = threading.Event()

    # ── Launch workers ────────────────────────────────────────────────────────
    t_start = time.time()
    workers = [
        VideoChunkWorker(
            chunk_id=cid, video_path=args.video,
            start_frame=s, end_frame=e, fps=fps, skip=args.skip,
            result_queue=result_queue, progress_lock=progress_lock,
            progress_counter=progress_counter,
        )
        for cid, s, e in chunk_defs
    ]

    ProgressPrinter(total_frames, progress_counter, progress_lock, done_event).start()

    for w in workers:
        w.start()
        log.info("  ▶ %s  [frames %d – %d]", w.name, w.start_frame, w.end_frame)

    # ── Collect ───────────────────────────────────────────────────────────────
    chunk_results = []
    for _ in range(len(workers)):
        chunk_results.append(result_queue.get())

    done_event.set()
    elapsed = time.time() - t_start
    log.info("Done │ %.1f s │ %.1f fps effective", elapsed, total_frames / elapsed)

    # ── Merge ─────────────────────────────────────────────────────────────────
    log.info("Merging …")
    stats, all_frames, blink_frames = merge_results(
        chunk_results, duration_s, fps, args.skip, n_workers, elapsed
    )

    # ── Outputs ───────────────────────────────────────────────────────────────
    json_p = out_dir / "facial_analysis_results.json"
    with open(json_p, "w") as f:
        json.dump(stats, f, indent=2)
    log.info("JSON → %s", json_p)

    save_csv(all_frames, str(out_dir / "frame_data.csv"))

    if not args.no_clip:
        log.info("Generating annotated clip (%.0fs) …", args.clip_duration)
        save_annotated_clip(args.video, blink_frames,
                            str(out_dir / "annotated_sample.mp4"),
                            fps, args.clip_duration)

    # ── Summary ───────────────────────────────────────────────────────────────
    s  = stats["summary"]
    b  = stats["blink"]
    dm = stats["face_dimensions_pixels"]
    pr = stats["processing"]

    print("\n" + "═" * 65)
    print("  FACIAL ANALYSIS  —  RESULTS")
    print("═" * 65)
    print(f"  Video duration          : {s['video_duration_hms']}")
    print(f"  Frames w/ face          : {s['frames_with_face']:,} / {s['total_frames_analysed']:,}  ({s['detection_rate_pct']}%)")
    print()
    print("  ─── BLINK RATE ────────────────────────────────────────")
    print(f"  Total blinks            : {b['total_blinks']:,}")
    print(f"  Blinks / second         : {b['blinks_per_second']:.5f}")
    print(f"  Blinks / minute         : {b['blinks_per_minute']:.2f}")
    print(f"  Blinks / hour           : {b['blinks_per_hour']:.1f}")
    print(f"  Avg inter-blink gap     : {b['avg_inter_blink_s']:.3f} s  (σ={b['std_inter_blink_s']:.3f})")
    print()
    print("  ─── FACE DIMENSIONS (pixels, median) ──────────────────")
    print(f"  Face height (head→chin) : {dm['face_height_px']['median']:.1f} px")
    print(f"  Face width  (ear→ear)   : {dm['face_width_px']['median']:.1f} px")
    print(f"  Left  eye   W × H       : {dm['left_eye_width']['median']:.1f} × {dm['left_eye_height']['median']:.1f} px")
    print(f"  Right eye   W × H       : {dm['right_eye_width']['median']:.1f} × {dm['right_eye_height']['median']:.1f} px")
    print(f"  Est. nose   W × H       : {dm['est_nose_width']['median']:.1f} × {dm['est_nose_height']['median']:.1f} px")
    print(f"  Est. mouth  W × H       : {dm['est_mouth_width']['median']:.1f} × {dm['est_mouth_height']['median']:.1f} px")
    print()
    print("  ─── PROCESSING ────────────────────────────────────────")
    print(f"  Workers / frame-skip    : {pr['workers']} / {pr['frame_skip']}")
    print(f"  Wall-clock time         : {pr['elapsed_seconds']} s")
    print(f"  Effective throughput    : {pr['effective_fps']} fps")
    print(f"  Output directory        : {out_dir.resolve()}")
    print("═" * 65 + "\n")


if __name__ == "__main__":
    main()
