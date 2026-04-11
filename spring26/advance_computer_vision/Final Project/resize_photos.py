"""
resize_photos.py — Batch photo resizer
Usage:
  python resize_photos.py <input_folder> [output_folder] [--long-side PX] [--quality Q]

Defaults:
  output_folder : input_folder/resized/
  --long-side   : 1200   (good for sharing/debugging, ~0.5-1MB per shot)
  --quality     : 88     (JPEG quality, 85-92 is visually lossless)

Examples:
  python resize_photos.py ./photos/
  python resize_photos.py ./photos/ ./small/ --long-side 800
  python resize_photos.py ./photos/ ./small/ --long-side 1600 --quality 90
"""

import cv2, argparse, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading, os

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
_lock = threading.Lock()

def resize_one(src: Path, dst: Path, long_side: int, quality: int):
    t0  = time.perf_counter()
    img = cv2.imread(str(src))
    if img is None:
        return src.name, False, "unreadable"
    h, w  = img.shape[:2]
    scale = long_side / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w*scale), int(h*scale)),
                         interpolation=cv2.INTER_LANCZOS4)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    orig_mb  = src.stat().st_size / 1e6
    out_mb   = dst.stat().st_size / 1e6
    ms       = (time.perf_counter() - t0) * 1000
    with _lock:
        print(f"  {src.name:<35} {orig_mb:5.1f}MB → {out_mb:4.1f}MB  "
              f"({out_mb/orig_mb*100:3.0f}%)  [{ms:.0f}ms]")
    return src.name, True, None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output", nargs="?", default=None)
    p.add_argument("--long-side", type=int,   default=1200)
    p.add_argument("--quality",   type=int,   default=88)
    p.add_argument("--workers",   type=int,   default=None)
    args = p.parse_args()

    src = Path(args.input)
    dst = Path(args.output) if args.output else src / "resized"
    images = sorted(p for p in src.iterdir()
                    if p.is_file() and p.suffix.lower() in SUPPORTED)
    if not images:
        print("No images found."); return

    workers = args.workers or min(len(images), os.cpu_count() or 2, 8)
    print(f"\nResizing {len(images)} image(s) → {dst}/")
    print(f"Long side: {args.long_side}px  |  Quality: {args.quality}  |  Workers: {workers}\n")

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(resize_one, img, dst / img.name,
                        args.long_side, args.quality): img
            for img in images
        }
        ok = sum(1 for f in as_completed(futures) if f.result()[1])

    elapsed = time.perf_counter() - t0
    print(f"\n  {ok}/{len(images)} done in {elapsed:.1f}s  →  {dst}/")

if __name__ == "__main__":
    main()
