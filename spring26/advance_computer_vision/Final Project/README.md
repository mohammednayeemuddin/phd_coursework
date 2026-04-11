# Bird Filtering

**Wildlife Photo Curator & Bird Family Bucketing — Pure Computer Vision, No ML Required**

Automatically groups, labels, and quality-ranks bird photographs from a field session. Given a folder of images, it buckets similar shots together, identifies the bird family in each bucket, scores every photo across six quality axes, and outputs a clean folder hierarchy with the best picks flagged — no internet, no model downloads, no GPU.

---

## How It Works

```
Input folder (hundreds of raw shots)
        │
        ▼
 Stage 1 — Bucket        Spatial fingerprint (3×2 HSV grid) + average-linkage
                         clustering groups burst/similar shots together.
                         Each bucket gets a bird family label:
                         Duck/Waterfowl · Wading Bird · Dark Waterbird · Raptor
        │
        ▼
 Stage 2 — Quality       6 axes scored in the subject zone only (not globally):
                         Sharpness · BG separation · Subject fill
                         Exposure · Colour richness · Composition
        │
        ▼
 Output
   results/
   ├── BUCKETS/          Every image sorted into its scene folder, ranked by quality
   │   ├── Duck__Waterfowl_-_scene_1/
   │   │   ├── BEST_01_DSC00456.JPG
   │   │   ├── BEST_02_DSC00457.JPG
   │   │   └── 03_DSC00458.JPG
   │   └── Wading_Bird_-_scene_2/ ...
   ├── BEST_PICKS/       Flat folder — one best pick per scene, named by bucket
   ├── bird_quality_report.json
   └── bird_quality_report.csv
```

---

## Requirements

- Python 3.9+
- macOS 12+, Ubuntu 20.04+, or Windows 10+
- No GPU, no internet connection, no model downloads

```bash
pip install opencv-python numpy
```

---

## Quick Start

```bash
# Analyse a folder, pick best 2 per scene
python bird_cv.py ./photos/ --top 2

# Pick only the single best per scene, dump top 5 in each bucket folder
python bird_cv.py ./photos/ --top 1 --dump-top 5

# Resize selected picks on export (saves storage)
python bird_cv.py ./photos/ --top 1 --save-resized 1920

# Custom output folder
python bird_cv.py ./photos/ --top 2 --output ./my_results/
```

---

## All Options

| Flag | Default | Description |
|---|---|---|
| `--output` | `results/` | Output directory |
| `--top N` | `3` | Best picks per bucket (flagged BEST_ and copied to BEST_PICKS/) |
| `--dump-top N` | all | Only copy top N images into each bucket folder |
| `--group-threshold F` | `0.08` | Similarity threshold — lower = stricter grouping (0.03–0.10) |
| `--save-resized PX` | off | Resize exported images to this long-side pixel count |
| `--workers N` | auto | Parallel workers (default: min(images, CPU cores, 8)) |

---

## Files

```
FeatherIdentify/
├── bird_cv.py          # Main script — bucketing + quality scoring + output
├── resize_photos.py    # Optional utility: batch-resize large originals before analysis
└── README.md
```

---

## resize_photos.py (optional pre-step)

If your originals are large (10–20MB RAW/JPEG), resize them first. Analysis quality is identical from 1200px upward — sharpness rankings are preserved, colour features change by less than 1.5%.

```bash
# Resize to 1200px long side, save to ./small/
python resize_photos.py ./original_photos/ ./small/ --long-side 1200

# Options
python resize_photos.py ./photos/ ./out/ --long-side 1600 --quality 90 --workers 8
```

| Flag | Default | Description |
|---|---|---|
| `--long-side PX` | `1200` | Long-side pixel target |
| `--quality N` | `88` | JPEG quality 1–100 |
| `--workers N` | auto | Parallel workers |

---

## Quality Scoring

All six axes are measured in the **central subject zone only** — background bokeh does not penalise a sharp subject.

| Score (0–10) | What it measures | Weight |
|---|---|---|
| Subject sharpness | Laplacian variance in centre zone | 35% |
| BG separation | Subject sharpness ÷ background sharpness | 20% |
| Subject fill | Edge density centre vs global | 18% |
| Exposure | Brightness spread + clipping penalty | 15% |
| Colour richness | Mean saturation in subject zone | 7% |
| Composition | Sharpness peak near rule-of-thirds | 5% |

---

## Bird Family Labels

| Label | Detected via |
|---|---|
| Duck / Waterfowl | Orange feet anchor + white chest + brown flanks |
| Wading Bird | Tall contour aspect ratio + orange/pale plumage |
| Dark Waterbird | High dark-pixel ratio + wading aspect |
| Raptor | Sky-blue ratio + wide horizontal aspect |
| Mixed / Multiple Species | No dominant signal |

> **Note:** Detection is at **family level** only. Species-level identification (e.g. Northern Shoveler vs Mallard) requires ML and is outside the scope of this pure-CV implementation.

---

## Tuning the Grouping Threshold

| Threshold | Behaviour |
|---|---|
| `0.03–0.05` | Strict — only near-identical burst shots group together |
| `0.08` *(default)* | Same scene with slightly different framing |
| `0.10–0.15` | Loose — related shots across a session may merge |

---

## Performance

| Setup | Speed |
|---|---|
| 2-core machine, 77 photos at 1920px | ~19s wall time |
| 8-core machine, 200 photos at 1920px | ~25–35s estimated |

Scales linearly with CPU cores via `ThreadPoolExecutor`.

---

## License

MIT
