---
name: bird-photo-curator
description: Curate, bucket, and quality-rank a folder of wildlife/bird photographs using the pure-CV FeatherIdentify pipeline (bird_cv.py + resize_photos.py). Use when the user wants to sort a field-session photo dump into scenes, label bird families, pick the sharpest shots per scene, batch-resize originals, or tune/extend the clustering and quality-scoring rules. Triggers on "bird photos", "cull my shots", "best picks", "bucket these images", "bird_cv.py", "FeatherIdentify", "resize_photos.py".
---

# Bird Photo Curator (FeatherIdentify)

Pure-computer-vision wildlife photo curation. No ML, no model downloads, no GPU, no
internet. Everything is OpenCV + NumPy heuristics.

**Location:** `spring26/advance_computer_vision/Final Project/`
- `bird_cv.py` — the pipeline (bucketing + family labelling + quality scoring + output)
- `resize_photos.py` — optional pre-step batch resizer
- `README.md` — user-facing docs

**Hard constraint:** this is a *pure CV* deliverable. Do not introduce ML models,
pretrained weights, torch/tensorflow, or network calls. Any improvement must be a
classical-CV heuristic. Detection is deliberately **family-level only** — species ID
(Mallard vs Northern Shoveler) is out of scope.

## Requirements

```bash
pip install opencv-python numpy      # Python 3.9+
```

## Running it

```bash
# Standard: bucket a folder, flag the best 2 per scene
python bird_cv.py ./photos/ --top 2

# Tight cull: one best pick per scene, only top 5 copied into each bucket folder
python bird_cv.py ./photos/ --top 1 --dump-top 5

# Shrink exported copies (originals stay untouched)
python bird_cv.py ./photos/ --top 1 --save-resized 1920 --output ./my_results/
```

Optional pre-step when originals are 10–20 MB — rankings are unchanged from 1200px up:

```bash
python resize_photos.py ./original_photos/ ./small/ --long-side 1200 --quality 88
```

### Flags

| Flag | Actual code default | Notes |
|---|---|---|
| `--output` | `results` | output directory |
| `--top N` | `3` | best picks per bucket → `BEST_` prefix + copied to `BEST_PICKS/` |
| `--dump-top N` | all | limit how many images land in each bucket folder |
| `--group-threshold F` | `0.05` | |
| `--save-resized PX` | off | long-side px for exported copies (JPEG q92) |
| `--workers N` | `min(images, cpu_count//2)` | physical cores; OpenCV pinned to 1 thread each |

## Output layout

```
results/
├── BUCKETS/<Family_-_scene_N>/   all images, renamed  BEST_01_x.JPG / 03_y.JPG
├── BEST_PICKS/                   flat, <bucket_tag>_R01_<file>.JPG
├── bird_quality_report.json
└── bird_quality_report.csv
```

Files are **copied**, never moved — the input folder is never mutated.

## Pipeline internals (where to edit)

**Stage 1A — family features** (`extract_family_features`): 6 ratios measured in the
central zone (rows 12–82%, cols 8–92%) of an 800px-wide copy — `dark_ratio`,
`white_ratio`, `orange_ratio`, `brown_ratio`, `sky_ratio`, `aspect_ratio` (bounding box
of the largest closed-Canny contour). HSV thresholds here are hand-calibrated against
real field photos, which are far less saturated than stock imagery — **retune, don't
"clean up"**.

**Stage 1A — classifier** (`classify_bird_family`): additive weighted score per family,
negatives clipped to 0. Returns `unknown` when the top score < 0.3, and `mixed_scene`
when the top-two margin < 0.20. Note the guard: `orange > 0.60 and white < 0.05` is
read as a flamingo/leg close-up (→ wading bird), not a duck.

**Stage 1B — fingerprint** (`image_fingerprint`): 3×2 spatial grid of HSV histograms
(12 H + 6 S + 6 V bins per cell) on a 96×54 thumb, plus a flattened 32×18 grey thumb.
Spatial, so scenes with the same palette but different layout stay apart.

**Stage 1B — clustering** (`group_by_similarity`): average-linkage on cosine distance —
a photo joins a cluster only if its mean distance to *all* current members is under
threshold. This is deliberate: greedy/single-linkage chain-links A≈B≈C into one bucket
when A and C look nothing alike. Don't swap it for a greedy pass.

**Stage 2 — quality** (`compute_quality`), all scored in the subject zone so background
bokeh never penalises a sharp bird:

| Axis | Function | Weight |
|---|---|---|
| Subject sharpness | Laplacian variance, 18% margin | 0.35 |
| BG separation | subject var ÷ mean border var | 0.20 |
| Subject fill | centre edge density ÷ global | 0.18 |
| Exposure | brightness mean/spread + clipping penalty | 0.15 |
| Colour richness | mean S in subject zone | 0.07 |
| Composition | blurred-Laplacian peak vs rule of thirds | 0.05 |

Weights appear **twice** — in `compute_quality`'s `quality_overall` sum. Change them
there and keep the README table in sync.

Analysis runs at `ANALYSIS_LONG_SIDE = 1920` after a `bilateralFilter(9, 50, 50)`.
Family features are computed on the *original* image, quality on the resized copy.

## Tuning the grouping threshold

| Value | Behaviour |
|---|---|
| 0.03–0.05 | strict — near-identical burst frames only (current default) |
| 0.08 | same scene, slightly different framing |
| 0.10–0.15 | loose — related shots across a whole session merge |

Symptom → fix: one giant bucket ⇒ lower the threshold; every photo its own bucket ⇒ raise it.

## Performance (optimised 2026-08-26 — see PERFORMANCE.md)

Runs ~3x faster than the original: 77 images @1920px in 0.64s, 24 @6000px in 0.36s on 16
physical cores. What is already done, so you don't redo it:

- **Decode is resolution-aware.** `imread_for_analysis()` picks libjpeg's ½/¼ DCT scale
  using `_peek_size()`, a dependency-free JPEG/PNG header probe. Never "optimise" this
  into a plain `cv2.imread` — it is the single biggest win on real originals.
- **`DENOISE_D = 5`** (was 9). Bilateral cost is O(d²) and this op was 77% of the
  pipeline. Set it back to 9 to restore original scores at ~half the speed.
- **`score_composition`** builds its saliency map at quarter resolution — only the
  normalised peak position is used.
- **`compute_quality`** scores each axis once (four were computed twice).
- **`cv2.setNumThreads(1)` per worker.** OpenCV threads its own ops, so W workers × N
  internal threads oversubscribed the box. Do not remove this when touching the pool.

Remaining headroom is small and awkward: after the above, no single op exceeds ~20% of
the budget. Don't micro-optimise the scoring functions — they are ~10% combined.

## Benchmarking

```bash
python bench/gen_corpus.py ./bench_data              # synthetic burst-structured corpus
python bench/profile_pipeline.py ./bench_data/corpus_1920
python bench/check_determinism.py ./bench_data/corpus_1920   # exits 1 on drift
```

The corpus is synthetic: structure is realistic, sensor noise is not. Re-validate any
denoise change against real field photos before trusting score deltas.

## Working notes

- Unreadable images don't abort the run — the worker catches, records `error`, and that
  file is dropped from `fps`, so it never reaches a bucket or the report.
- Threaded printing goes through `tprint()`/`_print_lock`; use it for any new worker
  output or lines will interleave.
- **Runs are NOT reproducible as written.** Quality scores are deterministic, but
  `fps{}` is filled in `as_completed()` order and `group_by_similarity` seeds its greedy
  clusters by iterating that dict — so bucket membership, and therefore the BEST picks,
  depend on which thread finished first (~2 of 8 four-run batches drift). Fix is one
  word: `names = sorted(fingerprints.keys())`. Apply it before A/B-ing any tuning change,
  or you will read thread scheduling as a scoring difference.
- `compute_quality` calls `score_bg_separation`, `score_subject_fill`,
  `score_color_richness` and `score_composition` **twice each** — once in the
  `quality_overall` sum, once for their own report field. ~11% of serial cost, free to remove.
- Performance: `bilateralFilter(9,50,50)` is ~77% of all serial work. Worker default
  `min(images, cpu_count, 8)` both under-uses big machines and oversubscribes, since each
  worker calls an internally-threaded OpenCV. See `PERFORMANCE.md` and `bench/`.
- Bucket folder names are sanitised with `replace(" ","_").replace("/","").replace("—","-")`;
  a new family label containing other punctuation needs that list extended.
