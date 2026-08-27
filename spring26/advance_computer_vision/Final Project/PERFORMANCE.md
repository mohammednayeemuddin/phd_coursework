# Performance Baseline — bird_cv.py

Recorded 2026-08-26. Reproduce with the harness in `bench/`.

## Test setup

No photo corpus and no test suite existed, so both were built. Numbers below come from
a **synthetic** corpus (`bench/gen_corpus.py`): deterministic scenes shot as bursts, with
per-frame focus blur and exposure drift, and bird blobs painted in the exact HSV ranges
`extract_family_features` looks for. Structure is realistic; sensor noise is not, so the
*denoise* sensitivities below are the least trustworthy figures here and should be
re-checked against real field photos.

| | |
|---|---|
| Machine | AMD Ryzen 9 8940HX — 16 physical / 32 logical cores, 61 GB RAM |
| Python / OpenCV | 3.13.12 / cv2 5.0.0, numpy 2.5.2 |
| Corpus A | 77 images @ 1920×1280 (matches the README's benchmark row) |
| Corpus B | 24 images @ 6000×4000 (the originals the README says users actually have) |
| Timing | best-of-N wall clock around `run(..., top_n=2)`, output dir wiped each rep |

```bash
python bench/gen_corpus.py ./bench_data
python bench/profile_pipeline.py ./bench_data/corpus_1920
python bench/check_determinism.py ./bench_data/corpus_1920
```

## Baseline: end-to-end

| Corpus | Wall (best) | Per image |
|---|---|---|
| A — 77 @ 1920px | **2.02 – 2.08 s** | ~26 ms |
| B — 24 @ 6000px | **1.32 s** | ~55 ms |

The README's "2-core machine, 77 photos at 1920px → ~19s" is not comparable to this
32-thread box; treat 2.0 s as the local baseline, not a refutation of that row.

## Baseline: where the time goes

Phase split (corpus A) — analysis is effectively the whole run:

| Phase | Time | Share |
|---|---|---|
| analysis (parallel) | 2.25 s | 99.3% |
| export / file copy | 0.01 s | 0.6% |
| reporting (json+csv) | 0.00 s | 0.1% |
| clustering | 0.00 s | 0.0% |

Serial cost per image, OpenCV pinned to one thread (the honest CPU cost — 160–174 ms
depending on sample):

| Operation | ms/img | Share |
|---|---|---|
| **bilateralFilter(9,50,50)** | **124.2** | **77.5%** |
| score_composition (×2) | 13.2 | 8.2% |
| imread | 5.0 | 3.1% |
| score_bg_separation (×2) | 3.8 | 2.3% |
| extract_family_features | 3.2 | 2.0% |
| score_sharpness | 2.9 | 1.8% |
| score_subject_fill (×2) | 2.8 | 1.7% |
| cvtColor HSV / gray | 2.4 | 1.5% |
| score_exposure | 1.4 | 0.9% |
| score_color_richness (×2) | 0.7 | 0.4% |
| resize, fingerprint | 0.7 | 0.4% |

One filter is three-quarters of the pipeline. Everything else is rounding error.

## Two defects found while measuring

**1. Four scorers run twice per image.** In `compute_quality`, `score_bg_separation`,
`score_subject_fill`, `score_color_richness` and `score_composition` are each called once
inside the `quality_overall` sum and again to populate their own report field — identical
inputs, identical results, double the cost. Worth ~11% of serial time.

**2. Bucket assignment is not deterministic.** `fps{}` is filled in `as_completed()`
order, and `group_by_similarity` seeds its greedy clusters by iterating that dict. Which
thread finishes first therefore decides cluster seeding. The same folder analysed twice
can produce different buckets and **different BEST picks** — measured at roughly 2 of 8
batches of 4 runs on corpus A, i.e. it fires intermittently, not every run. Scores
themselves are stable; only grouping drifts. `bench/check_determinism.py` guards this.

This matters beyond correctness: it is the noise floor for any A/B comparison. A variant
that keeps 7 of 8 picks is not measurably worse than the current code, which does not
reliably keep 8 of 8 against itself.

## Worker scaling (baseline, corpus A)

The default is `min(images, cpu_count, 8)` — 8 workers here, leaving 24 logical cores
idle. Each worker also calls into an OpenCV that has 32 internal threads of its own, so
the pool oversubscribes badly.

| Workers | Wall |
|---|---|
| 8 (default) | 2.29 s |
| 16 | 1.73 s |
| 24 | **1.59 s** |
| 32 | 2.24 s |

## Optimisation results

Measured against a determinism-fixed reference so bucket drift cannot be mistaken for a
real difference. "Picks kept" = BEST picks matching that reference.

| Variant | Corpus A | vs base | Corpus B | vs base | Scores | Picks kept |
|---|---|---|---|---|---|---|
| baseline | 2.08 s | 1.00x | 1.32 s | 1.00x | — | 8/8 · 4/4 |
| **A — exact** | 1.73 s | 1.21x | 1.08 s | 1.22x | identical | 8/8 · 4/4 |
| **B — + d=5 denoise** | 0.66 s | 3.15x | 0.74 s | 1.77x | max Δ1.55 | 7/8 · 4/4 |
| **G — + reduced decode** | 0.65 s | 3.21x | **0.36 s** | **3.70x** | max Δ1.55 | 7/8 · 4/4 |

**Tier A — exact, output byte-identical.** Deduplicate the four doubled scorers; pin
OpenCV to one thread per worker and raise the worker cap above 8. Ceiling is ~1.2x: with
`bilateralFilter` untouched at 77% of the work, no output-preserving change can reach 2x.

**Tier B — reaches the 2x goal.** Drops the bilateral diameter from 9 to 5 (cost is
O(d²): 81 taps → 25) and builds the composition saliency map at quarter resolution, where
only the *normalised* position of the sharpness peak is used anyway. Quality scores shift
by up to 1.55; ranking correlation stays at r≈0.98.

**Tier G — Tier B plus resolution-aware decode. This is what shipped.** A 6000px original is downscaled to 1920
regardless, so full 24 MP decode is wasted; libjpeg's ½/¼ DCT scaling cuts decode from
91 ms to 37 ms and moves the resulting analysis frame by ~0.3/255 levels. Needs image
dimensions before decoding, via a dependency-free JPEG/PNG header probe. This is what
turns a 1.77x into 3.70x on real originals.

### Denoise alternatives considered (corpus A, 40 images)

| Variant | Filter ms | Speedup | mean Δscore | rank r | top-8 |
|---|---|---|---|---|---|
| bilateral(9) — current | 120.8 | 1.00x | — | 1.0000 | 8/8 |
| bilateral(7) | 72.6 | 1.67x | 0.067 | 0.9884 | 8/8 |
| **bilateral(5)** | 20.1 | 6.02x | 0.106 | 0.9826 | 8/8 |
| medianBlur(5) | 4.2 | 28.6x | 0.196 | 0.9553 | 8/8 |
| bilateral @half + upscale | 32.6 | 3.70x | 0.726 | 0.9507 | 8/8 |
| none | 0.03 | — | 0.169 | 0.9512 | 8/8 |

Also tried: denoising only the grayscale plane, since `compute_quality` never reads the
colour image (`img_bgr` is an unused parameter) and `hsv` feeds only a zone *mean*.
That is 3.15x cheaper than filtering 3-channel BGR, but it perturbs ranking more than
simply shrinking the kernel, so tier B is the better trade.


## What was applied

Tier G is installed in `bird_cv.py` as of 2026-08-26, with the worker default tuned to
half the logical core count (physical cores) rather than one worker per hyperthread —
32 workers measured slower than 16 on this machine.

Verified after install, against the determinism-fixed reference:

| Corpus | Before | After | Speedup | Buckets | Picks kept |
|---|---|---|---|---|---|
| A — 77 @ 1920px | 1.94 s | **0.64 s** | **3.02x** | same | 7/8 |
| B — 24 @ 6000px | 1.33 s | **0.36 s** | **3.69x** | same | 4/4 |

Determinism: 0 of 8 four-run batches drift, down from 2 of 8.

Changes, in order of contribution:

1. `imread_for_analysis()` — decode at libjpeg's ½/¼ scale when the original is large
   enough, chosen from a dependency-free JPEG/PNG header probe (`_peek_size`).
2. `DENOISE_D = 5` — bilateral diameter 9→5, cost is O(d²).
3. `score_composition` builds its saliency map at quarter resolution.
4. `compute_quality` scores each axis once instead of twice.
5. `cv2.setNumThreads(1)` per worker, worker default raised from a cap of 8 to physical
   core count.
6. `group_by_similarity` iterates `sorted(...)` — the determinism fix.
7. The end-of-run summary printed `wall/CPU` as "% parallelism", so a 14x speedup showed
   as "7%". Now reports speedup directly.

Regression guard: `python bench/check_determinism.py <corpus>` exits non-zero on drift.

### Caveat on the score shift

The ≤1.55 quality-score shift and the 7/8 pick retention are measured on synthetic
imagery whose noise characteristics are not those of a real sensor, and whose scores
cluster tightly enough that small deltas reshuffle ranks readily. Before relying on the
scores as reported numbers, re-run `bench/` against a real field-session folder. If the
picks turn out to matter more than the throughput, `DENOISE_D = 9` restores the original
denoise at roughly half the current speed and costs nothing else.
