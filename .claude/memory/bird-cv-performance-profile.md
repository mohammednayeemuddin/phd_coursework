---
name: bird-cv-performance-profile
description: bird_cv.py was optimised ~3x on 2026-08-26; the denoise diameter DENOISE_D is the dial that trades speed against exact quality scores.
metadata:
  type: project
---

`bird_cv.py` was profiled and optimised on 2026-08-26 (baseline and full record in
`spring26/advance_computer_vision/Final Project/PERFORMANCE.md`, tools in `bench/`).
`cv2.bilateralFilter` was 77% of serial cost; after the work no single op exceeds ~20%.
Result: 3.0x at 1920px, 3.7x on 6000px originals.

**Why:** the user chose speed over byte-identical scores. Output-preserving changes alone
capped at ~1.2x, so reaching the 2x goal required shrinking the denoise kernel
(`DENOISE_D` 9→5), which shifts quality scores by up to 1.55 (rank r≈0.98). Score deltas
were validated only on a synthetic corpus, so they are the least certain numbers on file.

**How to apply:** `DENOISE_D = 9` is the one-line revert to original scoring at about half
the current speed — offer it if the reported quality numbers ever need to match the
pre-optimisation report. Don't propose micro-optimising the scoring functions (~10% of
budget combined) or replacing the reduced-scale decode with a plain `cv2.imread`.
Related: [[bird-cv-bucketing-race]], [[acv-final-project-featheridentify]].
