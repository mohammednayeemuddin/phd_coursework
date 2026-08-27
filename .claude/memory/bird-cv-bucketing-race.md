---
name: bird-cv-bucketing-race
description: bird_cv.py's bucket assignment used to depend on thread completion order; fixed, but it remains the reason to compare against a deterministic reference.
metadata:
  type: project
---

`bird_cv.py` used to populate `fps{}` in `as_completed()` order and seed
`group_by_similarity`'s greedy clusters by iterating that dict, so bucket membership —
and which photos got flagged BEST — depended on which worker finished first. Measured
2026-08-26 at ~2 of 8 four-run batches drifting. **Fixed** by
`names = sorted(fingerprints.keys())`; now 0 of 8.

**Why:** it was primarily a measurement trap. Any A/B on this pipeline showed "different
picks" from thread scheduling alone, which made optimisations look lossy when they
weren't — the pre-fix code could not reproduce its own picks.

**How to apply:** when benchmarking, still compare against a run of the *current* code
rather than an old copy, and use `bench/check_determinism.py` as the guard. If picks ever
start drifting again, suspect newly introduced dict-order dependence before suspecting
the scoring maths. See [[bird-cv-performance-profile]].
