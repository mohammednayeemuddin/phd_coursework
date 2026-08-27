---
name: acv-final-project-featheridentify
description: The Spring 2026 Advanced Computer Vision final project is FeatherIdentify, a pure-CV bird photo curator with a strict no-ML constraint.
metadata:
  type: project
---

The final project for Advanced Computer Vision (Spring 2026 term, PhD coursework) lives
in `spring26/advance_computer_vision/Final Project/` and is called FeatherIdentify —
`bird_cv.py` buckets a folder of wildlife photos by visual similarity, labels each bucket
with a bird family, and quality-ranks the shots; `resize_photos.py` is an optional
pre-step resizer.

**Why:** the whole point of the deliverable is that it is *pure* classical computer
vision — OpenCV + NumPy heuristics only, no ML models, no pretrained weights, no GPU, no
internet. Reaching for a detector or classifier would defeat the assignment even though
it would score better. Species-level ID is knowingly out of scope; family level is the
stated ceiling.

**How to apply:** when improving accuracy, propose classical-CV changes (HSV threshold
retuning, new contour/edge features, different clustering linkage) and never suggest
adding torch/tensorflow, YOLO, or any downloaded model. The skill
`.claude/skills/bird-photo-curator/SKILL.md` holds the operational detail. See
[[bird-cv-readme-code-drift]].
