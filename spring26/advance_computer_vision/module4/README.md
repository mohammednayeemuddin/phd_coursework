# Edge Detection

This project implements edge detection and object segmentation using OpenCV. The script identifies and extracts object boundaries from images using morphological operations and contour detection.

## Overview

The edge detection algorithm processes images through several stages:
1. **Grayscale conversion** - Converts color images to grayscale
2. **Denoising** - Applies Gaussian blur to reduce noise
3. **Thresholding** - Uses Otsu's method to automatically find the optimal threshold
4. **Morphological operations** - Cleans up the mask by filling holes and removing noise
5. **Contour detection** - Finds and extracts the largest object boundary
6. **Visualization** - Draws the boundary and bounding box on the original image

## Usage

```bash
python detection.py
```
## Requirements

Install dependencies:
```bash
pip install opencv-python
```
## Output Examples

The script generates:
- **output_mask.png**: Binary image showing white object on black background
- **output_boundary.png**: Original image with detected boundary highlighted
