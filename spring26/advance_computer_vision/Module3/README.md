# Image Blurring - Spatial vs Fourier Domain

This project demonstrates the **Convolution Theorem** by implementing and comparing image blurring in both spatial and Fourier domains using Gaussian kernels.

## Overview

The Convolution Theorem states that convolution in the spatial domain is equivalent to multiplication in the frequency domain. This implementation verifies this principle by applying Gaussian blur to images using two methods:

1. **Spatial Domain**: Direct convolution using `cv2.filter2D()` with periodic boundaries
2. **Fourier Domain**: FFT-based multiplication using `scipy.fft`

## Requirements

```bash
pip install numpy opencv-python matplotlib scipy
```

## Usage

```bash
python blur.py
```

The script will:
1. Load an image from the `images/` directory
2. Apply Gaussian blur using both spatial and Fourier methods
3. Compute and display MSE and max difference between the results
4. Generate a comparison visualization saved as `real_image_comparison.png`
