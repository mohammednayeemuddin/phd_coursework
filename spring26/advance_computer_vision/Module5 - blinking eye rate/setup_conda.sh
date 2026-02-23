#!/bin/bash
# Setup script for Blink Detector with Conda

# Step 1: Create conda environment
conda create -n blink_detector python=3.10 -y

# Step 2: Activate environment
conda activate blink_detector

# Step 3: Install dependencies
pip install opencv-python mediapipe numpy

# Step 4: Run the detector
python blink_detector_live.py
