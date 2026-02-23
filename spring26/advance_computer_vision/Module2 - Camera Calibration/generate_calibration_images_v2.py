"""
============================================================================
SIMULATED CALIBRATION IMAGE GENERATOR (v2 - Fixed)
============================================================================
Generates synthetic checkerboard images that OpenCV can properly detect.

Author: GSU CV Course
Usage: python generate_calibration_images_v2.py
============================================================================
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = "images/calibration"
NUM_IMAGES = 20

# Simulated camera (typical smartphone)
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1440
FOCAL_LENGTH = 1200

# Distortion coefficients
SIMULATED_K1 = -0.05
SIMULATED_K2 = 0.02
SIMULATED_P1 = 0.001
SIMULATED_P2 = 0.001

# Checkerboard: INNER CORNERS (what OpenCV detects)
INNER_COLS = 9  # inner corners in x
INNER_ROWS = 6  # inner corners in y
SQUARE_SIZE_MM = 30.0

# ============================================================================
# FUNCTIONS
# ============================================================================

def create_checkerboard_image(rows: int, cols: int, square_size: int = 80) -> np.ndarray:
    """
    Create a clean checkerboard pattern image.
    
    Args:
        rows: Number of INNER corner rows (so rows+1 squares vertically)
        cols: Number of INNER corner columns (so cols+1 squares horizontally)
        square_size: Size of each square in pixels
    """
    # We need (cols+1) x (rows+1) squares to have cols x rows inner corners
    n_squares_x = cols + 1
    n_squares_y = rows + 1
    
    # Add white border around the board
    border = square_size
    
    board_width = n_squares_x * square_size + 2 * border
    board_height = n_squares_y * square_size + 2 * border
    
    # Create white image
    img = np.ones((board_height, board_width), dtype=np.uint8) * 255
    
    # Draw black squares
    for i in range(n_squares_y):
        for j in range(n_squares_x):
            if (i + j) % 2 == 1:  # Black squares
                x1 = border + j * square_size
                y1 = border + i * square_size
                x2 = x1 + square_size
                y2 = y1 + square_size
                img[y1:y2, x1:x2] = 0
    
    return img


def get_checkerboard_corners_3d(rows: int, cols: int, 
                                 square_size: float) -> np.ndarray:
    """Get 3D coordinates of inner corners."""
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size
    # Center the board
    objp[:, 0] -= (cols - 1) * square_size / 2
    objp[:, 1] -= (rows - 1) * square_size / 2
    return objp


def create_camera_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Create camera intrinsic matrix."""
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def warp_checkerboard(board_img: np.ndarray, 
                       K: np.ndarray, 
                       dist_coeffs: np.ndarray,
                       rvec: np.ndarray, 
                       tvec: np.ndarray,
                       output_size: Tuple[int, int],
                       square_size_pixels: int,
                       inner_cols: int,
                       inner_rows: int) -> np.ndarray:
    """
    Warp checkerboard image to simulate camera view.
    """
    board_h, board_w = board_img.shape[:2]
    
    # Define the 4 corners of the board image
    border = square_size_pixels
    n_squares_x = inner_cols + 1
    n_squares_y = inner_rows + 1
    
    # Source points (corners of the checkerboard in the flat image)
    src_pts = np.array([
        [border, border],
        [border + n_squares_x * square_size_pixels, border],
        [border + n_squares_x * square_size_pixels, border + n_squares_y * square_size_pixels],
        [border, border + n_squares_y * square_size_pixels]
    ], dtype=np.float32)
    
    # 3D points of board corners (in mm, centered)
    half_w = n_squares_x * SQUARE_SIZE_MM / 2
    half_h = n_squares_y * SQUARE_SIZE_MM / 2
    
    board_corners_3d = np.array([
        [-half_w, -half_h, 0],
        [half_w, -half_h, 0],
        [half_w, half_h, 0],
        [-half_w, half_h, 0]
    ], dtype=np.float32)
    
    # Project 3D corners to 2D
    dst_pts, _ = cv2.projectPoints(board_corners_3d, rvec, tvec, K, dist_coeffs)
    dst_pts = dst_pts.reshape(-1, 2).astype(np.float32)
    
    # Check if projected points are within image bounds (with margin)
    margin = 50
    if (dst_pts[:, 0].min() < margin or 
        dst_pts[:, 0].max() > output_size[0] - margin or
        dst_pts[:, 1].min() < margin or 
        dst_pts[:, 1].max() > output_size[1] - margin):
        return None
    
    # Compute homography
    H, _ = cv2.findHomography(src_pts, dst_pts)
    
    # Warp the board
    output = cv2.warpPerspective(board_img, H, output_size, 
                                  borderMode=cv2.BORDER_CONSTANT, 
                                  borderValue=128)  # Gray background
    
    return output


def generate_random_pose(distance_range: Tuple[float, float] = (500, 900),
                          angle_range: float = 40) -> Tuple[np.ndarray, np.ndarray]:
    """Generate random camera pose."""
    # Random rotation
    rx = np.random.uniform(-angle_range, angle_range)
    ry = np.random.uniform(-angle_range, angle_range)
    rz = np.random.uniform(-10, 10)
    
    rx, ry, rz = np.radians([rx, ry, rz])
    
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    
    R = Rz @ Ry @ Rx
    rvec, _ = cv2.Rodrigues(R)
    
    # Random translation
    z = np.random.uniform(*distance_range)
    x = np.random.uniform(-80, 80)
    y = np.random.uniform(-80, 80)
    tvec = np.array([[x], [y], [z]], dtype=np.float64)
    
    return rvec, tvec


def add_noise(img: np.ndarray, noise_level: float = 3) -> np.ndarray:
    """Add Gaussian noise to image."""
    noise = np.random.normal(0, noise_level, img.shape)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def verify_detection(img: np.ndarray, pattern_size: Tuple[int, int]) -> bool:
    """Verify that OpenCV can detect corners in the image."""
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    return ret


def generate_dataset():
    """Generate complete calibration dataset."""
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path("images/dimensions").mkdir(parents=True, exist_ok=True)
    
    # Create base checkerboard image (high resolution)
    square_px = 100  # pixels per square in the source image
    board_img = create_checkerboard_image(INNER_ROWS, INNER_COLS, square_px)
    
    # Convert to 3-channel for warping
    board_img_color = cv2.cvtColor(board_img, cv2.COLOR_GRAY2BGR)
    
    # Camera matrix
    cx = IMAGE_WIDTH / 2 + np.random.uniform(-10, 10)
    cy = IMAGE_HEIGHT / 2 + np.random.uniform(-10, 10)
    K = create_camera_matrix(FOCAL_LENGTH, FOCAL_LENGTH * 1.005, cx, cy)
    
    # Distortion
    dist_coeffs = np.array([SIMULATED_K1, SIMULATED_K2, SIMULATED_P1, SIMULATED_P2, 0], 
                           dtype=np.float64)
    
    print("=" * 60)
    print("GENERATING CALIBRATION IMAGES")
    print("=" * 60)
    print(f"\nGround Truth Camera Matrix:")
    print(K)
    print(f"\nGround Truth Distortion: [k1={SIMULATED_K1}, k2={SIMULATED_K2}, p1={SIMULATED_P1}, p2={SIMULATED_P2}]")
    print(f"\nPattern: {INNER_COLS}x{INNER_ROWS} inner corners")
    print(f"Image size: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    print("-" * 60)
    
    pattern_size = (INNER_COLS, INNER_ROWS)
    generated = 0
    attempts = 0
    max_attempts = NUM_IMAGES * 10
    
    while generated < NUM_IMAGES and attempts < max_attempts:
        attempts += 1
        
        rvec, tvec = generate_random_pose()
        
        # Warp checkerboard
        warped = warp_checkerboard(
            board_img_color, K, dist_coeffs, rvec, tvec,
            (IMAGE_WIDTH, IMAGE_HEIGHT), square_px, INNER_COLS, INNER_ROWS
        )
        
        if warped is None:
            continue
        
        # Convert to grayscale and add noise
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        gray = add_noise(gray, noise_level=2)
        gray = cv2.GaussianBlur(gray, (3, 3), 0.5)
        
        # Verify OpenCV can detect it
        if not verify_detection(gray, pattern_size):
            continue
        
        # Save
        filename = f"calib_{generated+1:03d}.jpg"
        filepath = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(filepath, gray)
        
        generated += 1
        print(f"  [{generated:2d}/{NUM_IMAGES}] {filename} ✓")
    
    print("-" * 60)
    print(f"Generated {generated} valid images")
    
    # Save ground truth
    gt_path = os.path.join(OUTPUT_DIR, "ground_truth.npz")
    np.savez(gt_path, K=K, dist=dist_coeffs, 
             pattern_size=np.array(pattern_size),
             square_size=SQUARE_SIZE_MM)
    print(f"Ground truth saved: {gt_path}")
    
    # Generate test object
    generate_test_object(K, dist_coeffs)
    
    return K, dist_coeffs


def generate_test_object(K: np.ndarray, dist_coeffs: np.ndarray):
    """Generate test object image for dimension measurement."""
    
    obj_width_cm = 20.0
    obj_height_cm = 13.0
    distance_m = 2.5
    
    # Convert to mm
    w_mm, h_mm = obj_width_cm * 10, obj_height_cm * 10
    z_mm = distance_m * 1000
    
    # 3D corners
    obj_3d = np.array([
        [-w_mm/2, -h_mm/2, 0],
        [w_mm/2, -h_mm/2, 0],
        [w_mm/2, h_mm/2, 0],
        [-w_mm/2, h_mm/2, 0]
    ], dtype=np.float32)
    
    # Slight angle
    rvec = np.array([[0.08], [0.05], [0.02]], dtype=np.float64)
    tvec = np.array([[0], [0], [z_mm]], dtype=np.float64)
    
    # Project
    corners_2d, _ = cv2.projectPoints(obj_3d, rvec, tvec, K, dist_coeffs)
    corners_2d = corners_2d.reshape(-1, 2).astype(np.int32)
    
    # Create image
    img = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8) * 180
    
    # Draw rectangle
    cv2.fillPoly(img, [corners_2d], (80, 140, 200))
    cv2.polylines(img, [corners_2d], True, (40, 80, 160), 3)
    
    # Add noise
    img = add_noise(img, 4)
    
    filepath = "images/dimensions/test_object.jpg"
    cv2.imwrite(filepath, img)
    
    corners_list = [tuple(c) for c in corners_2d.tolist()]
    
    print(f"\nTest object saved: {filepath}")
    print(f"Size: {obj_width_cm} x {obj_height_cm} cm at {distance_m} m")
    print(f"Corners: {corners_list}")
    
    # Save test info
    np.savez("images/dimensions/test_info.npz",
             corners=corners_2d,
             width_cm=obj_width_cm,
             height_cm=obj_height_cm,
             distance_m=distance_m)


def visualize_samples(n_show: int = 6):
    """Display and verify sample images."""
    images = sorted(Path(OUTPUT_DIR).glob("calib_*.jpg"))[:n_show]
    pattern_size = (INNER_COLS, INNER_ROWS)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, ax in enumerate(axes):
        if idx < len(images):
            img = cv2.imread(str(images[idx]), cv2.IMREAD_GRAYSCALE)
            
            # Detect corners
            ret, corners = cv2.findChessboardCorners(img, pattern_size, None)
            
            # Draw
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if ret:
                cv2.drawChessboardCorners(vis, pattern_size, corners, ret)
            
            ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            status = "✓ DETECTED" if ret else "✗ FAILED"
            ax.set_title(f"{images[idx].name}\n{status}", 
                        color='green' if ret else 'red')
        ax.axis('off')
    
    plt.suptitle(f'Generated Images with Corner Detection\nPattern: {INNER_COLS}x{INNER_ROWS}', 
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "preview_with_corners.png"), dpi=100)
    plt.show()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    K, dist = generate_dataset()
    
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"""
UPDATE YOUR NOTEBOOK WITH:

PATTERN_SIZE = ({INNER_COLS}, {INNER_ROWS})
SQUARE_SIZE_MM = {SQUARE_SIZE_MM}
IMAGES_PATH = "images/calibration/*.jpg"

For Step 3 (test object):
TEST_IMAGE_PATH = "images/dimensions/test_object.jpg"
OBJECT_DISTANCE_M = 2.5
EXPECTED_WIDTH_CM = 20.0
EXPECTED_HEIGHT_CM = 13.0

Check images/dimensions/test_info.npz for corner coordinates.
""")
    
    visualize_samples()
