"""
CSc 8830: Computer Vision - Assignment 6
Part B: Structure from Motion (SfM) from 4 Viewpoints

README:
    This script demonstrates Structure from Motion using synthetic data
    from 4 camera positions observing a planar 2D object (a rectangle/
    star shape). It reconstructs 3D points and estimates the object
    boundary using homography and triangulation.

Usage:
    python structure_from_motion.py

Dependencies:
    pip install opencv-python-headless numpy matplotlib scipy

Output:
    - sfm_views.png              : The 4 camera views with projected points
    - sfm_reconstruction.png     : 3D reconstruction and boundary estimate
    - sfm_epipolar.png           : Epipolar geometry visualization
    - sfm_camera_setup.png       : Camera arrangement diagram

References:
    - Hartley, R., & Zisserman, A. (2003). Multiple View Geometry in Computer Vision.
    - Szeliski, R. (2010). Computer Vision: Algorithms and Applications.
    - Faugeras, O. (1993). Three-Dimensional Computer Vision.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import cv2
import os


# ─────────────────────────────────────────────
# 1. DEFINE 3D OBJECT POINTS (Planar 2D object)
# ─────────────────────────────────────────────

def create_object_points():
    """
    Create a flat planar 2D object (a stylized hexagon/star polygon)
    lying in the Z=0 plane. Returns Nx3 array.
    """
    # 12-point star-like polygon on the Z=0 plane
    angles_outer = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    angles_inner = angles_outer + np.pi / 8
    r_outer = 1.0
    r_inner = 0.5

    pts = []
    for ao, ai in zip(angles_outer, angles_inner):
        pts.append([r_outer * np.cos(ao), r_outer * np.sin(ao), 0.0])
        pts.append([r_inner * np.cos(ai), r_inner * np.sin(ai), 0.0])

    # Add center
    pts.append([0.0, 0.0, 0.0])

    return np.array(pts, dtype=np.float64)


# ─────────────────────────────────────────────
# 2. CAMERA SETUP
# ─────────────────────────────────────────────

def make_camera_matrix(fx=800, fy=800, cx=320, cy=240):
    """Intrinsic camera matrix K."""
    K = np.array([[fx,  0, cx],
                  [ 0, fy, cy],
                  [ 0,  0,  1]], dtype=np.float64)
    return K


def make_rotation(yaw_deg, pitch_deg=0.0):
    """Rotation matrix from yaw + pitch (degrees)."""
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)

    Ry = np.array([[np.cos(yaw),  0, np.sin(yaw)],
                   [0,            1, 0           ],
                   [-np.sin(yaw), 0, np.cos(yaw)]])

    Rx = np.array([[1, 0,             0           ],
                   [0, np.cos(pitch), -np.sin(pitch)],
                   [0, np.sin(pitch),  np.cos(pitch)]])
    return Rx @ Ry


def setup_cameras():
    """
    Four cameras arranged around the object at different yaw angles.
    Camera 0: front (0°), Camera 1: left (90°),
    Camera 2: back (180°), Camera 3: right side angle (45°)
    Each camera looks at origin from distance ~4 units.
    Returns list of dicts with K, R, t, P.
    """
    K = make_camera_matrix()
    configs = [
        {"yaw": 0,   "pitch": 10,  "dist": 4.0, "label": "Front"},
        {"yaw": 75,  "pitch": 15,  "dist": 4.2, "label": "Left"},
        {"yaw": 155, "pitch": 8,   "dist": 3.8, "label": "Rear-Left"},
        {"yaw": 45,  "pitch": 20,  "dist": 4.5, "label": "Front-Right"},
    ]
    cameras = []
    for cfg in configs:
        R = make_rotation(cfg["yaw"], cfg["pitch"])
        # Camera center in world
        cam_center = np.array([
            cfg["dist"] * np.sin(np.radians(cfg["yaw"])),
            -0.5,
            cfg["dist"] * np.cos(np.radians(cfg["yaw"]))])
        t = -R @ cam_center
        P = K @ np.hstack([R, t.reshape(3, 1)])
        cameras.append({"K": K, "R": R, "t": t,
                        "P": P, "center": cam_center,
                        "label": cfg["label"]})
    return cameras


# ─────────────────────────────────────────────
# 3. PROJECT POINTS TO IMAGE
# ─────────────────────────────────────────────

def project_points(P, X3d):
    """Project Nx3 world points through 3x4 projection matrix P → Nx2 image coords."""
    N = X3d.shape[0]
    X_h = np.hstack([X3d, np.ones((N, 1))])   # Nx4 homogeneous
    x_h = (P @ X_h.T).T                         # Nx3
    x = x_h[:, :2] / x_h[:, 2:3]
    return x


# ─────────────────────────────────────────────
# 4. VISUALIZE THE 4 VIEWS
# ─────────────────────────────────────────────

def visualize_views(cameras, X3d, out_path):
    """Show the 4 synthetic camera images with projected object points."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    colors = plt.cm.plasma(np.linspace(0, 1, len(X3d)))

    for i, cam in enumerate(cameras):
        pts2d = project_points(cam["P"], X3d)

        # Draw synthetic image background
        img = np.ones((480, 640, 3)) * 0.08

        axes[i].imshow(img, extent=[0, 640, 480, 0])
        axes[i].scatter(pts2d[:, 0], pts2d[:, 1],
                        c=colors, s=60, zorder=5, edgecolors='white', linewidths=0.5)

        # Connect boundary points (skip center)
        boundary = np.vstack([pts2d[:-1], pts2d[0]])
        axes[i].plot(boundary[:, 0], boundary[:, 1],
                     'c-', alpha=0.6, linewidth=1.5)

        # Annotate point indices
        for j, (px, py) in enumerate(pts2d[:-1:2]):  # outer ring only
            axes[i].annotate(str(j // 2), (px, py),
                             textcoords="offset points", xytext=(4, 4),
                             fontsize=7, color='white')

        axes[i].set_xlim(0, 640)
        axes[i].set_ylim(480, 0)
        axes[i].set_title(f'Camera {i+1}: {cam["label"]}\n'
                          f'R yaw≈{["0°","75°","155°","45°"][i]}, '
                          f'center≈{cam["center"].round(2)}',
                          fontsize=10)
        axes[i].set_xlabel('u (pixels)')
        axes[i].set_ylabel('v (pixels)')

    fig.suptitle('Part B — Structure from Motion: 4 Camera Views\n'
                 '(Synthetic planar object projected to each view)', fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[✓] Camera views saved: {out_path}")


# ─────────────────────────────────────────────
# 5. TRIANGULATION (Linear DLT)
# ─────────────────────────────────────────────

def triangulate_dlt(P1, p1, P2, p2):
    """
    Linear triangulation (DLT) for a single point correspondence.
    Given projection matrices P1, P2 and 2D points p1, p2 (in pixels),
    returns the 3D point X in world coordinates.
    """
    A = np.array([
        p1[0] * P1[2] - P1[0],
        p1[1] * P1[2] - P1[1],
        p2[0] * P2[2] - P2[0],
        p2[1] * P2[2] - P2[1],
    ])
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]


def reconstruct_all_points(cameras, X3d_gt):
    """
    Triangulate all points using camera pairs (0,1), (0,2), (0,3)
    then average for robustness.
    """
    views = [project_points(cam["P"], X3d_gt) for cam in cameras]
    N = X3d_gt.shape[0]
    recon = []
    pairs = [(0, 1), (0, 2), (0, 3)]

    for j in range(N):
        pts = []
        for i1, i2 in pairs:
            X = triangulate_dlt(cameras[i1]["P"], views[i1][j],
                                cameras[i2]["P"], views[i2][j])
            pts.append(X)
        recon.append(np.mean(pts, axis=0))
    return np.array(recon)


# ─────────────────────────────────────────────
# 6. VISUALIZE RECONSTRUCTION
# ─────────────────────────────────────────────

def visualize_reconstruction(X3d_gt, X3d_recon, cameras, out_path):
    """3D plot: ground truth vs reconstructed, with camera positions and boundary."""
    fig = plt.figure(figsize=(14, 6))

    # ── Left: 3D reconstruction ──
    ax1 = fig.add_subplot(121, projection='3d')

    # Ground truth
    ax1.scatter(X3d_gt[:, 0], X3d_gt[:, 1], X3d_gt[:, 2],
                c='cyan', s=60, label='GT Points', depthshade=False)
    # Reconstructed
    ax1.scatter(X3d_recon[:, 0], X3d_recon[:, 1], X3d_recon[:, 2],
                c='orange', s=40, marker='^', label='Reconstructed', depthshade=False,
                alpha=0.9)
    # Boundary line GT
    bd = np.vstack([X3d_gt[:-1], X3d_gt[0]])
    ax1.plot(bd[:, 0], bd[:, 1], bd[:, 2], 'c-', alpha=0.5)

    # Camera positions
    for i, cam in enumerate(cameras):
        cc = cam["center"]
        ax1.scatter(*cc, c='red', s=120, marker='s', zorder=10)
        ax1.text(cc[0], cc[1], cc[2] + 0.2, f'C{i+1}', fontsize=9, color='red')
        ax1.plot([cc[0], 0], [cc[1], 0], [cc[2], 0], 'r--', alpha=0.25)

    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.set_title('3D Reconstruction\n(GT=cyan, Recon=orange, Cameras=red)', fontsize=10)
    ax1.legend(fontsize=8)

    # ── Right: Reprojection error ──
    ax2 = fig.add_subplot(122)
    errors = np.linalg.norm(X3d_gt - X3d_recon, axis=1)
    colors = plt.cm.RdYlGn_r(errors / (errors.max() + 1e-9))
    bars = ax2.bar(range(len(errors)), errors, color=colors, edgecolor='white')
    ax2.set_title('3D Reconstruction Error per Point\n|GT − Reconstructed| (world units)',
                  fontsize=11)
    ax2.set_xlabel('Point Index')
    ax2.set_ylabel('Euclidean Error (world units)')
    ax2.axhline(np.mean(errors), color='red', linestyle='--',
                label=f'Mean={np.mean(errors):.4f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Part B — SfM Reconstruction Results', fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[✓] Reconstruction plot saved: {out_path}")
    return errors


# ─────────────────────────────────────────────
# 7. CAMERA SETUP DIAGRAM
# ─────────────────────────────────────────────

def visualize_camera_setup(cameras, X3d_gt, out_path):
    """Top-down view of camera arrangement around the object."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw object boundary (top-down = XZ plane since object is in Z=0)
    obj_x = X3d_gt[:-1, 0]
    obj_z = X3d_gt[:-1, 1]
    ax.fill(obj_x, obj_z, alpha=0.3, color='cyan', label='Object')
    ax.plot(np.append(obj_x, obj_x[0]),
            np.append(obj_z, obj_z[0]), 'c-', linewidth=2)
    ax.scatter(obj_x, obj_z, c='cyan', s=40, zorder=4)

    # Camera icons
    cam_colors = ['red', 'green', 'orange', 'purple']
    for i, cam in enumerate(cameras):
        cx, cy_ = cam["center"][0], cam["center"][1]
        ax.scatter(cx, cy_, c=cam_colors[i], s=200,
                   marker='s', zorder=5, label=f'C{i+1}: {cam["label"]}')
        ax.annotate(f'C{i+1}', (cx, cy_), xytext=(cx + 0.1, cy_ + 0.1),
                    fontsize=11, fontweight='bold', color=cam_colors[i])
        # Draw "look-at" arrow toward origin
        ax.annotate('', xy=(0, 0), xytext=(cx * 0.85, cy_ * 0.85),
                    arrowprops=dict(arrowstyle='->', color=cam_colors[i],
                                   lw=1.5, alpha=0.6))

    ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)
    ax.axhline(0, color='gray', alpha=0.3); ax.axvline(0, color='gray', alpha=0.3)
    ax.set_xlabel('World X'); ax.set_ylabel('World Y')
    ax.set_title('Camera Setup — Top-Down View\n(4 viewpoints around planar object)',
                 fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[✓] Camera setup diagram saved: {out_path}")


# ─────────────────────────────────────────────
# 8. EPIPOLAR GEOMETRY VISUALIZATION
# ─────────────────────────────────────────────

def visualize_epipolar(cameras, X3d_gt, out_path):
    """Show epipolar lines between Camera 1 and Camera 2."""
    P1, P2 = cameras[0]["P"], cameras[1]["P"]
    K = cameras[0]["K"]
    R1, t1 = cameras[0]["R"], cameras[0]["t"]
    R2, t2 = cameras[1]["R"], cameras[1]["t"]

    pts1 = project_points(P1, X3d_gt)
    pts2 = project_points(P2, X3d_gt)

    # Essential matrix E = [t]_x R  (relative pose)
    R_rel = R2 @ R1.T
    t_rel = t2 - R_rel @ t1
    tx = np.array([[0, -t_rel[2], t_rel[1]],
                   [t_rel[2], 0, -t_rel[0]],
                   [-t_rel[1], t_rel[0], 0]])
    E = tx @ R_rel
    # Fundamental matrix F = K^{-T} E K^{-1}
    K_inv = np.linalg.inv(K)
    F = K_inv.T @ E @ K_inv

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    for ax, pts, view_idx in zip(axes, [pts1, pts2], [0, 1]):
        img = np.ones((480, 640, 3)) * 0.08
        axes[view_idx].imshow(img, extent=[0, 640, 480, 0])
        axes[view_idx].scatter(pts[:, 0], pts[:, 1],
                               c='cyan', s=50, zorder=5)
        axes[view_idx].set_xlim(0, 640)
        axes[view_idx].set_ylim(480, 0)

    # Draw epipolar lines in image 2 for each point in image 1
    u_range = np.linspace(0, 640, 200)
    for j in range(min(6, len(pts1))):
        p1_h = np.array([pts1[j, 0], pts1[j, 1], 1.0])
        l2 = F @ p1_h   # epipolar line in view 2: l2^T x2 = 0
        # l2 = [a, b, c] → au + bv + c = 0 → v = -(au + c)/b
        if abs(l2[1]) > 1e-6:
            v_vals = -(l2[0] * u_range + l2[2]) / l2[1]
            mask = (v_vals >= 0) & (v_vals <= 480)
            axes[1].plot(u_range[mask], v_vals[mask],
                         alpha=0.5, linewidth=1.2,
                         color=plt.cm.Set1(j / 6.0))
        axes[0].scatter(*pts1[j], s=80, c=[plt.cm.Set1(j / 6.0)], zorder=6)
        axes[1].scatter(*pts2[j], s=80, c=[plt.cm.Set1(j / 6.0)],
                        zorder=6, marker='^')

    axes[0].set_title('Camera 1 (Front)\nSource Points', fontsize=11)
    axes[1].set_title('Camera 2 (Left)\nEpipolar Lines + Corresponding Points', fontsize=11)
    for ax in axes:
        ax.set_xlabel('u (pixels)')
        ax.set_ylabel('v (pixels)')

    fig.suptitle('Part B — Epipolar Geometry (F matrix from cameras 1 & 2)', fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[✓] Epipolar geometry saved: {out_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    print("\n=== PART B: STRUCTURE FROM MOTION ===\n")

    # Object and cameras
    X3d_gt = create_object_points()
    cameras = setup_cameras()

    print(f"  Object points: {X3d_gt.shape[0]}")
    print(f"  Cameras: {len(cameras)}")
    for i, cam in enumerate(cameras):
        print(f"    Camera {i+1} ({cam['label']}): center={cam['center'].round(3)}")

    # Visualize 4 views
    visualize_views(cameras, X3d_gt, "output/sfm_views.png")

    # Triangulate / reconstruct
    X3d_recon = reconstruct_all_points(cameras, X3d_gt)
    errors = visualize_reconstruction(X3d_gt, X3d_recon, cameras,
                                       "output/sfm_reconstruction.png")

    print(f"\n  Reconstruction errors (world units):")
    print(f"    Mean : {np.mean(errors):.6f}")
    print(f"    Max  : {np.max(errors):.6f}")
    print(f"    Min  : {np.min(errors):.6f}")

    # Camera setup diagram
    visualize_camera_setup(cameras, X3d_gt, "output/sfm_camera_setup.png")

    # Epipolar geometry
    visualize_epipolar(cameras, X3d_gt, "output/sfm_epipolar.png")

    print("\n[✓] Part B complete. All outputs in ./output/\n")
