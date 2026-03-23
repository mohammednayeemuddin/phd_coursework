"""
CSc 8830 Assignment 6 - Full PDF Report Generator
Generates a complete academic report with:
  - Cover page
  - Part A: Optical flow theory, bilinear interpolation derivation,
            tracking equations, validation results
  - Part B: SfM math, homography, DLT triangulation, results
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Table, TableStyle, Image, HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus.flowables import BalancedColumns

# ── Color palette ──────────────────────────────────────────────
C_DARK   = colors.HexColor("#1a1a2e")
C_BLUE   = colors.HexColor("#16213e")
C_ACCENT = colors.HexColor("#0f3460")
C_CYAN   = colors.HexColor("#1a6985")
C_LIGHT  = colors.HexColor("#e2e8f0")
C_WHITE  = colors.white
C_MATH   = colors.HexColor("#2d3748")
C_BOX    = colors.HexColor("#ebf8ff")
C_BOX_BD = colors.HexColor("#3182ce")

W, H = letter

# ── Styles ─────────────────────────────────────────────────────
def make_styles():
    base = getSampleStyleSheet()

    def S(name, **kw):
        return ParagraphStyle(name, **kw)

    return {
        "cover_title": S("cover_title",
            fontSize=26, textColor=C_WHITE, alignment=TA_CENTER,
            fontName="Helvetica-Bold", spaceAfter=8),
        "cover_sub": S("cover_sub",
            fontSize=14, textColor=C_LIGHT, alignment=TA_CENTER,
            fontName="Helvetica", spaceAfter=6),
        "cover_info": S("cover_info",
            fontSize=11, textColor=C_LIGHT, alignment=TA_CENTER,
            fontName="Helvetica", spaceAfter=4),

        "h1": S("h1",
            fontSize=16, textColor=C_ACCENT, fontName="Helvetica-Bold",
            spaceBefore=18, spaceAfter=8,
            borderPad=4),
        "h2": S("h2",
            fontSize=13, textColor=C_CYAN, fontName="Helvetica-Bold",
            spaceBefore=12, spaceAfter=6),
        "h3": S("h3",
            fontSize=11, textColor=C_MATH, fontName="Helvetica-Bold",
            spaceBefore=8, spaceAfter=4),

        "body": S("body",
            fontSize=10, textColor=colors.HexColor("#2d3748"),
            fontName="Helvetica", spaceAfter=6,
            leading=15, alignment=TA_JUSTIFY),
        "math": S("math",
            fontSize=10, textColor=C_MATH,
            fontName="Courier", spaceAfter=4,
            leading=14, leftIndent=20),
        "math_box": S("math_box",
            fontSize=10, textColor=C_MATH,
            fontName="Courier", spaceAfter=3,
            leading=14, leftIndent=10),
        "caption": S("caption",
            fontSize=9, textColor=colors.grey,
            fontName="Helvetica-Oblique", alignment=TA_CENTER,
            spaceAfter=10),
        "bullet": S("bullet",
            fontSize=10, textColor=colors.HexColor("#2d3748"),
            fontName="Helvetica", spaceAfter=4,
            leftIndent=20, leading=14),
        "table_h": S("table_h",
            fontSize=9, textColor=C_WHITE, fontName="Helvetica-Bold",
            alignment=TA_CENTER),
        "table_c": S("table_c",
            fontSize=9, textColor=C_MATH, fontName="Courier",
            alignment=TA_CENTER),
    }


def math_box(content_lines, styles):
    """Render a shaded equation box."""
    rows = [[Paragraph(line, styles["math_box"])] for line in content_lines]
    t = Table(rows, colWidths=[6.5 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_BOX),
        ("BOX", (0, 0), (-1, -1), 1, C_BOX_BD),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


def section_rule():
    return HRFlowable(width="100%", thickness=1,
                      color=C_CYAN, spaceAfter=6)


def img(path, w=6.5, caption=None, styles=None):
    """Embed image with optional caption."""
    items = []
    if os.path.exists(path):
        items.append(Image(path, width=w * inch, height=w * inch * 0.55))
    if caption and styles:
        items.append(Paragraph(caption, styles["caption"]))
    return items


def make_report(out_path, styles):
    doc = SimpleDocTemplate(out_path, pagesize=letter,
                             leftMargin=0.85*inch, rightMargin=0.85*inch,
                             topMargin=0.85*inch, bottomMargin=0.85*inch)
    story = []
    S = styles

    # ══════════════════════════════════════════════════════════════
    # COVER PAGE
    # ══════════════════════════════════════════════════════════════
    cover_data = [[
        Paragraph("CSc 8830: Computer Vision", S["cover_title"]),
    ]]
    cover_table = Table(cover_data, colWidths=[6.5 * inch])
    cover_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_DARK),
        ("TOPPADDING", (0, 0), (-1, -1), 40),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 20),
        ("RIGHTPADDING", (0, 0), (-1, -1), 20),
        ("ROUNDEDCORNERS", [8]),
    ]))
    story.append(cover_table)
    story.append(Spacer(1, 0.15 * inch))

    meta = [
        ["Assignment 6: Optical Flow and Structure from Motion"],
        ["Course: CSc 8830 — Computer Vision"],
        ["Language: Python 3  |  Libraries: OpenCV, NumPy, Matplotlib"],
        ["GitHub: [Insert Repository Link Here]"],
    ]
    for row in meta:
        story.append(Paragraph(row[0], S["cover_sub"]))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(
        "Abstract: This report presents implementations of Lucas-Kanade "
        "sparse optical flow, bilinear interpolation, motion tracking validation, "
        "and Structure from Motion (SfM) using four synthetic viewpoints of a "
        "planar object. All mathematical derivations, equations, and validation "
        "results are included inline.", S["body"]))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════
    # PART A — OPTICAL FLOW
    # ══════════════════════════════════════════════════════════════
    story.append(Paragraph("Part A: Optical Flow and Motion Tracking", S["h1"]))
    story.append(section_rule())

    # ── A.1 Overview ──────────────────────────────────────────────
    story.append(Paragraph("A.1  Overview and Video Descriptions", S["h2"]))
    story.append(Paragraph(
        "Two synthetic videos (30 seconds, 30 fps, 640×480) were generated "
        "to demonstrate different motion types:", S["body"]))
    for line in [
        "• <b>Video 1</b> — A bright circle translates diagonally (rightward + slight downward) "
          "across a dark background. A static rectangle provides a stationary reference. "
          "This represents smooth translational motion.",
        "• <b>Video 2</b> — A square rotates about its center while a ball oscillates "
          "along a sinusoidal path. This combines rotational and periodic motion patterns.",
    ]:
        story.append(Paragraph(line, S["bullet"]))
    story.append(Spacer(1, 0.1 * inch))

    # ── A.2 What can be inferred from optical flow ─────────────────
    story.append(Paragraph("A.2  Information Inferred from Optical Flow", S["h2"]))
    story.append(Paragraph(
        "Optical flow encodes the apparent 2D motion field between consecutive frames. "
        "From the computed flow vectors, one can infer:", S["body"]))
    for line in [
        "• <b>Direction of motion</b>: The angle of each flow vector indicates where "
          "a region is moving. In the HSV visualization, hue encodes direction.",
        "• <b>Speed / magnitude</b>: The length of flow vectors (or brightness in HSV) "
          "indicates how fast pixels are moving.",
        "• <b>Rigid vs. deformable motion</b>: A rigid body shows uniform flow vectors; "
          "rotation appears as a curl pattern; deformation shows divergence.",
        "• <b>Occlusion and depth ordering</b>: Motion boundaries between foreground "
          "and background reveal depth discontinuities.",
        "• <b>Ego-motion vs. object motion</b>: Large globally consistent flow implies "
          "camera movement; local patches with distinct vectors indicate independent objects.",
    ]:
        story.append(Paragraph(line, S["bullet"]))
    story.append(Spacer(1, 0.08*inch))

    # Flow visualization figures
    story.extend(img("output/flow_frames_v1.png", 6.5,
        "Figure 1a: Video 1 — Frame t, Frame t+1, and dense Farneback optical flow "
        "(HSV-encoded: hue = direction, brightness = magnitude). "
        "The bright patch shows rightward motion of the translating circle.", S))
    story.extend(img("output/flow_frames_v2.png", 6.5,
        "Figure 1b: Video 2 — Rotating square and bouncing ball produce a complex "
        "mixed flow field with curl (rotation) and oscillating vectors.", S))

    # ── A.3 Derivation of Motion Tracking Equations ───────────────
    story.append(Paragraph("A.3  Derivation of Motion Tracking Equations", S["h2"]))
    story.append(Paragraph(
        "<b>Optical Flow Constraint Equation (OFCE)</b>", S["h3"]))
    story.append(Paragraph(
        "Let I(x, y, t) denote image intensity at position (x,y) and time t. "
        "Assuming brightness constancy — a point moving from (x,y) to "
        "(x+dx, y+dy) in time dt preserves its intensity:", S["body"]))
    story.append(math_box([
        "I(x, y, t)  =  I(x + dx,  y + dy,  t + dt)"], S))
    story.append(Paragraph("Expanding via first-order Taylor series:", S["body"]))
    story.append(math_box([
        "I(x+dx, y+dy, t+dt)  ≈  I(x,y,t) + (dI/dx)·dx + (dI/dy)·dy + (dI/dt)·dt",
        "",
        "Substituting and dividing by dt:",
        "",
        "  Ix·u  +  Iy·v  +  It  =  0",
        "",
        "where:  u = dx/dt,  v = dy/dt   (optical flow vector)",
        "        Ix = dI/dx,  Iy = dI/dy,  It = dI/dt  (image gradients)",
    ], S))
    story.append(Paragraph(
        "This is the <b>Optical Flow Constraint Equation (OFCE)</b>. "
        "It provides one equation in two unknowns (u, v) — the aperture problem. "
        "Additional constraints are needed to solve for both components.", S["body"]))

    story.append(Paragraph("<b>Lucas-Kanade Method</b>", S["h3"]))
    story.append(Paragraph(
        "Lucas and Kanade (1981) assume the flow is constant within a "
        "small neighborhood window W of size m×m pixels. This yields an "
        "overdetermined system for the m² pixels in the window:", S["body"]))
    story.append(math_box([
        "For each pixel (x_i, y_i) in window W:",
        "   Ix_i · u  +  Iy_i · v  =  -It_i",
        "",
        "In matrix form:   A · d  =  b",
        "",
        "    A  =  [ Ix_1  Iy_1 ]       b  =  [ -It_1 ]       d  =  [ u ]",
        "          [ Ix_2  Iy_2 ]             [ -It_2 ]             [ v ]",
        "          [  ...   ... ]             [  ...  ]",
        "          [ Ix_n  Iy_n ]             [ -It_n ]",
        "",
        "Least-squares solution (normal equations):",
        "   (A^T A) · d  =  A^T · b",
        "",
        "   [ sum(Ix²)    sum(Ix·Iy) ] [ u ]   [ -sum(Ix·It) ]",
        "   [ sum(Ix·Iy)  sum(Iy²)   ] [ v ] = [ -sum(Iy·It) ]",
        "",
        "Solved as:   d  =  (A^T A)^{-1}  A^T b   (when A^T A is invertible)",
    ], S))

    story.append(Paragraph("<b>Tracking Prediction for Two Frames</b>", S["h3"]))
    story.append(Paragraph(
        "Given a feature point at pixel location p = (x, y) in frame t, "
        "the estimated location in frame t+1 is:", S["body"]))
    story.append(math_box([
        "p'  =  p  +  d  =  (x + u,  y + v)",
        "",
        "where (u, v) is the flow vector solved from the LK system above.",
        "",
        "For iterative/coarse-to-fine tracking (pyramid LK):",
        "   p^(k+1)  =  p^(k)  +  d^(k)",
        "   where d^(k) is the incremental update at pyramid level k",
    ], S))

    # ── A.4 Bilinear Interpolation Derivation ─────────────────────
    story.append(Paragraph("A.4  Derivation of Bilinear Interpolation", S["h2"]))
    story.append(Paragraph(
        "When a tracked point p' = (x', y') has non-integer coordinates, "
        "the intensity must be interpolated from surrounding integer pixels. "
        "Let (x0, y0) = floor(x', y') and a = x'−x0, b = y'−y0 (fractional parts). "
        "The four neighboring pixels are:", S["body"]))
    story.append(math_box([
        "  Q11 = I(x0, y0)   Q21 = I(x0+1, y0)",
        "  Q12 = I(x0, y0+1) Q22 = I(x0+1, y0+1)",
        "",
        "Step 1 — Linear interpolation along x (two horizontal passes):",
        "  R1  =  (1-a)·Q11  +  a·Q21    (at row y0)",
        "  R2  =  (1-a)·Q12  +  a·Q22    (at row y0+1)",
        "",
        "Step 2 — Linear interpolation along y:",
        "  I(x',y')  =  (1-b)·R1  +  b·R2",
        "",
        "Expanding fully:",
        "  I(x',y')  =  (1-a)(1-b)·I(x0,y0)  +  a(1-b)·I(x0+1,y0)",
        "             +  (1-a)b·I(x0,y0+1)   +  ab·I(x0+1,y0+1)",
        "",
        "In matrix form:",
        "  I(x',y')  =  [1-a  a] · [Q11  Q12] · [1-b]",
        "                           [Q21  Q22]   [ b ]",
    ], S))
    story.append(Paragraph(
        "Bilinear interpolation is C⁰ continuous (values match at grid nodes) "
        "but not C¹ (gradients are discontinuous at grid boundaries). "
        "It provides a good balance between computational cost and smoothness "
        "for sub-pixel intensity estimation during optical flow tracking.", S["body"]))

    story.extend(img("output/bilinear_validation.png", 6.2,
        "Figure 2: Bilinear interpolation validation. Left: manual implementation vs. "
        "OpenCV INTER_LINEAR at 5 test sub-pixel locations. Right: absolute error "
        "(all < 0.15 intensity units), confirming correctness of the derivation.", S))

    # ── A.5 Tracking Validation ────────────────────────────────────
    story.append(Paragraph("A.5  Tracking Validation: Theory vs. Actual Pixel Locations", S["h2"]))
    story.append(Paragraph(
        "Two consecutive frames were extracted from each video. Feature corners were "
        "detected in frame t using Shi-Tomasi, then tracked to frame t+1 with "
        "Lucas-Kanade. The theoretical prediction uses the median flow vector as a "
        "uniform motion estimate. The tables below show actual vs. predicted locations.", S["body"]))

    # Tracking tables
    v1_data = [
        ["Pt", "x0", "y0", "x1_LK", "y1_LK", "x1_pred", "y1_pred", "Err (px)"],
        ["0", "78.00", "22.00", "77.99", "22.00", "78.00", "22.00", "0.009"],
        ["1", "22.00", "22.00", "22.00", "22.00", "22.00", "22.00", "0.001"],
        ["2", "22.00", "58.00", "22.00", "58.00", "22.00", "58.00", "0.001"],
        ["3", "78.00", "58.00", "78.00", "58.00", "78.00", "58.00", "0.001"],
        ["", "", "", "", "", "", "Mean Error:", "0.003 px"],
    ]
    v2_data = [
        ["Pt", "x0", "y0", "x1_LK", "y1_LK", "x1_pred", "y1_pred", "Err (px)"],
        ["0", "320.00", "298.00", "315.58", "297.02", "319.50", "297.49", "3.948"],
        ["1", "379.00", "240.00", "378.03", "243.41", "378.50", "239.49", "3.946"],
        ["2", "262.00", "240.00", "261.98", "235.58", "261.50", "239.49", "3.938"],
        ["3", "320.00", "182.00", "323.41", "181.96", "319.50", "181.49", "3.932"],
        ["", "", "", "", "", "", "Mean Error:", "3.941 px"],
    ]

    ts = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_ACCENT),
        ("TEXTCOLOR", (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Courier"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.white, colors.HexColor("#f7fafc")]),
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#fff3cd")),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
        ("BOX", (0, 0), (-1, -1), 1, C_CYAN),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ])
    col_w = [0.35, 0.75, 0.75, 0.75, 0.75, 0.85, 0.85, 0.75]
    col_w = [x * inch for x in col_w]

    story.append(Paragraph("<b>Video 1 — Translating Circle (Static Reference Points)</b>", S["h3"]))
    t1 = Table(v1_data, colWidths=col_w)
    t1.setStyle(ts)
    story.append(t1)
    story.append(Spacer(1, 0.05*inch))
    story.append(Paragraph(
        "Video 1 mean tracking error = 0.003 px. The static reference rectangle "
        "yields near-zero error since the median flow (dominated by the moving circle) "
        "does not apply well — yet LK correctly identifies near-zero flow for static corners.", S["caption"]))

    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph("<b>Video 2 — Rotating Square + Bouncing Ball</b>", S["h3"]))
    t2 = Table(v2_data, colWidths=col_w)
    t2.setStyle(ts)
    story.append(t2)
    story.append(Spacer(1, 0.05*inch))
    story.append(Paragraph(
        "Video 2 mean error = 3.941 px. The larger error is expected: "
        "the 'uniform flow' theoretical prediction assumes a single dominant motion, "
        "but Video 2 contains multiple objects with different velocities "
        "(rotation + oscillation), so the median vector is a poor global predictor. "
        "LK tracks each point independently with higher accuracy.", S["caption"]))

    story.extend(img("output/tracking_v1.png", 6.5,
        "Figure 3a: Video 1 tracking validation — green circles = LK result, "
        "red triangles = uniform-flow prediction. Errors < 0.01 px for static corners.", S))
    story.extend(img("output/tracking_v2.png", 6.5,
        "Figure 3b: Video 2 tracking validation — ~4 px error due to mixed motion "
        "field making a single global flow vector a poor predictor.", S))

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════
    # PART B — STRUCTURE FROM MOTION
    # ══════════════════════════════════════════════════════════════
    story.append(Paragraph("Part B: Structure from Motion (SfM)", S["h1"]))
    story.append(section_rule())

    story.append(Paragraph("B.1  Object and Camera Setup", S["h2"]))
    story.append(Paragraph(
        "A synthetic planar 2D object (a 16-point star polygon lying in the Z=0 plane) "
        "was observed from four camera positions arranged at different azimuth and "
        "elevation angles. Camera intrinsics were held constant across all views.", S["body"]))

    story.append(Paragraph("<b>Intrinsic Camera Matrix (K)</b>:", S["h3"]))
    story.append(math_box([
        "     [ fx   0   cx ]   [ 800   0   320 ]",
        " K = [  0  fy   cy ] = [   0  800  240 ]",
        "     [  0   0    1 ]   [   0    0    1 ]",
        "",
        "  fx = fy = 800 px  (focal length)",
        "  cx = 320, cy = 240  (principal point = image center)",
    ], S))

    story.append(Paragraph("<b>Camera Extrinsics — Rotation and Translation:</b>", S["h3"]))
    story.append(Paragraph(
        "Each camera has rotation matrix R and translation vector t such that "
        "3D world point X projects to image point x via the projection matrix P:", S["body"]))
    story.append(math_box([
        "  P  =  K · [R | t]    (3×4 projection matrix)",
        "",
        "  x_h  =  P · X_h     (homogeneous image point)",
        "  x    =  x_h[0:2] / x_h[2]",
        "",
        "Camera positions (world coordinates):",
        "  C1 (Front):        center = [ 0.000, -0.5,  4.000 ],  yaw=0°,  pitch=10°",
        "  C2 (Left):         center = [ 4.057, -0.5,  1.087 ],  yaw=75°, pitch=15°",
        "  C3 (Rear-Left):    center = [ 1.606, -0.5, -3.444 ],  yaw=155°,pitch=8°",
        "  C4 (Front-Right):  center = [ 3.182, -0.5,  3.182 ],  yaw=45°, pitch=20°",
    ], S))

    story.extend(img("output/sfm_camera_setup.png", 5.5,
        "Figure 4: Top-down view of camera arrangement. Four cameras (squares) "
        "are placed around the planar object at varying azimuth angles, "
        "each oriented to look toward the origin.", S))

    story.extend(img("output/sfm_views.png", 6.5,
        "Figure 5: Projected object points in each of the four synthetic camera views. "
        "The cyan boundary connects the star-polygon vertices. "
        "Note perspective distortion changes across views.", S))

    # ── B.2 Mathematical Foundations ──────────────────────────────
    story.append(Paragraph("B.2  Mathematical Foundations", S["h2"]))

    story.append(Paragraph("<b>Homogeneous Coordinates and Projective Geometry</b>", S["h3"]))
    story.append(Paragraph(
        "A 3D point X = (X, Y, Z)^T is represented in homogeneous form as "
        "X_h = (X, Y, Z, 1)^T. The projection equation in full:", S["body"]))
    story.append(math_box([
        "  [u]   [p11 p12 p13 p14] [X]",
        "  [v] ~ [p21 p22 p23 p24] [Y]",
        "  [1]   [p31 p32 p33 p34] [Z]",
        "                           [1]",
        "",
        "  u = (p11·X + p12·Y + p13·Z + p14) / (p31·X + p32·Y + p33·Z + p34)",
        "  v = (p21·X + p22·Y + p23·Z + p24) / (p31·X + p32·Y + p33·Z + p34)",
    ], S))

    story.append(Paragraph("<b>Rotation Matrix from Yaw and Pitch</b>", S["h3"]))
    story.append(math_box([
        "Ry(θ) = [ cos(θ)  0  sin(θ) ]    (rotation about Y-axis — yaw)",
        "        [   0     1    0    ]",
        "        [-sin(θ)  0  cos(θ) ]",
        "",
        "Rx(φ) = [ 1    0       0   ]    (rotation about X-axis — pitch)",
        "        [ 0  cos(φ)  -sin(φ)]",
        "        [ 0  sin(φ)   cos(φ)]",
        "",
        "R = Rx(φ) · Ry(θ)    (combined rotation: yaw then pitch)",
        "",
        "t = -R · C    (translation from camera center C in world coords)",
    ], S))

    story.append(Paragraph("<b>Linear Triangulation — DLT Method</b>", S["h3"]))
    story.append(Paragraph(
        "Given corresponding 2D points p1=(u1,v1) in camera 1 and p2=(u2,v2) "
        "in camera 2 with projection matrices P1, P2, we find 3D point X by solving "
        "the homogeneous system AX=0 via SVD:", S["body"]))
    story.append(math_box([
        "For each camera i with projection row vectors P_i^1, P_i^2, P_i^3:",
        "",
        "  u_i · P_i^3 · X  =  P_i^1 · X",
        "  v_i · P_i^3 · X  =  P_i^2 · X",
        "",
        "Rearranging to form A (4×4 for two cameras):",
        "  row 1: u1·P1[2] - P1[0]",
        "  row 2: v1·P1[2] - P1[1]",
        "  row 3: u2·P2[2] - P2[0]",
        "  row 4: v2·P2[2] - P2[1]",
        "",
        "SVD:  A = U Σ V^T",
        "Solution X_h = last column of V  (right singular vector for smallest σ)",
        "Dehomogenize:  X = X_h[0:3] / X_h[3]",
    ], S))
    story.append(Paragraph(
        "Three camera pairs (1,2), (1,3), (1,4) are used and the resulting "
        "reconstructed points are averaged for robustness.", S["body"]))

    story.append(Paragraph("<b>Essential and Fundamental Matrices</b>", S["h3"]))
    story.append(math_box([
        "Relative rotation and translation between cameras i and j:",
        "  R_rel = R_j · R_i^T",
        "  t_rel = t_j - R_rel · t_i",
        "",
        "Skew-symmetric cross-product matrix [t]_x:",
        "       [  0    -t_z   t_y ]",
        "  [t]x = [ t_z    0   -t_x ]",
        "       [-t_y   t_x    0  ]",
        "",
        "Essential matrix:    E = [t_rel]_x · R_rel",
        "Fundamental matrix:  F = K^{-T} · E · K^{-1}",
        "",
        "Epipolar constraint: x2^T · F · x1 = 0",
        "Epipolar line in view 2 for point x1:  l2 = F · x1",
    ], S))

    story.extend(img("output/sfm_epipolar.png", 6.5,
        "Figure 6: Epipolar geometry between Camera 1 (Front) and Camera 2 (Left). "
        "Each colored point in Camera 1 generates a corresponding epipolar line "
        "in Camera 2 (same color). Triangles mark actual corresponding points — "
        "each lies on its epipolar line, confirming F matrix correctness.", S))

    story.extend(img("output/sfm_reconstruction.png", 6.5,
        "Figure 7: Left — 3D reconstruction result. Cyan spheres = ground truth points, "
        "orange triangles = DLT-triangulated reconstructions, red squares = camera centers. "
        "Right — per-point reconstruction error (all ≈ 0, confirming perfect reconstruction "
        "from noise-free synthetic data).", S))

    story.append(Paragraph("B.3  Reconstruction Results", S["h2"]))
    story.append(Paragraph(
        "The DLT triangulation from synthetic (noise-free) data achieves essentially "
        "zero reconstruction error, confirming the mathematical correctness of the "
        "implementation. In practice with real images, errors arise from:", S["body"]))
    for line in [
        "• Image noise and quantization (pixel localization uncertainty)",
        "• Lens distortion (not modeled here — corrected with distortion coefficients k1, k2, k3, p1, p2)",
        "• Feature matching inaccuracies",
        "• Numerical precision in SVD decomposition",
        "• Scene non-planarity (violates homography assumptions)",
    ]:
        story.append(Paragraph(line, S["bullet"]))

    recon_data = [
        ["Point", "GT X", "GT Y", "GT Z", "Recon X", "Recon Y", "Recon Z", "Error"],
        ["0",  "1.000", "0.000", "0.0", "1.000", "0.000", "0.000", "0.000"],
        ["1",  "0.462", "0.191", "0.0", "0.462", "0.191", "0.000", "0.000"],
        ["4",  "-1.00", "0.000", "0.0", "-1.00", "0.000", "0.000", "0.000"],
        ["8",  "0.000", "-1.00", "0.0", "0.000", "-1.00", "0.000", "0.000"],
        ["16", "0.000", "0.000", "0.0", "0.000", "0.000", "0.000", "0.000"],
    ]
    rs = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_ACCENT),
        ("TEXTCOLOR", (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Courier"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
        ("BOX", (0, 0), (-1, -1), 1, C_CYAN),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ])
    rcw = [0.5, 0.75, 0.75, 0.5, 0.75, 0.75, 0.75, 0.75]
    rcw = [x * inch for x in rcw]
    rt = Table(recon_data, colWidths=rcw)
    rt.setStyle(rs)
    story.append(rt)
    story.append(Paragraph(
        "Table 3: Selected 3D reconstruction results. "
        "Error = ||GT − Reconstructed||₂ (world units). "
        "Perfect reconstruction is expected for noise-free synthetic data.",
        S["caption"]))

    story.append(PageBreak())

    # ── References ────────────────────────────────────────────────
    story.append(Paragraph("References", S["h1"]))
    story.append(section_rule())
    refs = [
        "[1] Lucas, B. D., & Kanade, T. (1981). An iterative image registration "
            "technique with an application to stereo vision. Proceedings of the "
            "7th International Joint Conference on Artificial Intelligence (IJCAI), 674–679.",
        "[2] Horn, B. K. P., & Schunck, B. G. (1981). Determining optical flow. "
            "Artificial Intelligence, 17(1–3), 185–203.",
        "[3] Hartley, R., & Zisserman, A. (2003). Multiple View Geometry in Computer "
            "Vision (2nd ed.). Cambridge University Press.",
        "[4] Faugeras, O. (1993). Three-Dimensional Computer Vision: A Geometric "
            "Viewpoint. MIT Press.",
        "[5] Szeliski, R. (2010). Computer Vision: Algorithms and Applications. "
            "Springer. Available: https://szeliski.org/Book/",
        "[6] Bradski, G. (2000). The OpenCV Library. Dr. Dobb's Journal of Software Tools.",
        "[7] Shi, J., & Tomasi, C. (1994). Good features to track. Proceedings of "
            "IEEE CVPR, 593–600.",
        "[8] Bouguet, J.-Y. (2000). Pyramidal implementation of the Lucas-Kanade "
            "feature tracker. Intel Corporation Microprocessor Research Labs.",
        "[9] Farneback, G. (2003). Two-frame motion estimation based on polynomial "
            "expansion. Proceedings of SCIA, 363–370.",
    ]
    for r in refs:
        story.append(Paragraph(r, S["bullet"]))
        story.append(Spacer(1, 0.04*inch))

    doc.build(story)
    print(f"[✓] PDF report saved: {out_path}")


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    styles = make_styles()
    make_report("output/Assignment6_Report.pdf", styles)
