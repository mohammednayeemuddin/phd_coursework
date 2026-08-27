"""
Generate a synthetic bird-photo corpus with realistic structure:
 - several distinct SCENES (sky/water/grass palettes)
 - each scene shot as a BURST of near-identical frames (small jitter)
 - per-frame focus blur varies so quality ranking has real spread
 - bird blobs use the exact HSV signatures bird_cv.py hunts for
   (orange feet, white chest, brown flanks, dark bodies)
Deterministic: fixed seed -> byte-identical corpus every run.
"""
import cv2, numpy as np, sys
from pathlib import Path

rng = np.random.default_rng(20260826)

# family palettes in BGR
FAMILIES = {
    "duck":   dict(body=(70,110,150),  chest=(235,238,240), feet=(30,120,225), dark=False, tall=0.75),
    "wader":  dict(body=(180,195,205), chest=(225,230,235), feet=(40,130,220), dark=False, tall=2.1),
    "darkwb": dict(body=(45,42,40),    chest=(70,68,66),    feet=(40,110,180), dark=True,  tall=1.6),
    "raptor": dict(body=(60,75,95),    chest=(150,160,170), feet=(35,115,200), dark=False, tall=0.5),
}
SCENE_BG = {
    "duck":   ((190,150,110), (95,120,105)),   # water blue-grey -> weedy
    "wader":  ((205,190,165), (110,135,120)),  # pale marsh
    "darkwb": ((150,140,130), (70,85,75)),     # dim shallows
    "raptor": ((235,190,140), (200,175,150)),  # open sky
}

def gradient_bg(h, w, top, bot):
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    img = (np.array(top, np.float32) * (1 - t) + np.array(bot, np.float32) * t)
    return np.repeat(img, w, axis=1)

def add_texture(img, strength, seed):
    r = np.random.default_rng(seed)
    h, w = img.shape[:2]
    # low-freq mottling (water ripples / foliage) + fine grain
    low = r.normal(0, strength, (h // 24, w // 24, 3)).astype(np.float32)
    low = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC)
    fine = r.normal(0, strength * 0.35, (h, w, 3)).astype(np.float32)
    return img + low + fine

def draw_bird(img, cx, cy, scale, fam, seed):
    r = np.random.default_rng(seed)
    p = FAMILIES[fam]
    tall = p["tall"]
    bw = int(scale)
    bh = int(scale * tall)
    # body ellipse
    cv2.ellipse(img, (cx, cy), (bw, bh), r.integers(-12, 12), 0, 360, p["body"], -1)
    # chest patch (white/pale -> white_ratio feature)
    cv2.ellipse(img, (cx - bw // 4, cy + bh // 5), (bw // 2, bh // 2), 0, 0, 360, p["chest"], -1)
    # brown flank streaks -> brown_ratio
    for i in range(6):
        y = cy - bh // 2 + i * max(1, bh // 6)
        cv2.line(img, (cx - bw, y), (cx + bw, y + int(r.integers(-4, 4))),
                 (60, 95, 140), max(1, scale // 30))
    # head
    hx, hy = cx + int(bw * 0.7), cy - int(bh * 0.85)
    cv2.circle(img, (hx, hy), max(3, bw // 3), p["body"], -1)
    # bill + feet (orange -> orange_ratio anchor)
    cv2.ellipse(img, (hx + bw // 3, hy), (bw // 3, bw // 8), 0, 0, 360, p["feet"], -1)
    for fx in (-bw // 3, bw // 3):
        cv2.ellipse(img, (cx + fx, cy + bh), (bw // 4, bw // 8), 0, 0, 360, p["feet"], -1)
    # feather edge detail so Canny/contours have real work
    for i in range(28):
        a = r.uniform(0, 2 * np.pi); rr = r.uniform(0.3, 1.0)
        x = int(cx + np.cos(a) * bw * rr); y = int(cy + np.sin(a) * bh * rr)
        cv2.line(img, (x, y), (x + int(r.integers(-9, 9)), y + int(r.integers(-9, 9))),
                 tuple(int(c * r.uniform(0.75, 1.25)) for c in p["body"]), 1)
    return img

def make_scene(fam, scene_id, n_frames, W, H):
    top, bot = SCENE_BG[fam]
    # per-scene palette + framing shift: different locations on different days,
    # not the same pond twice. Keeps distinct scenes out of one another's buckets.
    sr = np.random.default_rng(7000 + scene_id)
    shift = sr.uniform(-55, 55, 3)
    warm  = sr.uniform(0.80, 1.20)
    top = tuple(float(np.clip(c * warm + shift[i], 25, 250)) for i, c in enumerate(top))
    bot = tuple(float(np.clip(c / warm + shift[i] * 0.6, 20, 235)) for i, c in enumerate(bot))
    frames = []
    sr_h = sr.uniform(0.18, 0.52)          # horizon height varies by scene
    base_cx, base_cy = int(W * sr.uniform(0.25, 0.70)), int(H * sr.uniform(0.38, 0.68))
    scale = int(min(W, H) * sr.uniform(0.08, 0.20))
    for k in range(n_frames):
        img = gradient_bg(H, W, top, bot)
        img = add_texture(img, 9.0, scene_id * 1000 + k)
        # horizon / bank line
        hz = float(sr_h)
        cv2.line(img, (0, int(H * hz)), (W, int(H * (hz + 0.03))),
                 tuple(c * 0.82 for c in bot), max(2, H // 220))
        # burst jitter: small camera shift, not a new scene
        cx = base_cx + int(rng.integers(-W // 60, W // 60))
        cy = base_cy + int(rng.integers(-H // 60, H // 60))
        img = draw_bird(img, cx, cy, scale, fam, scene_id * 100 + k)
        # focus quality varies across the burst -> real ranking spread
        blur = [0, 0, 3, 5, 9, 13][k % 6]
        if blur:
            img = cv2.GaussianBlur(img, (blur * 2 + 1,) * 2, 0)
        # exposure drift
        img = img * rng.uniform(0.88, 1.12) + rng.uniform(-12, 12)
        frames.append(np.clip(img, 0, 255).astype(np.uint8))
    return frames

def build(outdir: Path, n_images: int, long_side: int, quality: int):
    outdir.mkdir(parents=True, exist_ok=True)
    for f in outdir.glob("*.JPG"):
        f.unlink()
    W, H = long_side, int(long_side * 2 / 3)
    fams = list(FAMILIES)
    made, scene_id = 0, 0
    while made < n_images:
        fam = fams[scene_id % len(fams)]
        n = min(int(rng.integers(4, 10)), n_images - made)
        for k, img in enumerate(make_scene(fam, scene_id, n, W, H)):
            cv2.imwrite(str(outdir / f"DSC{5000 + made:05d}.JPG"), img,
                        [cv2.IMWRITE_JPEG_QUALITY, quality])
            made += 1
        scene_id += 1
    mb = sum(f.stat().st_size for f in outdir.glob("*.JPG")) / 1e6
    print(f"{outdir.name}: {made} images @ {W}x{H}  ({mb:.1f} MB total, {mb/made:.2f} MB avg)")

if __name__ == "__main__":
    base = Path(sys.argv[1])
    build(base / "corpus_1920", 77, 1920, 92)
