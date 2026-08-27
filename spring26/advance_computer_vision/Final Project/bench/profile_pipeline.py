"""
profile_pipeline.py — where does bird_cv.py actually spend its time?

  python bench/gen_corpus.py ./bench_data          # build a corpus first
  python bench/profile_pipeline.py ./bench_data/corpus_1920

Reports three things:
  1. serial cost per operation (OpenCV pinned to 1 thread — the honest CPU cost)
  2. how many times each scorer is invoked per image
  3. end-to-end wall time across worker counts
"""
import sys, time, io, contextlib, shutil, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2
import bird_cv as B


def invocation_counts(sample: Path):
    counts = {}
    originals = {}
    for name in ("score_sharpness", "score_bg_separation", "score_subject_fill",
                 "score_exposure", "score_color_richness", "score_composition"):
        originals[name] = getattr(B, name)

        def make(orig, key):
            def wrapper(*a, **k):
                counts[key] = counts.get(key, 0) + 1
                return orig(*a, **k)
            return wrapper
        setattr(B, name, make(originals[name], name))

    img = cv2.imread(str(sample))
    img = cv2.resize(img, (1920, 1280))
    B.compute_quality(img, cv2.cvtColor(img, cv2.COLOR_BGR2HSV),
                      cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    for name, orig in originals.items():
        setattr(B, name, orig)
    return counts


def serial_ops(images):
    acc = {}

    def t(key, fn, *a, **k):
        t0 = time.perf_counter()
        r = fn(*a, **k)
        acc[key] = acc.get(key, 0.0) + (time.perf_counter() - t0)
        return r

    for p in images:
        im = t("imread", B.imread_for_analysis, p)
        t("fingerprint", B.image_fingerprint, im)
        t("family_feats", B.extract_family_features, im)
        h0, w0 = im.shape[:2]
        s = B.ANALYSIS_LONG_SIDE / max(h0, w0)
        ir = t("resize", lambda: cv2.resize(im, (int(w0 * s), int(h0 * s)),
                                            interpolation=cv2.INTER_AREA) if s < 1 else im.copy())
        ir = t("bilateralFilter", cv2.bilateralFilter, ir, B.DENOISE_D, 50, 50)
        hsv = t("cvt_hsv", cv2.cvtColor, ir, cv2.COLOR_BGR2HSV)
        gray = t("cvt_gray", cv2.cvtColor, ir, cv2.COLOR_BGR2GRAY)
        t("score_sharpness", B.score_sharpness, gray)
        t("score_bg_sep", B.score_bg_separation, gray)
        t("score_fill", B.score_subject_fill, gray)
        t("score_exposure", B.score_exposure, gray)
        t("score_color", B.score_color_richness, hsv)
        t("score_composition", B.score_composition, gray)
    return {k: v / len(images) * 1000 for k, v in acc.items()}


def wall_sweep(corpus, out, workers):
    times = {}
    for w in workers:
        best = float("inf")
        for _ in range(3):
            shutil.rmtree(out, ignore_errors=True)
            t0 = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                B.run(str(corpus), str(out), 2, n_workers=w)
            best = min(best, time.perf_counter() - t0)
        times[w] = best
    shutil.rmtree(out, ignore_errors=True)
    return times


def main():
    corpus = Path(sys.argv[1])
    images = sorted(p for p in corpus.iterdir() if p.suffix.lower() in B.SUPPORTED)
    sample_n = min(20, len(images))

    print(f"\n{'=' * 70}\n  PROFILE — {corpus.name}, {len(images)} images")
    print(f"  cv2 {cv2.__version__} | {os.cpu_count()} logical cores\n{'=' * 70}")

    counts = invocation_counts(images[0])
    print("\n-- scorer invocations per compute_quality() call --")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"   {k:<24} {v}x{'   <-- computed twice' if v > 1 else ''}")

    cv2.setNumThreads(1)
    ops = serial_ops(images[:sample_n])
    total = sum(ops.values())
    print(f"\n-- serial cost per image (cv2 threads=1, {sample_n} images) --")
    for k, v in sorted(ops.items(), key=lambda kv: -kv[1]):
        print(f"   {k:<20} {v:8.2f} ms  {v / total * 100:5.1f}%")
    print(f"   {'TOTAL':<20} {total:8.2f} ms")

    cv2.setNumThreads(0)
    print("\n-- end-to-end wall time by worker count --")
    for w, t in wall_sweep(corpus, corpus.parent / "_profile_out", (8, 16, 24, 32)).items():
        print(f"   {w:>3} workers   {t:6.2f}s   {t / len(images) * 1000:6.1f} ms/img")
    print()


if __name__ == "__main__":
    main()
