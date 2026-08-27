"""
check_determinism.py — does the same input folder produce the same picks twice?

  python bench/check_determinism.py ./bench_data/corpus_1920 [runs]

Regression guard for the as_completed()/dict-ordering race in group_by_similarity().
Exits non-zero if bucket assignment or BEST picks drift between identical runs.
"""
import sys, json, shutil, io, contextlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import bird_cv as B

corpus = Path(sys.argv[1])
runs_n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
out_base = corpus.parent / "_det"

reports = []
for i in range(runs_n):
    out = out_base.with_name(f"_det_{i}")
    shutil.rmtree(out, ignore_errors=True)
    with contextlib.redirect_stdout(io.StringIO()):
        B.run(str(corpus), str(out), 2)
    reports.append({d["filename"]: d for d in
                    json.loads((out / "bird_quality_report.json").read_text())})
    shutil.rmtree(out, ignore_errors=True)

names = sorted(reports[0])
ref = reports[0]
print(f"\n{len(names)} images, {runs_n} identical runs\n")
print(f"{'run':>4}{'scores':>12}{'buckets':>10}{'picks':>16}")
print("-" * 42)
drift = False
for i, r in enumerate(reports):
    same_scores = all(abs(r[n]["quality_overall"] - ref[n]["quality_overall"]) < 1e-9 for n in names)
    same_buckets = all(r[n]["bucket_id"] == ref[n]["bucket_id"] for n in names)
    picks = {n for n in names if r[n]["selected"]}
    ref_picks = {n for n in names if ref[n]["selected"]}
    if not (same_buckets and picks == ref_picks):
        drift = True
    print(f"{i:>4}{('same' if same_scores else 'DIFFER'):>12}"
          f"{('same' if same_buckets else 'DIFFER'):>10}"
          f"{('same' if picks == ref_picks else f'{len(picks & ref_picks)}/{len(ref_picks)} kept'):>16}")
print("-" * 42)
print(f"deterministic: {not drift}\n")
sys.exit(1 if drift else 0)
