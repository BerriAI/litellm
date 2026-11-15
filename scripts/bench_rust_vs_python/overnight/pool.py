"""Pool every run in one results dir into per-run metrics and a median, bootstrap 95% CI, CV and drift per metric.

usage: pool.py <results dir, e.g. ~/bench/results/n15-load> <out.json>
"""

import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import report as rp

root = Path(sys.argv[1])
out = Path(sys.argv[2])
dirs = sorted((d for d in root.iterdir() if d.is_dir() and (d / "meta.json").exists()), key=lambda d: int(d.name.split("-r")[1]))
runs = []
for d in dirs:
    m = rp.run_metrics(d)
    meta = json.loads((d / "meta.json").read_text())
    def p(vals, q, scale=1000.0):
        return rp.percentile([v * scale for v in vals], q) if vals else float("nan")
    runs.append({
        "run": d.name,
        "round": int(d.name.split("-r")[1]),
        "started": meta.get("started_at") or (d / "meta.json").stat().st_mtime,
        "measured": m.measured,
        "errors": m.errors,
        "rust_share": m.rust_share,
        "prometheus_ok": m.prometheus_matches,
        "cpu_ms_per_request": m.cpu_ms_per_request,
        "overhead_p50_ms": p(m.overheads_s, 50),
        "overhead_p90_ms": p(m.overheads_s, 90),
        "overhead_p95_ms": p(m.overheads_s, 95),
        "overhead_p99_ms": p(m.overheads_s, 99),
        "ttft_overhead_p50_ms": p(m.ttft_overheads_s, 50),
        "ttft_overhead_p99_ms": p(m.ttft_overheads_s, 99),
        "header_overhead_p50_ms": rp.percentile(list(m.header_overheads_ms), 50) if m.header_overheads_ms else float("nan"),
        "pre_upstream_p50_ms": p(m.pre_upstream_s, 50),
        "stream_start_p50_ms": p(m.stream_start_s, 50),
        "tail_p50_ms": p(m.tail_s, 50),
        "tail_p99_ms": p(m.tail_s, 99),
        "peak_rss_mb": m.peak_rss_mb,
        "rss_growth_mb": m.rss_growth_mb,
        "loadgen_cpu_share": m.loadgen_cpu_share,
    })

rng = random.Random(7)
keys = [k for k in runs[0] if k.endswith(("_ms", "_mb", "_request"))]
summary = {}
for k in keys:
    vals = [r[k] for r in runs]
    meds = sorted(statistics.median(rng.choices(vals, k=len(vals))) for _ in range(4000))
    n = len(vals)
    xs = [r["round"] for r in runs]
    mx, my = statistics.mean(xs), statistics.mean(vals)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, vals)) / sum((x - mx) ** 2 for x in xs)
    first, last = statistics.median(vals[: n // 2]), statistics.median(vals[n // 2 :])
    summary[k] = {
        "median": statistics.median(vals),
        "ci95": [meds[100], meds[3899]],
        "run_sd": statistics.stdev(vals),
        "cv_pct": 100 * statistics.stdev(vals) / my,
        "min": min(vals),
        "max": max(vals),
        "drift_per_round": slope,
        "first_half_median": first,
        "second_half_median": last,
    }
out.write_text(json.dumps({"runs": runs, "summary": summary}, indent=1))
print(f"{'metric':26s} {'median':>9s} {'95% CI':>19s} {'cv%':>6s} {'min':>8s} {'max':>8s} {'1st half':>9s} {'2nd half':>9s}")
for k, s in summary.items():
    print(f"{k:26s} {s['median']:9.2f} [{s['ci95'][0]:8.2f},{s['ci95'][1]:8.2f}] {s['cv_pct']:6.1f} {s['min']:8.2f} {s['max']:8.2f} {s['first_half_median']:9.2f} {s['second_half_median']:9.2f}")
print("runs", len(runs), "errors", sum(r["errors"] for r in runs), "min rust share", min(r["rust_share"] for r in runs),
      "prometheus ok", all(r["prometheus_ok"] for r in runs), "max loadgen cpu", max(r["loadgen_cpu_share"] for r in runs))
