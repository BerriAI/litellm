"""Merge several profiled runs into one flame tree, scaled to the unprofiled CPU per request.

Each run gets equal weight. Output: merged.folded (microseconds per request), flame.json (tree, subsystems with
spread across runs, top functions).

usage: merge_flame.py <profiled results dir> <cpu ms per request> <out dir> <run dirs, e.g. N-r2,N-r3>
"""

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import flame as fl

prof_root = Path(sys.argv[1])
cpu_ms = float(sys.argv[2])
out = Path(sys.argv[3])
names = sys.argv[4].split(",")
out.mkdir(parents=True, exist_ok=True)

profiles = {n: fl.load_profile(prof_root / n) for n in names}
fractions = {}
for n, p in profiles.items():
    total = p.cpu_samples
    fractions[n] = {k: c / total for k, c in p.cpu_stacks.items()}
stacks = set().union(*fractions.values())
merged = {s: cpu_ms * statistics.mean(f.get(s, 0.0) for f in fractions.values()) for s in stacks}

(out / "merged.folded").write_text(
    "".join(f"{';'.join(s)} {round(v * 1000)}\n" for s, v in sorted(merged.items()) if round(v * 1000) > 0)
)

bucket_runs = {}
for n, f in fractions.items():
    b = Counter()
    for s, v in f.items():
        b[fl.bucket_of(s)] += v * cpu_ms
    bucket_runs[n] = b
bnames = [name for name, _ in fl.BUCKETS] + [fl.OTHER]
buckets = sorted(
    (
        {
            "bucket": b,
            "ms": statistics.mean(r[b] for r in bucket_runs.values()),
            "sd": statistics.stdev(r[b] for r in bucket_runs.values()),
        }
        for b in bnames
    ),
    key=lambda r: -r["ms"],
)

own, incl = Counter(), Counter()
for s, v in merged.items():
    own[s[-1]] += v
    for name in set(s):
        incl[name] += v

inclusive = Counter()
for s, v in merged.items():
    for d in range(1, len(s) + 1):
        inclusive[s[:d]] += v
kept = sorted((p for p, v in inclusive.items() if v >= cpu_ms * 0.002), key=lambda p: (len(p), p))
ids = {p: i for i, p in enumerate(kept)}
tree = [
    {"id": ids[p], "parent": ids.get(p[:-1], -1), "name": p[-1], "ms": round(inclusive[p], 3),
     "self": round(merged.get(p, 0.0), 3)}
    for p in kept
    if len(p) == 1 or p[:-1] in ids
]
kinds = {n: dict(p.kinds) for n, p in profiles.items()}
(out / "flame.json").write_text(json.dumps({
    "cpu_ms": cpu_ms,
    "runs": names,
    "buckets": buckets,
    "top_self": [{"name": k, "ms": round(v, 3)} for k, v in own.most_common(40)],
    "top_inclusive": [{"name": k, "ms": round(v, 3)} for k, v in incl.most_common(60)],
    "tree": tree,
    "kinds": kinds,
    "cpu_samples": {n: p.cpu_samples for n, p in profiles.items()},
    "warmup_cpu_ms": {n: round(fl.unprofiled_cpu_ms(prof_root / n), 2) for n in names},
}))
print("cpu samples per run", {n: p.cpu_samples for n, p in profiles.items()})
print("warmup cpu ms", {n: round(fl.unprofiled_cpu_ms(prof_root / n), 1) for n in names})
for b in buckets:
    print(f"{b['bucket']:34s} {b['ms']:7.2f} ms  sd {b['sd']:.2f}")
print("--- top self")
for k, v in own.most_common(25):
    print(f"{v:7.2f}  {k[:150]}")
