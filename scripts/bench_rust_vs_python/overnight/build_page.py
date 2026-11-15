"""Fill page_template.html with the pooled load and sequential summaries and the merged flame tree.

usage: build_page.py <load_pool.json> <seq_pool.json> <flame.json> <template.html> <out.html>
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import flame as fl

load_path, seq_path, flame_path, template_path, out_path = (Path(a) for a in sys.argv[1:6])
merged = json.loads(flame_path.read_text())
names = [n for n, _ in fl.BUCKETS] + [fl.OTHER]
by_id = {r["id"]: r for r in merged["tree"]}
KEEP = ("cpu_ms_per_request", "overhead_p50_ms", "overhead_p95_ms", "overhead_p99_ms", "ttft_overhead_p50_ms",
        "header_overhead_p50_ms", "pre_upstream_p50_ms", "tail_p50_ms", "peak_rss_mb")


def path(row):
    frames = []
    while row is not None:
        frames.append(row["name"])
        row = by_id.get(row["parent"])
    return frames[::-1]


def short(frame):
    match = re.match(r"^(.*?) \((.*)\)$", frame)
    name, where = (match.group(1), match.group(2)) if match else (frame, "")
    previous = None
    while previous != name:
        previous, name = name, re.sub(r"<[^<>]*>", "", name)
    name = re.sub(r"\{[a-z_]+#\d+\}", "{closure}", name)
    return name + (" · " + where.split("/")[-1] if where else "")


def runs(pool):
    return [{"run": r["run"], "round": r["round"], **{k: round(r[k], 3) for k in KEEP}} for r in pool["runs"]]


load, seq = json.loads(load_path.read_text()), json.loads(seq_path.read_text())
data = {
    "tree": [[r["id"], r["parent"], short(r["name"]), r["ms"], r["self"], names.index(fl.bucket_of(path(r))), r["name"]]
             for r in merged["tree"]],
    "buckets": [{"name": b["bucket"], "ms": round(b["ms"], 2), "sd": round(b["sd"], 2)} for b in merged["buckets"]],
    "bucketNames": names,
    "load": {"runs": runs(load), "summary": load["summary"]},
    "seq": {"runs": runs(seq), "summary": seq["summary"]},
    "cpu_ms": merged["cpu_ms"],
}
out_path.write_text(template_path.read_text().replace("__DATA__", json.dumps(data, separators=(",", ":"))))
print(f"wrote {out_path} ({out_path.stat().st_size // 1024} KB)")
