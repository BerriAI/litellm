"""Fill compare_template.html with two variants' pooled runs, paired per-round deltas and both merged flame trees.

usage: compare_page.py <base name> <new name> <base load_pool> <new load_pool> <base seq_pool> <new seq_pool>
                       <base flame.json> <new flame.json> <text.json> <template.html> <out.html> [<deltas out.json>]

text.json holds the experiment's own words: page_title, title, labels (two display names), and lede, meta, findings
and method as HTML fragments.
"""

import json
import random
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import flame as fl

KEEP = ("cpu_ms_per_request", "overhead_p50_ms", "overhead_p95_ms", "overhead_p99_ms", "ttft_overhead_p50_ms",
        "ttft_overhead_p99_ms", "header_overhead_p50_ms", "pre_upstream_p50_ms", "tail_p50_ms", "peak_rss_mb")
BUCKET_NAMES = [n for n, _ in fl.BUCKETS] + [fl.OTHER]


def short(frame):
    match = re.match(r"^(.*?) \((.*)\)$", frame)
    name, where = (match.group(1), match.group(2)) if match else (frame, "")
    previous = None
    while previous != name:
        previous, name = name, re.sub(r"<[^<>]*>", "", name)
    name = re.sub(r"\{[a-z_]+#\d+\}", "{closure}", name)
    return name + (" · " + where.split("/")[-1] if where else "")


def flame_data(merged):
    by_id = {r["id"]: r for r in merged["tree"]}

    def path(row):
        frames = []
        while row is not None:
            frames.append(row["name"])
            row = by_id.get(row["parent"])
        return frames[::-1]

    return {
        "tree": [[r["id"], r["parent"], short(r["name"]), r["ms"], r["self"], BUCKET_NAMES.index(fl.bucket_of(path(r))),
                  r["name"]] for r in merged["tree"]],
        "buckets": {b["bucket"]: {"ms": round(b["ms"], 2), "sd": round(b["sd"], 2)} for b in merged["buckets"]},
        "cpu_ms": merged["cpu_ms"],
        "runs": merged["runs"],
    }


def runs(pool):
    return [{"run": r["run"], "round": r["round"], **{k: round(r[k], 3) for k in KEEP}} for r in pool["runs"]]


def paired(base, new):
    rng = random.Random(11)
    by_round = {r["round"]: r for r in base["runs"]}
    pairs = [(by_round[r["round"]], r) for r in new["runs"] if r["round"] in by_round]
    out = {}
    for k in KEEP:
        diffs = [n[k] - b[k] for b, n in pairs]
        boots = sorted(statistics.median(rng.choices(diffs, k=len(diffs))) for _ in range(4000))
        base_median = base["summary"][k]["median"]
        out[k] = {
            "base": base_median,
            "new": new["summary"][k]["median"],
            "base_ci": base["summary"][k]["ci95"],
            "new_ci": new["summary"][k]["ci95"],
            "diff": statistics.median(diffs),
            "diff_ci": [boots[100], boots[3899]],
            "pct": 100 * statistics.median(diffs) / base_median,
            "new_lower_rounds": sum(d < 0 for d in diffs),
            "rounds": len(diffs),
        }
    return out


def main():
    base_name, new_name = sys.argv[1:3]
    lb, ln, sb, sn, fb, fn = (json.loads(Path(a).read_text()) for a in sys.argv[3:9])
    text = json.loads(Path(sys.argv[9]).read_text())
    template, out = Path(sys.argv[10]), Path(sys.argv[11])
    deltas = {"load": paired(lb, ln), "seq": paired(sb, sn)}
    data = {
        "names": [base_name, new_name],
        "bucketNames": BUCKET_NAMES,
        "load": {"runs": [runs(lb), runs(ln)], "deltas": deltas["load"]},
        "seq": {"runs": [runs(sb), runs(sn)], "deltas": deltas["seq"]},
        "flames": [flame_data(fb), flame_data(fn)],
    }
    slots = {
        "__PAGE_TITLE__": text["page_title"], "__TITLE__": text["title"], "__LEDE__": text["lede"],
        "__META__": text["meta"], "__FINDINGS__": text["findings"], "__METHOD__": text["method"],
        "__LABELS__": json.dumps(text["labels"]), "__DATA__": json.dumps(data, separators=(",", ":")),
    }
    out.write_text(re.sub("|".join(slots), lambda m: slots[m.group(0)], template.read_text()))
    if len(sys.argv) > 12:
        Path(sys.argv[12]).write_text(json.dumps(deltas, indent=1))
    for kind, d in deltas.items():
        print(f"== {kind}: {base_name} -> {new_name}")
        for k, v in d.items():
            print(f"{k:24s} {v['base']:9.2f} {v['new']:9.2f} {v['diff']:+9.2f} [{v['diff_ci'][0]:+8.2f},"
                  f"{v['diff_ci'][1]:+8.2f}] {v['pct']:+6.1f}%  new lower in {v['new_lower_rounds']}/{v['rounds']}")
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")


main()
