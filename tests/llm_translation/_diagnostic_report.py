"""Post-run analyzer for tests/llm_translation/ diagnostic JSONL files.

Usage::
    python tests/llm_translation/_diagnostic_report.py test-results/

Prints, per worker:
  - first/last record (to see growth delta)
  - tests where rss_mb jumped >100 MB or azure_closed flipped to True
  - top 10 cache_n growth transitions
  - any test where fds jumped >10 between setup/teardown

Intended for reading CI logs after a diagnostic run. Not imported by pytest.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def _load_records(diag_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Group records by worker id."""
    by_worker: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for path in sorted(diag_dir.glob("diagnostic-*.jsonl")):
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            by_worker[r.get("worker", path.stem)].append(r)
    return by_worker


def _fmt_row(r: Dict[str, Any]) -> str:
    return (
        f"  idx={r.get('idx'):>4} phase={r.get('phase'):<8} "
        f"rss={r.get('rss_mb'):>7} MiB  fds={r.get('fds'):>4}  "
        f"threads={r.get('threads'):>3}  tasks={r.get('tasks'):>3}  "
        f"cache={r.get('cache_n'):>3}  azure_closed={r.get('azure_closed')}  "
        f"gc={r.get('gc_objs')}  test={r.get('test')}"
    )


def _report_worker(worker: str, records: List[Dict[str, Any]]) -> None:
    print(f"\n=========================== worker {worker} ===========================")
    if not records:
        print("  (no records)")
        return

    first, last = records[0], records[-1]
    print(f"  first: {_fmt_row(first)}")
    print(f"  last:  {_fmt_row(last)}")
    print(
        f"  delta over run: rss +{last['rss_mb'] - first['rss_mb']:.1f} MiB, "
        f"fds +{last['fds'] - first['fds']}, "
        f"cache +{last['cache_n'] - first['cache_n']}, "
        f"gc +{last['gc_objs'] - first['gc_objs']}"
    )

    # Azure client closed transitions — now with *which* key flipped.
    # A "flip" is any new key that appears in record.closed_keys vs. the
    # previous record's closed_keys.
    prev_closed: set[str] = set()
    flips: List[tuple[Dict[str, Any], set[str]]] = []
    for r in records:
        cur_closed = set(r.get("closed_keys") or [])
        new_flips = cur_closed - prev_closed
        if new_flips:
            flips.append((r, new_flips))
        prev_closed = cur_closed
    if flips:
        print(f"\n  Client-closed transitions detected: {len(flips)}. Showing first 10:")
        for r, new in flips[:10]:
            print(f"    {_fmt_row(r)}")
            for k in sorted(new):
                print(f"      + closed key: {k!r}")

    # Large RSS jumps between adjacent records.
    jumps = []
    for a, b in zip(records, records[1:]):
        try:
            dr = (b["rss_mb"] or 0) - (a["rss_mb"] or 0)
        except Exception:
            continue
        if dr >= 50:
            jumps.append((dr, a, b))
    jumps.sort(key=lambda x: x[0], reverse=True)
    if jumps:
        print(f"\n  Top RSS jumps (>=50 MiB) — {len(jumps)} total, showing top 5:")
        for dr, a, b in jumps[:5]:
            print(f"    +{dr:.1f} MiB between idx {a.get('idx')} and {b.get('idx')}")
            print(f"      from: {_fmt_row(a)}")
            print(f"      to:   {_fmt_row(b)}")

    # Cache growth within a single test (setup→teardown).
    pairs: Dict[int, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for r in records:
        pairs[r["idx"]][r["phase"]] = r
    growth: List[tuple[int, Dict[str, Any], Dict[str, Any]]] = []
    for idx, phases in pairs.items():
        if "setup" in phases and "teardown" in phases:
            d = phases["teardown"]["cache_n"] - phases["setup"]["cache_n"]
            if d > 0:
                growth.append((d, phases["setup"], phases["teardown"]))
    growth.sort(key=lambda x: x[0], reverse=True)
    if growth:
        print(f"\n  Tests that grew cache_n — showing top 10:")
        for d, s, t in growth[:10]:
            print(f"    +{d}  {s['test']}")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: _diagnostic_report.py <dir-containing-diagnostic-*.jsonl>")
        return 2
    diag_dir = Path(sys.argv[1])
    by_worker = _load_records(diag_dir)
    if not by_worker:
        print(f"no diagnostic-*.jsonl files found in {diag_dir}")
        return 0
    for worker in sorted(by_worker):
        _report_worker(worker, by_worker[worker])
    return 0


if __name__ == "__main__":
    sys.exit(main())
