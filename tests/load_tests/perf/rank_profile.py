"""Rank functions from a yappi ystat profile by total / self CPU time.

Prints two tables:
  1. TOP by TOTAL time  (ttot — function + everything it calls)
  2. TOP by SELF time   (tsub — function's own code, callees excluded)

Usage:
    rank_profile.py <profile.ystat> [--top N] [--filter SUBSTR]
"""
import os
import sys

import yappi

USAGE = "usage: rank_profile.py <profile.ystat> [--top N] [--filter SUBSTR]"


def shorten(path: str) -> str:
    if not path:
        return ""
    # Collapse absolute site-packages / worktree prefixes for readability.
    parts = path.split(os.sep)
    for marker in ("site-packages", "litellm-perfTesting"):
        if marker in parts:
            i = parts.index(marker)
            return os.sep.join(parts[i + 1 :])
    return parts[-1]


def main() -> None:
    argv = sys.argv[1:]
    if not argv:
        print(USAGE, file=sys.stderr)
        sys.exit(1)
    path = argv[0]
    top_n = 30
    flt = None
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == "--top":
            top_n = int(argv[i + 1])
            i += 2
        elif a == "--filter":
            flt = argv[i + 1].lower()
            i += 2
        else:
            print(f"unknown arg {a}\n{USAGE}", file=sys.stderr)
            sys.exit(1)

    stats = yappi.YFuncStats().add(path)
    entries = list(stats)
    if not entries:
        print(f"# {path}: no entries")
        return

    total_ttot = sum(e.ttot for e in entries)

    def label(e: yappi.YFuncStat) -> str:
        fn = e.name
        fl = shorten(e.module or "")
        ln = e.lineno
        if fl and ln is not None and ln > 0:
            return f"{fn}  ({fl}:{ln})"
        if fl:
            return f"{fn}  ({fl})"
        return fn

    def passes(e: yappi.YFuncStat) -> bool:
        return flt is None or flt in label(e).lower()

    def fmt_table(title: str, key: str, primary: str) -> None:
        rows = [e for e in entries if passes(e)]
        rows.sort(key=lambda e: getattr(e, key), reverse=True)
        print(f"=== {title} ===")
        if primary == "ttot":
            print(
                f"{'TOTAL':>10} {'TOTAL%':>7}  {'SELF':>10}  {'NCALL':>8}  FUNCTION"
            )
        else:
            print(
                f"{'SELF':>10} {'SELF%':>6}  {'TOTAL':>10}  {'NCALL':>8}  FUNCTION"
            )
        print("-" * 100)
        for e in rows[:top_n]:
            tp = (e.ttot / total_ttot * 100) if total_ttot else 0.0
            sp = (e.tsub / total_ttot * 100) if total_ttot else 0.0
            if primary == "ttot":
                print(
                    f"{e.ttot:10.4f} {tp:6.1f}%  {e.tsub:10.4f}  {e.ncall:8d}  {label(e)}"
                )
            else:
                print(
                    f"{e.tsub:10.4f} {sp:5.1f}%  {e.ttot:10.4f}  {e.ncall:8d}  {label(e)}"
                )
        print()

    print(f"# profile:        {path}")
    print(f"# clock type:     CPU (async-suspended time excluded)")
    print(f"# grand total:    {total_ttot:.4f}s  (sum of ttot over all functions)")
    print(f"# entries:        {len(entries)}")
    if flt:
        print(f"# filter:         '{flt}' (case-insensitive substring)")
    print()

    fmt_table("TOP BY TOTAL TIME (function + descendants)", "ttot", "ttot")
    fmt_table("TOP BY SELF TIME (function's own code only)", "tsub", "tsub")


if __name__ == "__main__":
    main()
