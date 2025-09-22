#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a readable Smoke Test Report from a JSON artifact and a Markdown template.

Goals:
- Deterministic, repeatable rendering (no manual editing)
- Clear separation and labeling of Deterministic (-m smoke) vs Live/Non-deterministic (-m ndsmoke) suites
- Eliminate accidental strikethroughs or confusing formatting (prefer inline code for commands)
- Auto-compute Overall Status, KPIs, per-suite status, and a concise Overall Summary
- Populate a results table with fast-scannable emoji statuses

Supported Input JSON (flexible):
- Native schema:
{
  "run_meta": {
    "run_id": "2025-09-21 08:10 EDT",
    "author": "operator",
    "repo_branch": "org/repo@branch",
    "commit_short": "abc123",
    "commit_full": "abc123...def",
    "host": "Linux / 8c / 32GB / RTX 4090",
    "python": "3.12.6",
    "ollama": "localhost:11434",
    "models": ["deepseek-coder:33b"],
    "mini_agent_stack": {"api": "up", "exec": "up", "tools": "up"}
  },
  "tests": [
    {
      "nodeid": "tests/smoke/test_multilang_ollama_loop_ndsmoke.py::[ts]",
      "suite": "live",        # "deterministic" | "live" (optional; inferred if missing)
      "status": "passed",     # "passed" | "failed" | "skipped"
      "duration_s": 7.3,      # seconds (float); accepts time, time_s, duration, duration_s
      "model": "deepseek-coder:33b",
      "iters": 2,
      "notes": "TS_OK",
      "markers": ["ndsmoke"]  # optional; helps suite inference
    }
  ]
}

- Pytest JSON style:
{
  "created": "...",
  "duration": 123.4,
  "tests": [
    {
      "nodeid": "tests/smoke/test_foo.py::test_bar",
      "outcome": "passed",
      "duration": 0.12,
      "keywords": ["smoke"],
      "markers": ["smoke"]
    }
  ]
}

Usage:
  # Single file (either deterministic or live):
  python scripts/report/generate_smoke_report.py \
    --results local/artifacts/smoke_results.json \
    --template local/docs/01_guides/SMOKE_REPORT_TEMPLATE.md \
    --outdir local/artifacts/reports \
    [--outname smoke_report_YYYYMMDD_HHMMSS.md]

  # Multiple files (merge deterministic + live into one report):
  python scripts/report/generate_smoke_report.py \
    --results local/artifacts/smoke_results_det.json \
    --results local/artifacts/smoke_results_nd.json \
    --template local/docs/01_guides/SMOKE_REPORT_TEMPLATE.md \
    --outdir local/artifacts/reports \
    [--outname smoke_report_YYYYMMDD_HHMMSS.md]

The template must include the anchor:
  <!-- OVERALL_SUMMARY_ANCHOR -->

What this script does:
- Replace the Overall Status line to a concrete one with a single state
- Insert an auto-generated "## 🧾 Overall Summary" at the anchor
- Fill KPIs (Total/Passed/Failed/Skipped/Duration)
- Tag suites clearly as [Deterministic] and [Live]
- Render a results table with emoji statuses

Exit codes:
- 0 on success, non-zero on failure.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


EMOJI = {
    "passed": "✅",
    "failed": "❌",
    "skipped": "⏭️",
    "timeout": "⏱️",
    "flaky": "🟡",
}

# Match the Overall Status backticked content to replace it with a concrete status
OVERALL_STATUS_FLEX_RE = re.compile(
    r"(##\s+🚦\s*Overall Status:\s*`)([^`]*)(`)", re.MULTILINE
)

SUMMARY_ANCHOR = "<!-- OVERALL_SUMMARY_ANCHOR -->"


@dataclass
class TestRecord:
    nodeid: str
    status: str
    duration_s: float = 0.0
    suite: Optional[str] = None  # "deterministic" or "live"
    model: Optional[str] = None
    iters: Optional[int] = None
    notes: Optional[str] = None
    error: Optional[str] = None
    markers: List[str] = field(default_factory=list)


@dataclass
class Aggregates:
    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    duration_s: float = 0.0


def _mmss(seconds: float) -> str:
    seconds = int(round(seconds))
    m, s = divmod(seconds, 60)
    return f"{m:02d}:{s:02d}"


def infer_suite(nodeid: str, markers: List[str]) -> str:
    mset = {str(m).lower() for m in (markers or [])}
    if "ndsmoke" in mset or "live" in mset:
        return "live"
    if "smoke" in mset or "deterministic" in mset:
        return "deterministic"
    nid = (nodeid or "").lower()
    if "ndsmoke" in nid or "live" in nid:
        return "live"
    return "deterministic"


def load_results(path: Path) -> Tuple[List[TestRecord], Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    run_meta = data.get("run_meta", {})

    tests: List[TestRecord] = []

    # Native schema
    tests_raw = data.get("tests")
    if tests_raw and isinstance(tests_raw, list):
        for t in tests_raw:
            nodeid = t.get("nodeid") or t.get("name") or ""
            if not nodeid:
                continue
            status = (
                t.get("status")
                or t.get("outcome")
                or t.get("result")
                or t.get("state")
                or "skipped"
            )
            dur = (
                t.get("duration_s")
                or t.get("time_s")
                or t.get("duration")
                or t.get("time")
                or t.get("call_duration")
                or 0.0
            )
            try:
                duration_s = float(dur)
            except Exception:
                duration_s = 0.0

            markers = t.get("markers") or t.get("keywords") or []
            if isinstance(markers, dict):
                markers = list(markers.keys())
            if not isinstance(markers, list):
                markers = [str(markers)]

            suite = t.get("suite")
            if suite is None:
                suite = infer_suite(nodeid, markers)

            tests.append(
                TestRecord(
                    nodeid=nodeid,
                    status=str(status).lower(),
                    duration_s=duration_s,
                    suite=str(suite).lower() if suite else None,
                    model=t.get("model"),
                    iters=t.get("iters"),
                    notes=t.get("notes"),
                    error=t.get("error"),
                    markers=[str(m) for m in markers],
                )
            )
        return tests, run_meta

    # Pytest-json-report variant
    possible = data.get("report") or data
    tests_raw = possible.get("tests") if isinstance(possible, dict) else None
    if tests_raw and isinstance(tests_raw, list):
        for t in tests_raw:
            nodeid = t.get("nodeid") or t.get("name") or ""
            if not nodeid:
                continue
            status = t.get("outcome") or t.get("status") or "skipped"
            dur = t.get("duration") or t.get("call_duration") or 0.0
            try:
                duration_s = float(dur)
            except Exception:
                duration_s = 0.0
            markers = t.get("markers") or t.get("keywords") or []
            if isinstance(markers, dict):
                markers = list(markers.keys())
            if not isinstance(markers, list):
                markers = [str(markers)]
            suite = infer_suite(nodeid, markers)
            tests.append(
                TestRecord(
                    nodeid=nodeid,
                    status=str(status).lower(),
                    duration_s=duration_s,
                    suite=suite,
                    error=t.get("error"),
                    markers=[str(m) for m in markers],
                )
            )
        return tests, run_meta

    # Fallback: no tests found
    return tests, run_meta


def load_and_merge_results(paths: List[Path]) -> Tuple[List[TestRecord], Dict[str, Any]]:
    """
    Load multiple result files and merge tests and run_meta.
    - Tests are concatenated.
    - run_meta is shallow-merged (last file wins on key conflicts).
    """
    all_tests: List[TestRecord] = []
    merged_meta: Dict[str, Any] = {}
    for p in paths:
        try:
            _tests, _meta = load_results(p)
            all_tests.extend(_tests)
            if isinstance(_meta, dict):
                merged_meta.update(_meta)
        except Exception as e:
            # best-effort: continue on bad file
            merged_meta.setdefault("errors", []).append({"file": str(p), "error": str(e)})
            continue
    return all_tests, merged_meta


def aggregate_tests(tests: List[TestRecord]) -> Aggregates:
    agg = Aggregates()
    for t in tests:
        agg.total += 1
        if t.status == "passed":
            agg.passed += 1
        elif t.status == "failed":
            agg.failed += 1
        else:
            agg.skipped += 1
        agg.duration_s += t.duration_s
    return agg


def per_suite_status(tests: List[TestRecord], suite_name: str) -> Tuple[str, int, int]:
    suite_tests = [t for t in tests if (t.suite or "") == suite_name]
    if not suite_tests:
        return EMOJI["skipped"], 0, 0
    p = sum(1 for t in suite_tests if t.status == "passed")
    f = sum(1 for t in suite_tests if t.status == "failed")
    if f > 0:
        return EMOJI["failed"], p, len(suite_tests)
    if p == 0:
        return EMOJI["skipped"], p, len(suite_tests)
    return EMOJI["passed"], p, len(suite_tests)


def overall_status(d_status: str, l_status: str, have_data: bool) -> str:
    if not have_data:
        return "🟡 NEEDS TRIAGE"
    if d_status == EMOJI["failed"] or l_status == EMOJI["failed"]:
        return "❌ FAIL"
    if d_status == EMOJI["passed"] and l_status in (EMOJI["passed"], EMOJI["skipped"]):
        return "✅ PASS"
    if l_status == EMOJI["passed"] and d_status in (EMOJI["passed"], EMOJI["skipped"]):
        return "✅ PASS"
    return "🟡 NEEDS TRIAGE"


def detect_key_checks(tests: List[TestRecord]) -> List[str]:
    """
    Heuristics to surface important bullets at top-level summary:
    - Python repair loop (mini-agent iterate)
    - Multilang loop
    - Chutes live (routing)
    - Escalation on budget (if visible)
    """
    lines: List[str] = []

    # Repair loop: choose any node containing "compress_runs" or "repair" or "mini_agent" and "iterate"
    repair = next(
        (
            t
            for t in tests
            if ("compress_runs" in t.nodeid.lower()
                or "repair" in t.nodeid.lower()
                or ("mini_agent" in t.nodeid.lower() and "iterate" in t.nodeid.lower()))
        ),
        None,
    )
    if repair:
        lines.append(
            f"Python repair loop: {EMOJI.get(repair.status, '⏭️')} "
            f"(time { _mmss(repair.duration_s) }; iters={repair.iters if repair.iters is not None else 'n/a'})"
        )

    # Multilang aggregate: any node with "multilang"
    mls = [t for t in tests if "multilang" in t.nodeid.lower()]
    if mls:
        p = sum(1 for t in mls if t.status == "passed")
        lines.append(f"Multilang: {('✅' if p == len(mls) else '❌')} ({p}/{len(mls)})")

    # Chutes live
    chutes = [t for t in tests if "chutes" in t.nodeid.lower()]
    if chutes:
        p = sum(1 for t in chutes if t.status == "passed")
        lines.append(f"Chutes live: {('✅' if p == len(chutes) else '❌')} ({p}/{len(chutes)})")

    # Escalation heuristic
    esc = next((t for t in tests if "escalat" in t.nodeid.lower() or "escalat" in (t.notes or "").lower()), None)
    if esc:
        lines.append(f"Escalation on budget: {EMOJI.get(esc.status, '⏭️')} (model={esc.model or 'n/a'})")

    return lines


def render_overall_summary(d_status: str, d_p: int, d_t: int, l_status: str, l_p: int, l_t: int, key_checks: List[str]) -> str:
    lines: List[str] = []
    lines.append("## 🧾 Overall Summary")
    lines.append(f"- Suites: Deterministic {d_status} ({d_p}/{d_t}) | Live {l_status} ({l_p}/{l_t})")
    if key_checks:
        lines.append("- Key checks:")
        for kc in key_checks:
            lines.append(f"  - {kc}")
    return "\n".join(lines) + "\n"


def replace_overall_status(md: str, concrete: str) -> str:
    def _sub(m: re.Match) -> str:
        return f"{m.group(1)}{concrete}{m.group(3)}"
    return OVERALL_STATUS_FLEX_RE.sub(_sub, md, count=1)


def fill_kpis(md: str, agg: Aggregates) -> str:
    # Replace both escaped (&lt;N&gt;) and raw (<N>) placeholders
    replacements = {
        "&lt;N&gt;": str(agg.total),
        "&lt;P&gt;": str(agg.passed),
        "&lt;F&gt;": str(agg.failed),
        "&lt;S&gt;": str(agg.skipped),
        "&lt;mm:ss&gt;": _mmss(agg.duration_s),
        "<N>": str(agg.total),
        "<P>": str(agg.passed),
        "<F>": str(agg.failed),
        "<S>": str(agg.skipped),
        "<mm:ss>": _mmss(agg.duration_s),
    }
    for k, v in replacements.items():
        md = md.replace(k, v)
    return md


def insert_overall_summary(md: str, summary: str) -> str:
    # If a previous "## 🧾 Overall Summary" exists, remove it up to the next top-level heading to avoid duplication.
    if "## 🧾 Overall Summary" in md:
        parts = md.split("## 🧾 Overall Summary", 1)
        head, tail = parts[0], parts[1]
        # Cut tail at next "## " or end
        nxt = tail.find("\n## ")
        if nxt != -1:
            tail = tail[nxt+1:]  # keep the '## ' in tail
            md = head + tail
        else:
            md = head

    # Replace the anchor with anchor + summary (keeps anchor for idempotence)
    return md.replace(SUMMARY_ANCHOR, f"{SUMMARY_ANCHOR}\n{summary}")


def replace_suite_block_lines(md: str, heading_prefix: str, status_emoji: str, passed: int, total: int) -> str:
    """
    Find the heading (e.g., '### Suite: Deterministic') and update the 'Status' and 'Result' lines beneath it.
    """
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith(heading_prefix):
            # scan next ~10 lines for Status/Result
            for j in range(i+1, min(i+12, len(lines))):
                if lines[j].strip().startswith("- **Status:**"):
                    lines[j] = f"- **Status:** `{status_emoji}`"
                if lines[j].strip().startswith("- **Result:**"):
                    lines[j] = f"- **Result:** `{passed}/{total} passed`"
                # stop early if next heading
                if lines[j].strip().startswith("### Suite:") and j > i+1:
                    break
            break
        i += 1
    return "\n".join(lines) + ("\n" if md.endswith("\n") else "")


def render_results_table_rows(tests: List[TestRecord]) -> List[str]:
    def status_emoji(t: TestRecord) -> str:
        return EMOJI.get(t.status, "⏭️")

    rows: List[str] = []
    # Sort by suite then nodeid for stable output
    for t in sorted(tests, key=lambda x: ((x.suite or ""), x.nodeid)):
        time_str = _mmss(t.duration_s) if t.duration_s else "00:00"
        model = f"`{t.model}`" if t.model else ""
        iters = str(t.iters) if (t.iters is not None) else ""
        note_src = t.notes or t.error or ""
        if note_src and len(note_src) > 140:
            note_src = note_src[:140] + "…"
        rows.append(f"| `{t.nodeid}` | {status_emoji(t)} | {time_str} | {model} | {iters} | {note_src} |")
    return rows


def replace_results_table(md: str, tests: List[TestRecord]) -> str:
    """
    Replace rows in the Results Table, keeping header, alignment row, and Tip intact.
    """
    lines = md.splitlines()
    # Find table header
    hdr_idx = None
    tip_idx = None
    for idx, line in enumerate(lines):
        if line.strip().startswith("| Test Node ID |"):
            hdr_idx = idx
            continue
        if hdr_idx is not None and tip_idx is None and line.strip().startswith("> Tip:"):
            tip_idx = idx
            break

    # If we couldn't find both, bail out gracefully
    if hdr_idx is None or tip_idx is None:
        return md

    # The line immediately after header is the alignment row, keep both
    align_idx = hdr_idx + 1
    # Rows start at hdr_idx+2 up to tip_idx-1 (inclusive)
    new_rows = render_results_table_rows(tests)
    new_block = [lines[hdr_idx], lines[align_idx]] + new_rows

    updated = lines[:hdr_idx] + new_block + lines[tip_idx:]
    return "\n".join(updated) + ("\n" if md.endswith("\n") else "")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Smoke Test Report from JSON results and Markdown template.")
    parser.add_argument(
        "--results",
        required=True,
        action="append",
        help="Path(s) to JSON results; pass multiple --results to merge det + live",
    )
    parser.add_argument("--template", required=True, help="Path to Markdown template (e.g., local/docs/01_guides/SMOKE_REPORT_TEMPLATE.md)")
    parser.add_argument("--outdir", required=True, help="Output directory for the generated report (e.g., local/artifacts/reports)")
    parser.add_argument("--outname", default="", help="Optional output filename (default: smoke_report_YYYYMMDD_HHMMSS.md)")
    args = parser.parse_args()

    # Support multiple results files (merge det + live)
    results_paths = [Path(p) for p in (args.results if isinstance(args.results, list) else [args.results])]
    template_path = Path(args.template)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tests, run_meta = load_and_merge_results(results_paths)
    have_data = len(tests) > 0
    agg = aggregate_tests(tests)
    d_status, d_p, d_t = per_suite_status(tests, "deterministic")
    l_status, l_p, l_t = per_suite_status(tests, "live")
    overall = overall_status(d_status, l_status, have_data)
    key_checks = detect_key_checks(tests)
    summary = render_overall_summary(d_status, d_p, d_t, l_status, l_p, l_t, key_checks)

    md = template_path.read_text(encoding="utf-8")

    # 1) Overall Status
    md = replace_overall_status(md, overall)

    # 2) Insert Overall Summary at anchor
    md = insert_overall_summary(md, summary)

    # 3) Fill KPIs
    md = fill_kpis(md, agg)

    # 4) Update suite blocks
    md = replace_suite_block_lines(md, "### Suite: Deterministic", d_status, d_p, d_t)
    md = replace_suite_block_lines(md, "### Suite: Live", l_status, l_p, l_t)

    # 5) Results Table
    md = replace_results_table(md, tests)

    # Output file name
    outname = args.outname.strip()
    if not outname:
        outname = f"smoke_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    out_path = outdir / outname
    out_path.write_text(md, encoding="utf-8")

    print(f"WROTE REPORT: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
