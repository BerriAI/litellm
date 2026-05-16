"""Self-contained matrix plumbing.

This conftest adds ONLY:

* a ``pytest_terminal_summary`` hook that, when any matrix cell fails,
  reprints the result as the at-a-glance route x effort grid the manual QA
  sweep produced (scattered xdist failures are otherwise unreadable);
* a ``ci_record_enabled`` gate used to skip the live VCR-recorded
  provider-acceptance test off-CI.

It does NOT touch the shared ``tests/llm_translation/conftest.py`` VCR
plumbing — pytest loads that parent conftest for this subdirectory
automatically, so the live test still records/replays through Redis-VCR in
CI with zero modification to existing files.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, Tuple

_MATRIX_FILE = "test_reasoning_effort_wire_matrix.py"

# (route, entrypoint, canonical, effort_label) -> "pass" | "fail" | "skip"
_OUTCOMES: Dict[Tuple[str, str, str, str], str] = {}


def ci_record_enabled() -> bool:
    """True only where the live provider-acceptance call should run.

    Mirrors the repo's VCR gate: a recorded/replayed call needs
    ``CASSETTE_REDIS_URL`` (a CircleCI project secret). Off-CI it is unset,
    so the live half stays dormant and the deterministic offline half is the
    whole local signal. ``MATRIX_FORCE_LIVE=1`` is an explicit local override.
    """
    if os.environ.get("LITELLM_VCR_DISABLE") == "1":
        return os.environ.get("MATRIX_FORCE_LIVE") == "1"
    return bool(os.environ.get("CASSETTE_REDIS_URL")) or (
        os.environ.get("MATRIX_FORCE_LIVE") == "1"
    )


def _parse_cell_id(nodeid: str):
    if _MATRIX_FILE not in nodeid or "[" not in nodeid:
        return None
    param = nodeid.split("[", 1)[1].rsplit("]", 1)[0]
    parts = param.split("__")
    if len(parts) != 4:
        return None
    return tuple(parts)  # (route, entrypoint, canonical, effort_label)


def pytest_runtest_logreport(report):
    if report.when != "call" and not (report.when == "setup" and report.skipped):
        return
    cell = _parse_cell_id(report.nodeid)
    if cell is None:
        return
    if report.failed:
        _OUTCOMES[cell] = "fail"
    elif report.skipped:
        _OUTCOMES.setdefault(cell, "skip")
    elif report.passed and report.when == "call":
        _OUTCOMES.setdefault(cell, "pass")


_MARK = {"pass": "✅", "fail": "❌", "skip": "·"}


def pytest_terminal_summary(terminalreporter):
    if not _OUTCOMES or not any(v == "fail" for v in _OUTCOMES.values()):
        return

    tw = terminalreporter
    tw.write_sep("=", "reasoning_effort wire matrix — failure grid")

    # group: (route, entrypoint) -> {canonical -> {effort_label -> mark}}
    grids: Dict[Tuple[str, str], Dict[str, Dict[str, str]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    efforts_seen: Dict[Tuple[str, str], list] = defaultdict(list)
    for (route, entry, canonical, effort), outcome in _OUTCOMES.items():
        grids[(route, entry)][canonical][effort] = _MARK.get(outcome, "?")
        if effort not in efforts_seen[(route, entry)]:
            efforts_seen[(route, entry)].append(effort)

    for (route, entry), rows in sorted(grids.items()):
        has_fail = any(m == "❌" for cols in rows.values() for m in cols.values())
        if not has_fail:
            continue
        cols = efforts_seen[(route, entry)]
        tw.write_line("")
        tw.write_line(f"### {route} / {entry}")
        header = "model".ljust(14) + " | " + " | ".join(c[:7].ljust(7) for c in cols)
        tw.write_line(header)
        tw.write_line("-" * len(header))
        for canonical in sorted(rows):
            line = (
                canonical.ljust(14)
                + " | "
                + " | ".join(rows[canonical].get(c, " ").center(7) for c in cols)
            )
            tw.write_line(line)

    tw.write_line("")
    tw.write_line("Legend: ✅ pass  ❌ fail  · skipped(live/off-CI)")
