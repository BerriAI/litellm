#!/usr/bin/env python3
"""Emit the backend implicit/explicit Any counts for the trend dashboard.

Writes ``backend-lint-metrics.json`` as ``{rule: count}`` for ``reportAny`` (implicit
Any) and ``reportExplicitAny`` (explicit Any). Report-only: nothing gates on it. The
``basedpyright-code-budget.json`` ``limit`` stopped tracking the real count after its
limit collapse (it is now a loose ceiling), so this reports the real count instead.
The counting is reused from ``scripts/type_check_gate.py``, so the numbers match what
CI measures.

Usage:
    basedpyright --outputjson | python scripts/lint_metrics.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import type_check_gate

RULES = ("reportAny", "reportExplicitAny")
METRICS_PATH = Path(__file__).resolve().parent.parent / "backend-lint-metrics.json"


def main() -> None:
    counts = type_check_gate.count_basedpyright(sys.stdin.read())
    result = {rule: counts.get(rule, 0) for rule in RULES}
    if not any(result.values()):
        sys.exit("basedpyright reported no Any diagnostics; it likely crashed. Refusing to write zeros.")
    METRICS_PATH.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {METRICS_PATH.name}: {result}")


if __name__ == "__main__":
    main()
