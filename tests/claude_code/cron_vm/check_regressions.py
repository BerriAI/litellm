"""CLI: detect green→red regressions between the published matrix and a
freshly built one, so `run_daily.sh` can decide whether to enable
auto-merge on the daily docs PR.

All real logic lives in `tests.claude_code.matrix_builder.find_regressions`
(unit-tested under `_builder_unit_tests/`); this file only does the I/O and
maps the result onto an exit code the bash caller can branch on.

Exit codes (the bash gate depends on these exact values):

  0  no green→red regressions          -> safe to auto-merge
  3  one or more green→red regressions  -> do NOT auto-merge (human review)
  2  argparse/usage error (argparse default)

The `--old` file is allowed to be missing: on the first-ever publish there
is no baseline to regress against, so we exit 0.

Invoked from the cron worktree (`cd`'d in by run_daily.sh) so the
`tests.claude_code` package import resolves, mirroring build_matrix.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tests.claude_code.matrix_builder import find_regressions

REGRESSION_EXIT = 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--old",
        type=Path,
        required=True,
        help="currently published matrix JSON (may be absent on first publish)",
    )
    parser.add_argument(
        "--new",
        type=Path,
        required=True,
        help="freshly built matrix JSON",
    )
    args = parser.parse_args()

    if not args.old.exists():
        print(
            "no published matrix to compare against "
            "(first publish); treating as no regressions"
        )
        return 0

    old_matrix = json.loads(args.old.read_text())
    new_matrix = json.loads(args.new.read_text())

    regressions = find_regressions(old_matrix, new_matrix)
    if not regressions:
        print("no green->red regressions detected")
        return 0

    print(f"detected {len(regressions)} green->red regression(s):")
    for r in regressions:
        line = f"  - {r['feature_name']} [{r['provider']}]: pass -> fail"
        if r["error"]:
            line += f" ({r['error'][:160]})"
        print(line)
    return REGRESSION_EXIT


if __name__ == "__main__":
    sys.exit(main())
