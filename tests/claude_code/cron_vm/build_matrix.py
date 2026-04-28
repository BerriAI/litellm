"""Tiny CLI wrapper around `tests.claude_code.matrix_builder.build_from_paths`.

Exists only so `run_daily.sh` can hand the version metadata + paths into
the matrix builder without re-implementing it in bash. All real logic
lives in `matrix_builder.py`, which has its own unit tests under
`_builder_unit_tests/`.

Invoked from the cron worktree (where `uv sync` has installed pyyaml),
not the dev checkout — the bash script `cd`s into the worktree before
`uv run python`-ing this file.
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

from tests.claude_code.matrix_builder import build_from_paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--litellm-version", required=True)
    parser.add_argument("--claude-code-version", required=True)
    args = parser.parse_args()

    generated_at = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    build_from_paths(
        manifest_path=args.manifest,
        results_path=args.results,
        litellm_version=args.litellm_version,
        claude_code_version=args.claude_code_version,
        generated_at=generated_at,
        output_path=args.output,
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
