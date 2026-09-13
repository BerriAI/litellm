"""Tiny CLI wrapper around `claude_code.matrix_builder.build_from_paths`.

Exists only so `run_daily.sh` can hand the version metadata + paths into
the matrix builder without re-implementing it in bash. All real logic
lives in `matrix_builder.py`.

The suite imports its own modules with `tests/e2e/` on sys.path (that is
how pytest resolves them: `tests/e2e/` has no `__init__.py`, while
`claude_code/` does), so this script bootstraps the same root — two
levels up from this file — before importing.
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from claude_code.matrix_builder import (
    build_from_paths,
)  # noqa: E402  # needs the sys.path bootstrap above


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
    print(f"wrote {args.output}")  # noqa: T201  # CLI output read by run_daily.sh
    return 0


if __name__ == "__main__":
    sys.exit(main())
