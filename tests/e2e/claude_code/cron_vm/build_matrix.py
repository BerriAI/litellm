"""Tiny CLI wrapper around `claude_code.matrix_builder.build_from_paths`.

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
from dataclasses import dataclass
from pathlib import Path

from pydantic import TypeAdapter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from claude_code.matrix_builder import build_from_paths  # noqa: E402  # import needs the sys.path bootstrap above


@dataclass(frozen=True, slots=True)
class _Args:
    manifest: Path
    results: Path
    output: Path
    litellm_version: str
    claude_code_version: str


_ARGS_ADAPTER = TypeAdapter[_Args](_Args)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--litellm-version", required=True)
    parser.add_argument("--claude-code-version", required=True)
    args = _ARGS_ADAPTER.validate_python(vars(parser.parse_args()))

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
