"""
Fail CI when generated package artifacts are committed to git.

This keeps pull requests reviewable and reduces the risk of opaque release
artifacts being merged into source control.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import PurePosixPath


def is_build_artifact_path(path: str) -> bool:
    """Return True when a changed path points to a generated build artifact."""
    normalized = path.replace("\\", "/")
    pure_path = PurePosixPath(normalized)

    if any(part in {"dist", "build"} for part in pure_path.parts):
        return True

    if normalized.endswith(".whl") or normalized.endswith(".tar.gz"):
        return True

    return False


def find_build_artifact_paths(paths: list[str]) -> list[str]:
    """Return build artifact paths from a changed-file list."""
    return [path for path in paths if is_build_artifact_path(path)]


def get_changed_files(base_sha: str, head_sha: str) -> list[str]:
    """Return files changed between two git revisions."""
    result = subprocess.run(
        ["git", "diff", "--name-only", base_sha, head_sha],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", help="Base git SHA for diff checks")
    parser.add_argument("--head", help="Head git SHA for diff checks")
    parser.add_argument(
        "--paths",
        nargs="*",
        default=None,
        help="Optional explicit file paths to scan instead of using git diff",
    )
    return parser


def main() -> int:
    """CLI entrypoint."""
    args = build_parser().parse_args()

    if args.paths is not None and len(args.paths) > 0:
        changed_files = args.paths
    elif args.base and args.head:
        changed_files = get_changed_files(args.base, args.head)
    else:
        print(
            "Pass either --base and --head, or provide one or more paths via --paths.",
            file=sys.stderr,
        )
        return 2

    violating_files = find_build_artifact_paths(changed_files)

    if violating_files:
        print("Build artifacts should not be committed to git.")
        print("Publish archives from the release process instead.")
        for path in violating_files:
            print(f" - {path}")
        return 1

    print("No committed build artifacts detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
