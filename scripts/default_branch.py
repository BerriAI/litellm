from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Final


def _git(repo_root: Path, *args: str) -> str:
    try:
        result: Final = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(
            "Cannot verify the base branch against origin. Check remote access, "
            "or supply an explicit base ref (--base / BASE_REF). "
            f"Git operation failed: {exc}"
        ) from exc
    return result.stdout.strip()


def default_branch(repo_root: Path) -> str:
    output: Final = _git(repo_root, "ls-remote", "--symref", "origin", "HEAD")
    branches: Final = tuple(
        line.removeprefix("ref: refs/heads/").removesuffix("\tHEAD")
        for line in output.splitlines()
        if line.startswith("ref: refs/heads/") and line.endswith("\tHEAD")
    )
    if len(branches) != 1:
        raise SystemExit("Origin did not advertise a default branch. Supply an explicit base ref (--base / BASE_REF).")
    _git(repo_root, "check-ref-format", f"refs/heads/{branches[0]}")
    return branches[0]


def resolve_base_ref(base_ref: str | None, repo_root: Path) -> str:
    if base_ref:
        return base_ref
    branch: Final = default_branch(repo_root)
    _git(repo_root, "fetch", "--quiet", "origin", f"+refs/heads/{branch}:refs/remotes/origin/{branch}")
    return f"origin/{branch}"


def main() -> None:
    parser: Final = argparse.ArgumentParser(description="Resolve the live default branch of origin.")
    parser.add_argument("--base", help="Explicit comparison ref; skips default-branch discovery")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--branch", action="store_true", help="Print only the default branch name, without fetching")
    args: Final = parser.parse_args()
    print(default_branch(args.repo_root) if args.branch else resolve_base_ref(args.base, args.repo_root))


if __name__ == "__main__":
    main()
