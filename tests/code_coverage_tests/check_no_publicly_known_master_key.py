from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Final

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
PUBLICLY_KNOWN_MASTER_KEY_PREFIX: Final = "sk-" + "1234"
GENERATED_DASHBOARD_BUNDLE_PATHSPEC: Final = ":(exclude)litellm/proxy/_experimental/out"


def _file_violations(repo_root: Path, relative_path: str) -> tuple[str, ...]:
    path: Final = repo_root / relative_path
    if path.is_dir():
        return ()
    try:
        contents: Final = path.read_bytes()
    except FileNotFoundError:
        return ()
    if b"\0" in contents:
        return ()
    try:
        text: Final = contents.decode("utf-8")
    except UnicodeDecodeError:
        return ()
    return tuple(
        f"{relative_path}:{line_number}"
        for line_number, line in enumerate(text.splitlines(), start=1)
        if PUBLICLY_KNOWN_MASTER_KEY_PREFIX in line
    )


def _tracked_paths(repo_root: Path) -> tuple[str, ...]:
    result: Final = subprocess.run(
        ["git", "-C", os.fspath(repo_root), "ls-files", "-z", "--", ".", GENERATED_DASHBOARD_BUNDLE_PATHSPEC],
        check=True,
        stdout=subprocess.PIPE,
    )
    return tuple(os.fsdecode(path) for path in result.stdout.split(b"\0") if path)


def violations(repo_root: Path) -> Iterator[str]:
    for path in _tracked_paths(repo_root):
        yield from _file_violations(repo_root, path)


def main() -> int:
    violations_found: Final = tuple(violations(REPO_ROOT))
    for violation in violations_found:
        print(violation)
    if violations_found:
        print(f"\n{len(violations_found)} tracked line(s) contain the publicly known master key prefix")
        return 1
    print("No tracked files contain the publicly known master key prefix")
    return 0


if __name__ == "__main__":
    sys.exit(main())
