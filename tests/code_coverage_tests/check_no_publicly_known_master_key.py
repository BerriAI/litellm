from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Final

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
PUBLICLY_KNOWN_MASTER_KEY_PREFIX: Final = "sk-" + "1234"


def _file_violations(relative_path: str) -> tuple[str, ...]:
    path: Final = REPO_ROOT / relative_path
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


def _violations(tracked_paths: tuple[str, ...]) -> Iterator[str]:
    for path in tracked_paths:
        yield from _file_violations(path)


def main() -> int:
    result: Final = subprocess.run(
        ["git", "-C", os.fspath(REPO_ROOT), "ls-files", "-z"],
        check=True,
        stdout=subprocess.PIPE,
    )
    tracked_paths: Final = tuple(
        os.fsdecode(path) for path in result.stdout.split(b"\0") if path
    )
    violations: Final = tuple(_violations(tracked_paths))
    for violation in violations:
        print(violation)
    if violations:
        print(f"\n{len(violations)} tracked line(s) contain the publicly known master key prefix")
        return 1
    print("No tracked files contain the publicly known master key prefix")
    return 0


if __name__ == "__main__":
    sys.exit(main())
