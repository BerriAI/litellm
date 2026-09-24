#!/usr/bin/env python3
"""Requires every tests/python_*.rs file in the cost crate to name the Python
test it mirrors with a `// mirrors: <target>` line in its header.

The target names a real file under litellm/tests/ (or the fixture generator
when the file pins executed-Python output rather than a specific test), and
may qualify a test with `::` segments. Exit 1 with one line per offender.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
COST_TESTS = REPO_ROOT / "litellm-rust" / "crates" / "cost" / "tests"
PYTHON_TESTS = REPO_ROOT / "tests"
GENERATORS = {
    "crates/cost/tests/generate_python_fixtures.py",
    "crates/cost/tests/generate_python_reference.py",
}
MIRROR_RE = re.compile(r"^//\s*mirrors:\s*(\S.*)$", re.MULTILINE)
TARGET_RE = re.compile(r"^([\w./-]+\.py)(?:::[\w.-]+)*$")


def header(source: str) -> str:
    lines = source.splitlines()
    body_start = next(
        (index for index, line in enumerate(lines) if line.strip().startswith(("mod ", "use ", "fn ", "#["))),
        len(lines),
    )
    return "\n".join(lines[:body_start])


def main() -> int:
    rust_files = sorted(COST_TESTS.glob("python_*.rs"))
    if not rust_files:
        print("no tests/python_*.rs files found; checker is misconfigured", file=sys.stderr)
        return 1
    offenders: list[str] = []
    for rust_file in rust_files:
        source = rust_file.read_text()
        markers = MIRROR_RE.findall(header(source))
        if not markers:
            offenders.append(f"{rust_file.name}: missing `// mirrors:` marker")
            continue
        for target in markers:
            match = TARGET_RE.match(target)
            if not match:
                offenders.append(f"{rust_file.name}: malformed mirrors target {target!r}")
                continue
            path = match.group(1)
            if path in GENERATORS:
                continue
            if not (PYTHON_TESTS / path.removeprefix("litellm/tests/")).is_file():
                offenders.append(
                    f"{rust_file.name}: mirrors target {path!r} is not a file under litellm/tests/"
                )
    for line in offenders:
        print(line)
    return 1 if offenders else 0


if __name__ == "__main__":
    raise SystemExit(main())
