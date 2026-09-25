#!/usr/bin/env python3
"""Print `version=X.Y.0` from [project].version in pyproject.toml for $GITHUB_OUTPUT.

Usage
-----
    python3 read_rc_version.py [path/to/pyproject.toml] >> "$GITHUB_OUTPUT"

Exit code 1 with a `::error::` line on stderr when the version is not an X.Y.0 release.
"""

from __future__ import annotations

import pathlib
import re
import sys
from typing import Final

import tomllib

RELEASE_VERSION: Final = re.compile(r"[0-9]+\.[0-9]+\.0")


def read_version(pyproject: pathlib.Path) -> str:
    with pyproject.open("rb") as f:
        return tomllib.load(f)["project"]["version"]


def main(argv: list[str]) -> int:
    pyproject: Final = pathlib.Path(argv[1]) if len(argv) > 1 else pathlib.Path("pyproject.toml")
    version: Final = read_version(pyproject)
    if RELEASE_VERSION.fullmatch(version) is None:
        print(  # noqa: T201  # the ::error:: line to stderr is the workflow's failure signal
            f"::error::pyproject.toml version {version} is not an X.Y.0 release version", file=sys.stderr
        )
        return 1
    print(f"version={version}")  # noqa: T201  # stdout line is appended to $GITHUB_OUTPUT
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
