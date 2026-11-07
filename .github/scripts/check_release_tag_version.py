#!/usr/bin/env python3
"""Fail when a release tag and the pyproject version at that tag disagree.

Usage
-----
    python3 check_release_tag_version.py <tag> [path/to/pyproject.toml]

The tag may carry a leading `v` and a PEP 440 suffix (`rc`, `dev`, `post`, or a
vendor suffix such as `-stable`). Only the X.Y.Z release segment is compared, so
`v1.107.0-rc.2` matches a pyproject version of `1.107.0`.

Exit code 1 with a `::error::` line on stderr when the tag and the version do
not name the same release.
"""

from __future__ import annotations

import pathlib
import re
import sys
from typing import Final

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

_RELEASE: Final = re.compile(r"^(?P<release>[0-9]+\.[0-9]+\.[0-9]+)(?P<suffix>.*)$")
_ALLOWED_SUFFIX: Final = re.compile(r"^(?:$|[.\-+].*|(?:a|b|rc|dev|post)[0-9].*)$")


def _release_segment(value: str, *, leading_v: bool) -> str | None:
    candidate: Final = value[1:] if leading_v and value.startswith("v") else value
    match: Final = _RELEASE.fullmatch(candidate)
    if match is None or _ALLOWED_SUFFIX.fullmatch(match.group("suffix")) is None:
        return None
    return match.group("release")


def read_version(pyproject: pathlib.Path) -> str:
    with pyproject.open("rb") as f:
        return tomllib.load(f)["project"]["version"]


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(  # noqa: T201  # the ::error:: line to stderr is the workflow's failure signal
            "::error::usage: check_release_tag_version.py <tag> [path/to/pyproject.toml]",
            file=sys.stderr,
        )
        return 1

    tag: Final = argv[1]
    pyproject: Final = pathlib.Path(argv[2]) if len(argv) > 2 else pathlib.Path("pyproject.toml")
    tag_release: Final = _release_segment(tag, leading_v=True)
    if tag_release is None:
        print(  # noqa: T201  # the ::error:: line to stderr is the workflow's failure signal
            f"::error::{tag} is not a release tag with an X.Y.Z version", file=sys.stderr
        )
        return 1

    version: Final = read_version(pyproject)
    version_release: Final = _release_segment(version, leading_v=False)
    if version_release is None:
        print(  # noqa: T201  # the ::error:: line to stderr is the workflow's failure signal
            f"::error::pyproject.toml version {version} is not an X.Y.Z version", file=sys.stderr
        )
        return 1

    if tag_release != version_release:
        print(  # noqa: T201  # the ::error:: line to stderr is the workflow's failure signal
            f"::error::{tag} points at a commit whose pyproject.toml version is {version}. "
            f"Tag the commit for {tag_release}, or correct the version before tagging.",
            file=sys.stderr,
        )
        return 1

    print(f"{tag} matches pyproject.toml version {version}")  # noqa: T201  # the workflow shows this line
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
