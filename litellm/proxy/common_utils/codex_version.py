"""Compare Codex `client_version` strings, which are semver tags such as `0.166.0`, `0.166.0-alpha.5` and the
patch build `0.166.0-alpha.5.1`."""

from __future__ import annotations

import re
from typing import Final

_CODEX_VERSION: Final = re.compile(r"(\d{1,9})\.(\d{1,9})\.(\d{1,9})(?:-alpha\.(\d{1,9})(?:\.(\d{1,9}))?)?")


def _sort_key(version: str) -> tuple[int, int, int, bool, int, int] | None:
    match: Final = _CODEX_VERSION.fullmatch(version)
    if match is None:
        return None
    major, minor, patch, alpha, alpha_patch = match.groups()
    return (int(major), int(minor), int(patch), alpha is None, int(alpha or 0), int(alpha_patch or 0))


def codex_version_at_least(client_version: str | None, first: str) -> bool:
    """Whether `client_version` is `first` or newer. A missing or unreadable version counts as older."""
    current: Final = None if client_version is None else _sort_key(client_version)
    threshold: Final = _sort_key(first)
    return current is not None and threshold is not None and current >= threshold
