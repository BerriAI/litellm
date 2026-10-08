"""Unit tests for the Claude Code PR-gate version resolver.

Markerless harness tests: they feed the resolver a hand-built packument and a
fixed clock, so they run without a proxy, never reach the npm registry, and
carry no `e2e` marker.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Final, Mapping

import pytest

from claude_code.pr_gate_version_resolver import NoEligibleVersionError, resolve_pr_gate_version

NOW: Final = datetime(2026, 4, 25, 12, 0, tzinfo=timezone.utc)
INSIDE_THE_2_1_88_WINDOW: Final = datetime(2026, 4, 3, 12, 0, tzinfo=timezone.utc)


def _packument(times: Mapping[str, str], unpublished: frozenset[str] = frozenset()) -> dict[str, object]:
    return {
        "name": "@anthropic-ai/claude-code",
        "time": {"created": "2024-01-01T00:00:00.000Z", "modified": "2026-04-25T00:00:00.000Z", **times},
        "versions": {version: {"version": version} for version in times if version not in unpublished},
    }


def test_skips_a_version_npm_has_unpublished() -> None:
    metadata: Final = _packument(
        {
            "2.1.87": "2026-03-28T20:00:00.000Z",
            "2.1.88": "2026-03-30T22:36:48.424Z",
            "2.1.89": "2026-03-31T23:32:40.000Z",
        },
        unpublished=frozenset({"2.1.88"}),
    )
    assert resolve_pr_gate_version(metadata=metadata, as_of=INSIDE_THE_2_1_88_WINDOW) == "2.1.87"


def test_raises_when_the_only_old_enough_version_is_unpublished() -> None:
    metadata: Final = _packument(
        {"2.1.88": "2026-03-30T22:36:48.424Z", "2.1.89": "2026-03-31T23:32:40.000Z"},
        unpublished=frozenset({"2.1.88"}),
    )
    with pytest.raises(NoEligibleVersionError):
        resolve_pr_gate_version(metadata=metadata, as_of=INSIDE_THE_2_1_88_WINDOW)


def test_picks_the_newest_published_version_at_least_min_age_old() -> None:
    metadata: Final = _packument(
        {
            "2.1.118": "2026-04-15T10:00:00.000Z",
            "2.1.119": "2026-04-21T10:00:00.000Z",
            "2.2.0-rc.1": "2026-04-22T10:00:00.000Z",
            "2.1.120": "2026-04-23T10:00:00.000Z",
            "2.1.121": "2026-04-25T11:00:00.000Z",
        }
    )
    assert resolve_pr_gate_version(metadata=metadata, as_of=NOW) == "2.1.119"
