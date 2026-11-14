from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Final

from cli_harness.npm_version_resolver import MIN_AGE, Packument, newest_version_at_least

NOW: Final = datetime(2026, 4, 25, 12, 0, tzinfo=timezone.utc)
INSIDE_THE_2_1_88_WINDOW: Final = datetime(2026, 4, 3, 12, 0, tzinfo=timezone.utc)


def _packument(times: Mapping[str, str], unpublished: frozenset[str] = frozenset()) -> Packument:
    return Packument.from_json(
        {
            "name": "@scope/some-cli",
            "time": {"created": "2024-01-01T00:00:00.000Z", "modified": "2026-04-25T00:00:00.000Z", **times},
            "versions": {version: {"version": version} for version in times if version not in unpublished},
        }
    )


def test_skips_a_version_npm_has_unpublished() -> None:
    packument: Final = _packument(
        {
            "2.1.87": "2026-03-28T20:00:00.000Z",
            "2.1.88": "2026-03-30T22:36:48.424Z",
            "2.1.89": "2026-03-31T23:32:40.000Z",
        },
        unpublished=frozenset({"2.1.88"}),
    )
    assert newest_version_at_least(packument, min_age=MIN_AGE, as_of=INSIDE_THE_2_1_88_WINDOW) == "2.1.87"


def test_finds_nothing_when_the_only_old_enough_version_is_unpublished() -> None:
    packument: Final = _packument(
        {"2.1.88": "2026-03-30T22:36:48.424Z", "2.1.89": "2026-03-31T23:32:40.000Z"},
        unpublished=frozenset({"2.1.88"}),
    )
    assert newest_version_at_least(packument, min_age=MIN_AGE, as_of=INSIDE_THE_2_1_88_WINDOW) is None


def test_picks_the_newest_stable_version_at_least_min_age_old() -> None:
    packument: Final = _packument(
        {
            "2.1.118": "2026-04-15T10:00:00.000Z",
            "2.1.119": "2026-04-21T10:00:00.000Z",
            "2.2.0-rc.1": "2026-04-22T10:00:00.000Z",
            "2.1.120": "2026-04-23T10:00:00.000Z",
            "2.1.121": "2026-04-25T11:00:00.000Z",
        }
    )
    assert newest_version_at_least(packument, min_age=MIN_AGE, as_of=NOW) == "2.1.119"


def test_newest_means_latest_published_not_highest_semver() -> None:
    packument: Final = _packument(
        {
            "3.0.0": "2026-04-01T10:00:00.000Z",
            "2.9.9": "2026-04-10T10:00:00.000Z",
        }
    )
    assert newest_version_at_least(packument, min_age=timedelta(days=3), as_of=NOW) == "2.9.9"
