"""Fixture-mode selection and per-test determinism for record/replay e2e runs.

``E2E_FIXTURE_MODE`` is live (the default; nothing changes), record, or replay.
This module owns everything mode-shaped that is independent of the provider
edge itself: parsing the raw env value, the collection-time gate that aborts a
run whose mode can never work (unknown value, or replay against a missing or
stale bundle), the pytest report-header lines, the running test's node id, and
the deterministic per-test marker that lets a replay run regenerate exactly
the requests the record run sent. The provider-edge server that records and
serves provider traffic lives in provider_edge.py (LIT-5745).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final, Literal, assert_never

from fixture_bundle import (
    FreshBundle,
    StaleBundle,
    UnreadableBundle,
    check_freshness,
    format_age,
)

type FixtureMode = Literal["live", "record", "replay"]

FIXTURE_MODES: Final[tuple[FixtureMode, ...]] = ("live", "record", "replay")

SESSION_TEST_KEY: Final = "session"


@dataclass(frozen=True, slots=True)
class InvalidFixtureMode:
    value: str


def parse_fixture_mode(raw: str) -> FixtureMode | InvalidFixtureMode:
    normalized = raw.strip().lower() or "live"
    match normalized:
        case "live" | "record" | "replay":
            return normalized
        case _:
            return InvalidFixtureMode(value=raw)


def current_test_key() -> str:
    """The pytest node id of the running test, from the PYTEST_CURRENT_TEST env
    var pytest maintains (``<nodeid> (setup|call|teardown)``); ``session`` for
    calls outside any test (e.g. session-finish cleanup)."""
    raw = os.environ.get("PYTEST_CURRENT_TEST", "")
    if not raw:
        return SESSION_TEST_KEY
    return raw.rsplit(" (", 1)[0]


class ReplayMiss(AssertionError):
    """Replay had no recorded interaction for a provider call the proxy made.
    The suite drifted from the bundle (or the bundle from the suite): re-record."""


_marker_ordinals: Final[dict[str, int]] = {}


def deterministic_marker() -> str:
    """Stable stand-in for uuid-based unique markers in record and replay modes:
    the Nth marker of a test is a pure function of the test's node id and N, so a
    replay run regenerates exactly the model names, prompts, and tags the record
    run sent and every recorded provider interaction still matches its key."""
    test_key = current_test_key()
    ordinal = _marker_ordinals.get(test_key, 0)
    _marker_ordinals[test_key] = ordinal + 1
    return hashlib.sha1(f"{test_key}#{ordinal}".encode()).hexdigest()[:12]


def fixture_mode_collection_error(mode_raw: str, bundle_dir: Path, *, now: datetime) -> str | None:
    """Session-abort reason for a fixture-mode setup that can never work, or None.
    Called at collection time (conftest pytest_sessionstart) so a stale or missing
    bundle fails the whole run up front, naming the bundle age, instead of failing
    every test individually."""
    mode = parse_fixture_mode(mode_raw)
    match mode:
        case InvalidFixtureMode(value=value):
            return f"E2E_FIXTURE_MODE={value!r} is not one of {', '.join(FIXTURE_MODES)}"
        case "live" | "record":
            return None
        case "replay":
            freshness = check_freshness(bundle_dir, now=now)
            match freshness:
                case FreshBundle():
                    return None
                case StaleBundle(recorded_at=recorded_at, age=age, limit=limit):
                    return (
                        f"fixture bundle at {bundle_dir} is stale: recorded {recorded_at.isoformat()}, "
                        f"age {format_age(age)} exceeds the {limit.days}-day limit; "
                        "re-record with E2E_FIXTURE_MODE=record"
                    )
                case UnreadableBundle(reason=reason):
                    return f"E2E_FIXTURE_MODE=replay cannot use bundle at {bundle_dir}: {reason}"
                case _:
                    assert_never(freshness)
        case _:
            assert_never(mode)


def fixture_report_lines(mode_raw: str, bundle_dir: Path, *, now: datetime) -> list[str]:
    """pytest report-header lines; empty in live mode so an unset
    E2E_FIXTURE_MODE keeps today's output byte-identical."""
    mode = parse_fixture_mode(mode_raw)
    match mode:
        case InvalidFixtureMode() | "live":
            return []
        case "record":
            return [f"e2e fixture mode: record -> {bundle_dir}"]
        case "replay":
            freshness = check_freshness(bundle_dir, now=now)
            match freshness:
                case FreshBundle(manifest=manifest):
                    return [
                        f"e2e fixture mode: replay <- {bundle_dir} "
                        f"(recorded {manifest.recorded_at.isoformat()}, harness {manifest.harness_version})"
                    ]
                case StaleBundle() | UnreadableBundle():
                    return [f"e2e fixture mode: replay <- {bundle_dir}"]
                case _:
                    assert_never(freshness)
        case _:
            assert_never(mode)
