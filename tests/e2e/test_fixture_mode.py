"""Harness coverage for fixture-mode selection and determinism (LIT-5729/LIT-5745).

No proxy and no ``e2e`` marker. Pins the mode parser, the deterministic
per-test marker sequence a replay run must regenerate, the collection-time
gate (including the stale message that names the bundle's age), and the pytest
report header. The provider-edge record/replay behavior itself is pinned in
test_provider_edge.py.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from fixture_bundle import BUNDLE_FORMAT_VERSION, MANIFEST_FILENAME, Manifest
from fixture_mode import (
    InvalidFixtureMode,
    current_test_key,
    deterministic_marker,
    fixture_mode_collection_error,
    fixture_report_lines,
    parse_fixture_mode,
)

NOW = datetime(2026, 8, 18, 12, 0, 0, tzinfo=timezone.utc)


def write_manifest(root: Path, recorded_at: datetime) -> None:
    root.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(
        format_version=BUNDLE_FORMAT_VERSION, recorded_at=recorded_at, harness_version="abc1234"
    )
    (root / MANIFEST_FILENAME).write_text(manifest.model_dump_json(), encoding="utf-8")


class TestParseFixtureMode:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("live", "live"), ("record", "record"), ("replay", "replay"), ("", "live"), ("  REPLAY  ", "replay")],
    )
    def test_known_values_normalize(self, raw: str, expected: str) -> None:
        assert parse_fixture_mode(raw) == expected

    def test_unknown_value_is_invalid_with_the_original_spelling(self) -> None:
        assert parse_fixture_mode("cached") == InvalidFixtureMode(value="cached")


class TestDeterministicMarker:
    def test_sequence_is_a_pure_function_of_test_and_ordinal(self) -> None:
        """A replay process must regenerate exactly the markers the record
        process generated, so the Nth marker of a test is pinned to a pure
        function of the node id and N."""
        key = current_test_key()
        assert deterministic_marker() == hashlib.sha1(f"{key}#0".encode()).hexdigest()[:12]
        assert deterministic_marker() == hashlib.sha1(f"{key}#1".encode()).hexdigest()[:12]


class TestCurrentTestKey:
    def test_names_this_test_and_strips_the_phase(self) -> None:
        key = current_test_key()
        assert key.endswith("TestCurrentTestKey::test_names_this_test_and_strips_the_phase")
        assert "(call)" not in key


class TestCollectionGate:
    def test_invalid_mode_names_the_value_and_the_choices(self, tmp_path: Path) -> None:
        assert (
            fixture_mode_collection_error("cached", tmp_path, now=NOW)
            == "E2E_FIXTURE_MODE='cached' is not one of live, record, replay"
        )

    @pytest.mark.parametrize("mode_raw", ["live", "", "record"])
    def test_live_and_record_never_block_collection(self, mode_raw: str, tmp_path: Path) -> None:
        assert fixture_mode_collection_error(mode_raw, tmp_path / "missing", now=NOW) is None

    def test_replay_with_no_bundle_says_how_to_record_one(self, tmp_path: Path) -> None:
        reason = fixture_mode_collection_error("replay", tmp_path / "missing", now=NOW)
        assert reason is not None
        assert f"no {MANIFEST_FILENAME}" in reason
        assert "E2E_FIXTURE_MODE=record" in reason

    def test_stale_replay_bundle_fails_naming_its_age(self, tmp_path: Path) -> None:
        root = tmp_path / "bundle"
        write_manifest(root, NOW - timedelta(days=9, hours=5))
        reason = fixture_mode_collection_error("replay", root, now=NOW)
        assert reason is not None
        assert "age 9d5h exceeds the 7-day limit" in reason
        assert "re-record with E2E_FIXTURE_MODE=record" in reason

    def test_fresh_replay_bundle_collects(self, tmp_path: Path) -> None:
        root = tmp_path / "bundle"
        write_manifest(root, NOW - timedelta(days=2))
        assert fixture_mode_collection_error("replay", root, now=NOW) is None


class TestReportHeader:
    def test_live_mode_prints_nothing(self, tmp_path: Path) -> None:
        assert fixture_report_lines("live", tmp_path, now=NOW) == []
        assert fixture_report_lines("", tmp_path, now=NOW) == []

    def test_record_and_replay_name_the_bundle(self, tmp_path: Path) -> None:
        root = tmp_path / "bundle"
        recorded_at = NOW - timedelta(days=1)
        write_manifest(root, recorded_at)
        assert fixture_report_lines("record", root, now=NOW) == [
            f"e2e fixture mode: record -> {root}"
        ]
        replay_lines = fixture_report_lines("replay", root, now=NOW)
        assert len(replay_lines) == 1
        assert "replay" in replay_lines[0]
        assert recorded_at.isoformat() in replay_lines[0]
