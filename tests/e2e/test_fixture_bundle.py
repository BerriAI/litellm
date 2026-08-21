"""Harness coverage for the on-disk fixture bundle format (LIT-5729).

No proxy and no ``e2e`` marker: these pin the bundle CONTRACT - the seven-day
freshness gate that names the bundle's age, record mode's wipe safety (never
delete a directory that is not a bundle), collision-free per-test slugs, and
lossless Result round-trips - so replay can never silently drift from what
record wrote.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import BaseModel

from e2e_http import (
    NetworkError,
    RateLimitedError,
    Result,
    Success,
    UnauthorizedError,
    UnknownApiError,
    ValidationError,
)
from fixture_bundle import (
    BUNDLE_FORMAT_VERSION,
    MANIFEST_FILENAME,
    MAX_BUNDLE_AGE,
    BundleRecorder,
    FreshBundle,
    LoadedBundle,
    Manifest,
    RecordedRequest,
    RecordedResult,
    StaleBundle,
    UnreadableBundle,
    UnsafeBundleDir,
    check_freshness,
    format_age,
    from_result,
    interaction_filename,
    load_bundle,
    prepare_bundle,
    slug_for_test,
    to_result,
)

NOW = datetime(2026, 8, 18, 12, 0, 0, tzinfo=timezone.utc)


class Payload(BaseModel):
    value: str


def write_manifest(
    root: Path, recorded_at: datetime, *, format_version: int = BUNDLE_FORMAT_VERSION
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(
        format_version=format_version, recorded_at=recorded_at, harness_version="abc1234"
    )
    (root / MANIFEST_FILENAME).write_text(manifest.model_dump_json(), encoding="utf-8")


def prepared(root: Path) -> BundleRecorder:
    recorder = prepare_bundle(root)
    assert isinstance(recorder, BundleRecorder)
    return recorder


def plain_request(path: str) -> RecordedRequest:
    return RecordedRequest(method="post", path=path, headers={})


class TestResultRoundTrip:
    @pytest.mark.parametrize(
        "result",
        [
            Success(status_code=201, data=Payload(value="ok")),
            NetworkError(message="connection refused"),
            UnauthorizedError(),
            RateLimitedError(retry_after_seconds=7, body="slow down"),
            ValidationError(message="bad shape"),
            UnknownApiError(status_code=502, body="upstream exploded"),
        ],
    )
    def test_every_result_kind_survives_disk_and_back(self, result: Result[Payload]) -> None:
        assert to_result(from_result(result), Payload) == result


class TestFreshness:
    def test_bundle_at_the_limit_is_still_fresh(self, tmp_path: Path) -> None:
        root = tmp_path / "bundle"
        write_manifest(root, NOW - MAX_BUNDLE_AGE)
        assert isinstance(check_freshness(root, now=NOW), FreshBundle)

    def test_stale_bundle_reports_age_and_limit(self, tmp_path: Path) -> None:
        root = tmp_path / "bundle"
        write_manifest(root, NOW - timedelta(days=8, hours=3))
        freshness = check_freshness(root, now=NOW)
        assert isinstance(freshness, StaleBundle)
        assert freshness.age == timedelta(days=8, hours=3)
        assert format_age(freshness.age) == "8d3h"
        assert freshness.limit == MAX_BUNDLE_AGE

    def test_naive_recorded_at_is_read_as_utc(self, tmp_path: Path) -> None:
        root = tmp_path / "bundle"
        write_manifest(root, (NOW - timedelta(days=1)).replace(tzinfo=None))
        assert isinstance(check_freshness(root, now=NOW), FreshBundle)

    def test_missing_manifest_is_unreadable_with_recording_hint(self, tmp_path: Path) -> None:
        freshness = check_freshness(tmp_path / "absent", now=NOW)
        assert isinstance(freshness, UnreadableBundle)
        assert MANIFEST_FILENAME in freshness.reason
        assert "E2E_FIXTURE_MODE=record" in freshness.reason

    def test_corrupt_manifest_is_unreadable(self, tmp_path: Path) -> None:
        root = tmp_path / "bundle"
        root.mkdir()
        (root / MANIFEST_FILENAME).write_text("{not json", encoding="utf-8")
        assert isinstance(check_freshness(root, now=NOW), UnreadableBundle)

    def test_unknown_format_version_is_unreadable(self, tmp_path: Path) -> None:
        root = tmp_path / "bundle"
        write_manifest(root, NOW, format_version=BUNDLE_FORMAT_VERSION + 1)
        freshness = check_freshness(root, now=NOW)
        assert isinstance(freshness, UnreadableBundle)
        assert f"format_version {BUNDLE_FORMAT_VERSION + 1}" in freshness.reason


class TestPrepareBundle:
    def test_fresh_directory_gets_a_fresh_manifest(self, tmp_path: Path) -> None:
        root = tmp_path / "bundle"
        prepared(root)
        freshness = check_freshness(root, now=datetime.now(timezone.utc))
        assert isinstance(freshness, FreshBundle)
        assert freshness.manifest.format_version == BUNDLE_FORMAT_VERSION
        assert freshness.manifest.harness_version

    def test_record_wipes_the_previous_bundle_instead_of_reading_it(self, tmp_path: Path) -> None:
        root = tmp_path / "bundle"
        prepared(root).record(
            test_key="old.py::test_old",
            request=plain_request("/stale"),
            response=RecordedResult(kind="unauthorized"),
        )
        assert any(entry.is_dir() for entry in root.iterdir())
        prepared(root)
        assert {entry.name for entry in root.iterdir()} == {MANIFEST_FILENAME}

    def test_refuses_to_wipe_a_directory_that_is_not_a_bundle(self, tmp_path: Path) -> None:
        root = tmp_path / "precious"
        root.mkdir()
        (root / "notes.txt").write_text("keep me", encoding="utf-8")
        outcome = prepare_bundle(root)
        assert isinstance(outcome, UnsafeBundleDir)
        assert MANIFEST_FILENAME in outcome.reason
        assert (root / "notes.txt").read_text(encoding="utf-8") == "keep me"

    def test_refuses_a_path_that_is_a_file(self, tmp_path: Path) -> None:
        target = tmp_path / "not-a-dir"
        target.write_text("x", encoding="utf-8")
        outcome = prepare_bundle(target)
        assert isinstance(outcome, UnsafeBundleDir)
        assert "not a directory" in outcome.reason


class TestSlugs:
    def test_slug_for_test_is_deterministic(self) -> None:
        key = "tests/e2e/suite/test_mod.py::TestX::test_case"
        assert slug_for_test(key) == slug_for_test(key)

    def test_same_tail_in_different_files_never_collides(self) -> None:
        first = slug_for_test("tests/e2e/a/test_a.py::test_case")
        second = slug_for_test("tests/e2e/b/test_b.py::test_case")
        assert first != second
        assert first.startswith("test_case-")
        assert second.startswith("test_case-")

    def test_interaction_filename_orders_and_slugs(self) -> None:
        request = RecordedRequest(method="post", path="/chat/completions", headers={})
        assert interaction_filename(3, request) == "0003-post-chat-completions.json"


class TestRecordAndLoad:
    def test_load_returns_interactions_in_recorded_order(self, tmp_path: Path) -> None:
        root = tmp_path / "bundle"
        recorder = prepared(root)
        key = "suite/test_mod.py::test_ordered"
        for path in ("/first", "/second", "/third"):
            recorder.record(
                test_key=key,
                request=plain_request(path),
                response=RecordedResult(kind="unauthorized"),
            )
        loaded = load_bundle(root)
        assert isinstance(loaded, LoadedBundle)
        assert [
            interaction.request.path for interaction in loaded.interactions[slug_for_test(key)]
        ] == ["/first", "/second", "/third"]

    def test_interactions_group_per_test(self, tmp_path: Path) -> None:
        root = tmp_path / "bundle"
        recorder = prepared(root)
        for key in ("suite/test_a.py::test_one", "suite/test_b.py::test_two"):
            recorder.record(
                test_key=key,
                request=plain_request(f"/{key[-3:]}"),
                response=RecordedResult(kind="unauthorized"),
            )
        loaded = load_bundle(root)
        assert isinstance(loaded, LoadedBundle)
        assert set(loaded.interactions) == {
            slug_for_test("suite/test_a.py::test_one"),
            slug_for_test("suite/test_b.py::test_two"),
        }
