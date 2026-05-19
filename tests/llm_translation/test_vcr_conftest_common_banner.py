from __future__ import annotations

import os
import sys
from io import StringIO

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tests._vcr_conftest_common import (  # noqa: E402
    emit_cassette_cache_session_banner,
)
from tests._vcr_redis_persister import (  # noqa: E402
    _cache_health,
    reset_cassette_cache_health,
)


class _FakeTerminalReporter:
    def __init__(self) -> None:
        self.buf = StringIO()

    def write_sep(self, sep, title="", **kwargs):
        if title:
            self.buf.write(f"{sep * 5} {title} {sep * 5}\n")
        else:
            self.buf.write(f"{sep * 60}\n")

    def write_line(self, line):
        self.buf.write(f"{line}\n")

    @property
    def output(self) -> str:
        return self.buf.getvalue()


@pytest.fixture
def health_reset():
    reset_cassette_cache_health()
    yield
    reset_cassette_cache_health()


@pytest.fixture
def vcr_enabled(monkeypatch):
    """Make :func:`vcr_disabled` return False so the banner emits."""
    monkeypatch.setenv("CASSETTE_REDIS_URL", "redis://stub")
    monkeypatch.delenv("LITELLM_VCR_DISABLE", raising=False)
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)


@pytest.fixture
def patch_capacity_snapshot(monkeypatch):
    """Stub :func:`cassette_cache_capacity_snapshot` so we don't try to
    open a real Redis connection. The fixture returns a setter that
    each test uses to inject the snapshot it wants."""
    state = {"snapshot": None}

    def _stub():
        return state["snapshot"]

    import tests._vcr_conftest_common as common

    monkeypatch.setattr(common, "cassette_cache_capacity_snapshot", _stub)

    def _set(snapshot):
        state["snapshot"] = snapshot

    return _set


def test_banner_silent_when_no_failures_and_capacity_healthy(
    health_reset, vcr_enabled, patch_capacity_snapshot
):
    patch_capacity_snapshot(
        {"used_memory_bytes": 100, "maxmemory_bytes": 1000, "used_pct": 10.0}
    )
    reporter = _FakeTerminalReporter()

    emit_cassette_cache_session_banner(reporter)

    assert reporter.output == ""


def test_banner_red_section_when_save_failures_recorded(
    health_reset, vcr_enabled, patch_capacity_snapshot
):
    _cache_health["save_failures"] = 3
    _cache_health["save_failure_last_error"] = (
        "OutOfMemoryError: command not allowed when used memory > 'maxmemory'."
    )
    patch_capacity_snapshot(
        {"used_memory_bytes": 990, "maxmemory_bytes": 1000, "used_pct": 99.0}
    )
    reporter = _FakeTerminalReporter()

    emit_cassette_cache_session_banner(reporter)

    out = reporter.output
    assert "VCR CASSETTE CACHE DEGRADED" in out
    assert "3 cassette save failure(s)" in out
    assert "OutOfMemoryError" in out
    assert "99.0% of maxmemory" in out


def test_banner_red_section_when_load_failures_recorded(
    health_reset, vcr_enabled, patch_capacity_snapshot
):
    _cache_health["load_failures"] = 2
    _cache_health["load_failure_last_error"] = "ConnectionError: simulated outage"
    patch_capacity_snapshot(None)
    reporter = _FakeTerminalReporter()

    emit_cassette_cache_session_banner(reporter)

    out = reporter.output
    assert "VCR CASSETTE CACHE DEGRADED" in out
    assert "2 cassette load failure(s)" in out
    assert "ConnectionError" in out


def test_banner_yellow_high_water_when_no_failures_but_near_capacity(
    health_reset, vcr_enabled, patch_capacity_snapshot
):
    patch_capacity_snapshot(
        {"used_memory_bytes": 900, "maxmemory_bytes": 1000, "used_pct": 90.0}
    )
    reporter = _FakeTerminalReporter()

    emit_cassette_cache_session_banner(reporter)

    out = reporter.output
    assert "VCR CASSETTE CACHE NEAR CAPACITY" in out
    assert "90.0% of maxmemory" in out
    assert "VCR CASSETTE CACHE DEGRADED" not in out


def test_banner_silent_when_below_high_water_and_no_failures(
    health_reset, vcr_enabled, patch_capacity_snapshot
):
    patch_capacity_snapshot(
        {"used_memory_bytes": 800, "maxmemory_bytes": 1000, "used_pct": 80.0}
    )
    reporter = _FakeTerminalReporter()

    emit_cassette_cache_session_banner(reporter)

    assert reporter.output == ""


def test_banner_silent_when_vcr_disabled(
    monkeypatch, health_reset, patch_capacity_snapshot
):
    monkeypatch.delenv("CASSETTE_REDIS_URL", raising=False)
    _cache_health["save_failures"] = 5
    _cache_health["save_failure_last_error"] = "OutOfMemoryError: foo"
    patch_capacity_snapshot(
        {"used_memory_bytes": 999, "maxmemory_bytes": 1000, "used_pct": 99.9}
    )
    reporter = _FakeTerminalReporter()

    emit_cassette_cache_session_banner(reporter)

    assert reporter.output == ""


def test_banner_silent_on_xdist_worker(
    monkeypatch, vcr_enabled, health_reset, patch_capacity_snapshot
):
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw3")
    _cache_health["save_failures"] = 1
    _cache_health["save_failure_last_error"] = "OutOfMemoryError: bar"
    patch_capacity_snapshot(
        {"used_memory_bytes": 999, "maxmemory_bytes": 1000, "used_pct": 99.9}
    )
    reporter = _FakeTerminalReporter()

    emit_cassette_cache_session_banner(reporter)

    assert reporter.output == ""
