from __future__ import annotations

import os
import sys
from io import StringIO

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tests._vcr_conftest_common import (  # noqa: E402
    VCR_DIAG_EMIT_MAX_LINES,
    emit_cassette_cache_session_banner,
    emit_vcr_diagnostic_log,
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


# ---------------------------------------------------------------------------
# Diagnostic-log dedup + cap. CircleCI truncates step output to the last
# ~400 KB; an unbounded diagnostic dump pushes the VCR classification summary
# out of the retrievable window, so the dump must dedupe and cap.
# ---------------------------------------------------------------------------


def test_diagnostic_log_dedupes_repeated_blocks(tmp_path, monkeypatch):
    monkeypatch.setenv("LITELLM_VCR_DIAG_DIR", str(tmp_path))
    (tmp_path / "123.log").write_text(
        "\n".join(["[vcr-key-fingerprint-matcher] differ"] * 40 + ["unique line"]),
        encoding="utf-8",
    )
    reporter = _FakeTerminalReporter()

    emit_vcr_diagnostic_log(reporter)

    out = reporter.output
    # The repeated block collapses to a single line with an occurrence count.
    assert out.count("[vcr-key-fingerprint-matcher] differ") == 1
    assert "(x40)" in out
    assert "unique line" in out


def test_diagnostic_log_caps_unique_lines(tmp_path, monkeypatch):
    monkeypatch.setenv("LITELLM_VCR_DIAG_DIR", str(tmp_path))
    total = VCR_DIAG_EMIT_MAX_LINES + 50
    (tmp_path / "123.log").write_text(
        "\n".join(f"unique-diagnostic-{i}" for i in range(total)), encoding="utf-8"
    )
    reporter = _FakeTerminalReporter()

    emit_vcr_diagnostic_log(reporter)

    out = reporter.output
    emitted = sum(1 for ln in out.splitlines() if ln.startswith("unique-diagnostic-"))
    assert emitted == VCR_DIAG_EMIT_MAX_LINES
    assert "more unique diagnostic line(s) suppressed" in out


def test_diagnostic_log_silent_when_no_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("LITELLM_VCR_DIAG_DIR", str(tmp_path / "does-not-exist"))
    reporter = _FakeTerminalReporter()

    emit_vcr_diagnostic_log(reporter)

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


# ---------------------------------------------------------------------------
# Telemetry-leak suppression. Several modules set ``litellm.success_callback``
# at import time, so observability logging is globally enabled and an async
# flush can land in an unrelated test's VCR window and be saved as a spurious
# MISS:RECORDED episode. ``_should_drop_telemetry_record`` refuses to record a
# telemetry call for a non-telemetry test (it passes through live instead),
# while tests that actually assert on telemetry keep recording.
# ---------------------------------------------------------------------------


class _FakeRequest:
    def __init__(
        self, host, scheme="https", method="POST", path="/api/public/ingestion"
    ):
        self.host = host
        self.scheme = scheme
        self.uri = f"{scheme}://{host}{path}"
        self.headers = {}
        self.method = method
        self.body = b"{}"


@pytest.fixture
def current_test(monkeypatch):
    """Set the module-global current-test nodeid the suppressor reads."""
    import tests._vcr_conftest_common as common

    def _set(nodeid):
        monkeypatch.setattr(common, "_current_test_nodeid", nodeid)

    return _set


@pytest.mark.parametrize(
    "nodeid,host,method,expected_drop",
    [
        # Non-telemetry test: incidental telemetry leak is dropped (not recorded).
        (
            "tests/local_testing/test_lowest_latency_routing.py::test_lowest_latency_routing_buffer[1]",
            "us.cloud.langfuse.com",
            "POST",
            True,
        ),
        (
            "tests/local_testing/test_function_call_parsing.py::test_parse",
            "us.cloud.langfuse.com",
            "POST",
            True,
        ),
        (
            "tests/llm_translation/test_x.py::test_y",
            "otlp.arize.com",
            "POST",
            True,
        ),
        # Non-telemetry host on a non-telemetry test: never dropped.
        (
            "tests/local_testing/test_lowest_latency_routing.py::test_lowest_latency_routing_buffer[1]",
            "api.openai.com",
            "POST",
            False,
        ),
        # Telemetry EXPORT POSTs are fire-and-forget and dropped even for
        # telemetry-named tests: litellm's background flush makes them rotate
        # into a later telemetry test's window as a phantom MISS:RECORDED. The
        # e2e suite mocks the export client and asserts on the mock; read-back
        # tests assert on a GET — neither needs the recorded export POST.
        (
            "tests/local_testing/test_alangfuse.py::test_langfuse_logging",
            "us.cloud.langfuse.com",
            "POST",
            True,
        ),
        (
            "tests/logging_callback_tests/test_langfuse_e2e_test.py::test_e2e",
            "us.cloud.langfuse.com",
            "POST",
            True,
        ),
        (
            "tests/logging_callback_tests/test_dynamic_otel_keys.py::test_keys",
            "otlp.arize.com",
            "POST",
            True,
        ),
        # Read-back GETs that telemetry tests assert on are kept (matched by
        # method, so the export-POST drop does not touch them).
        (
            "tests/local_testing/test_alangfuse.py::test_langfuse_logging",
            "us.cloud.langfuse.com",
            "GET",
            False,
        ),
        # ...but a read-back GET on a NON-telemetry test is still incidental.
        (
            "tests/local_testing/test_function_call_parsing.py::test_parse",
            "us.cloud.langfuse.com",
            "GET",
            True,
        ),
        # The pass-through proxy test forwards a client POST to Langfuse
        # ingestion and asserts the replayed 207 — its export POST is kept.
        (
            "tests/local_testing/test_pass_through_endpoints.py::test_aaapass_through_endpoint_pass_through_keys_langfuse[False-0-207]",
            "us.cloud.langfuse.com",
            "POST",
            False,
        ),
    ],
)
def test_should_drop_telemetry_record(
    current_test, nodeid, host, method, expected_drop
):
    import tests._vcr_conftest_common as common

    current_test(nodeid)
    req = _FakeRequest(host, method=method)
    assert common._should_drop_telemetry_record(req) is expected_drop


def test_drop_is_suppressed_while_loading_stored_episodes(current_test):
    """During ``Cassette._load`` the drop MUST be inert.

    vcrpy replays each stored interaction through ``Cassette.append`` →
    ``before_record_request``; a ``None`` there silently drops the stored
    episode. If the telemetry drop fired on load, an already-recorded
    telemetry episode would be deleted the instant a non-telemetry-named
    test loaded it, forcing an endless live re-record (a phantom
    MISS:RECORDED on a cassette that was present in Redis). The drop must
    only stop *new* incidental recordings, never filter the cassette on read.
    """
    import tests._vcr_conftest_common as common

    # A non-telemetry test loading a stored Langfuse episode: dropped on
    # record, but must be KEPT while loading.
    current_test("tests/local_testing/test_lowest_latency_routing.py::test_buf")
    req = _FakeRequest("us.cloud.langfuse.com")

    assert common._should_drop_telemetry_record(req) is True  # record path

    common._vcr_load_guard.active = True
    try:
        assert common._vcr_load_in_progress() is True
        assert common._should_drop_telemetry_record(req) is False  # load path
    finally:
        common._vcr_load_guard.active = False
    assert common._should_drop_telemetry_record(req) is True


def test_load_guard_patch_is_idempotent():
    import vcr.cassette as cassette_mod

    import tests._vcr_conftest_common as common

    common.patch_vcrpy_cassette_load_guard()
    first = cassette_mod.Cassette._load
    common.patch_vcrpy_cassette_load_guard()
    assert cassette_mod.Cassette._load is first
    assert getattr(cassette_mod.Cassette._load, "_litellm_load_guarded", False)
