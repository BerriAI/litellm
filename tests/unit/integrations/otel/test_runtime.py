"""Regression tests for the SDK-free OTel runtime shim.

The proxy auth hot path calls ``phase_span`` and ``seed_request_identity`` on
every request. These wrappers resolve the SDK-backed implementations with a
lazy import. CPython never caches a failed import, so before memoization an
absent OTel SDK made every request re-scan ``sys.path`` and contend on the
import lock. These tests pin the import to a single resolution.
"""

import builtins

import pytest

import litellm.integrations.otel.runtime as runtime


def test_logger_not_reimported_after_first_resolution(monkeypatch):
    runtime._otel_runtime.cache_clear()

    counts = {"n": 0}
    real_import = builtins.__import__

    def counting_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "litellm.integrations.otel" and fromlist and "logger" in fromlist:
            counts["n"] += 1
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", counting_import)

    with runtime.phase_span("auth /v1/chat/completions"):
        pass
    after_first = counts["n"]

    for _ in range(49):
        with runtime.phase_span("auth /v1/chat/completions"):
            pass

    assert counts["n"] == after_first, (
        f"otel.logger re-imported {counts['n'] - after_first} times after the first "
        "resolution; it must be memoized so it does not re-scan sys.path per request"
    )

    runtime._otel_runtime.cache_clear()


def test_resolution_is_memoized():
    runtime._otel_runtime.cache_clear()

    for _ in range(25):
        with runtime.phase_span("p"):
            pass

    info = runtime._otel_runtime.cache_info()
    assert info.misses == 1
    assert info.hits >= 24

    runtime._otel_runtime.cache_clear()


def test_wrappers_no_op_when_runtime_absent(monkeypatch):
    monkeypatch.setattr(runtime, "_otel_runtime", lambda: None)

    with runtime.phase_span("auth") as span:
        assert span is None

    assert runtime.seed_request_identity({"token": "sk-x"}, model="gpt-4o") is None


def test_phase_span_forwards_redact_content_to_registered_logger(monkeypatch):
    """``redact_content`` must reach the v2 logger through the runtime shim: an
    exception raised inside the forwarded span carries only the redaction marker,
    even with the global flag off."""
    pytest.importorskip("opentelemetry")

    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    import litellm
    import litellm.integrations.otel.logger as otel_logger
    from litellm.integrations.otel import OpenTelemetryV2Config
    from litellm.integrations.otel.logger import OpenTelemetryV2
    from litellm.integrations.otel.plumbing import providers

    cfg = OpenTelemetryV2Config(exporter="in_memory", legacy_compat=True, baggage_team_metadata_keys=[])
    exporter = InMemorySpanExporter()
    logger = OpenTelemetryV2(config=cfg, tracer_provider=providers.build_tracer_provider(cfg, exporter=exporter))
    monkeypatch.setattr(otel_logger, "_registered_v2_logger", lambda: logger)

    litellm.turn_off_message_logging = False
    try:
        secret = "secret-prompt-marker"
        with pytest.raises(RuntimeError):
            with runtime.phase_span("auth", redact_content=True):
                raise RuntimeError(f"auth exploded: {secret}")
        (span,) = exporter.get_finished_spans()
        assert secret not in str(dict(span.attributes or {}))
        assert secret not in str([dict(e.attributes or {}) for e in span.events])
        assert secret not in str(span.status.description or "")
    finally:
        litellm.turn_off_message_logging = False
