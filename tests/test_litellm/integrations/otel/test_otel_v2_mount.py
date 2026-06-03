"""V2 entrypoint: the FastAPI instrumentation proxy_server mounts at app creation
(gated by LITELLM_OTEL_V2). The mount logic lives in
``litellm.integrations.otel.mount``; this exercises both that module's public
surface and the server-span + shared-provider behavior it produces.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

pytest.importorskip("opentelemetry")
pytest.importorskip("opentelemetry.instrumentation.fastapi")
fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)
from opentelemetry.trace import SpanKind  # noqa: E402

from litellm.integrations.opentelemetry_utils.gen_ai_semconv import (  # noqa: E402
    OTEL_SEMCONV_STABILITY_OPT_IN_ENV,
)
from litellm.integrations.otel.logger import OpenTelemetryV2  # noqa: E402
from litellm.integrations.otel.model.config import (  # noqa: E402
    OpenTelemetryV2Config,
    is_otel_v2_enabled,
)
from litellm.integrations.otel.mount import (  # noqa: E402
    PASSTHROUGH_PREFIXES,
    _ensure_http_semconv_opt_in,
    _passthrough_span_name_hook,
    instrument_fastapi_app,
)


class _FakeSpan:
    """Minimal recording span capturing what the hook writes."""

    def __init__(self, recording=True):
        self._recording = recording
        self.name = None
        self.attributes = {}

    def is_recording(self):
        return self._recording

    def update_name(self, name):
        self.name = name

    def set_attribute(self, key, value):
        self.attributes[key] = value


def _instrumented_app():
    """Mirror proxy_server's startup mount: a logger builds the shared provider,
    and the FastAPI instrumentor is attached to it."""
    app = fastapi.FastAPI()

    @app.get("/ping")
    def ping():
        return {"ok": True}

    logger = OpenTelemetryV2(config=OpenTelemetryV2Config(exporter="in_memory"))
    FastAPIInstrumentor.instrument_app(app, tracer_provider=logger._tracer_provider)
    return app, logger


def test_gate_toggles_with_env(monkeypatch):
    """The startup mount is guarded by this flag."""
    monkeypatch.delenv("LITELLM_OTEL_V2", raising=False)
    assert is_otel_v2_enabled() is False
    monkeypatch.setenv("LITELLM_OTEL_V2", "1")
    assert is_otel_v2_enabled() is True


def test_instrumented_app_emits_server_span():
    app, logger = _instrumented_app()
    exporter = InMemorySpanExporter()
    logger._tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    TestClient(app).get("/ping")

    server_spans = [
        s for s in exporter.get_finished_spans() if s.kind is SpanKind.SERVER
    ]
    assert server_spans, "FastAPI instrumentor should emit a SERVER span per request"
    attrs = server_spans[0].attributes or {}
    assert any("route" in k or "method" in k for k in attrs)


def test_logger_and_instrumentor_share_provider():
    """Gen-ai spans (logger) and server spans (instrumentor) write to one provider."""
    _, logger = _instrumented_app()
    assert logger._emitter._tracer is logger.tracer


def test_passthrough_hook_renames_catch_all_span():
    """A passthrough route gets its span renamed to the real request path."""
    span = _FakeSpan()
    _passthrough_span_name_hook(
        span, {"path": "/openai/v1/chat/completions", "method": "POST"}
    )
    assert span.name == "POST /openai/v1/chat/completions"
    assert span.attributes["http.route"] == "/openai/v1/chat/completions"


def test_passthrough_hook_leaves_non_passthrough_route_unchanged():
    """A normal route keeps its low-cardinality template name (hook no-ops)."""
    span = _FakeSpan()
    _passthrough_span_name_hook(span, {"path": "/v1/models", "method": "GET"})
    assert span.name is None
    assert "http.route" not in span.attributes


def test_passthrough_hook_ignores_non_recording_span():
    span = _FakeSpan(recording=False)
    _passthrough_span_name_hook(
        span, {"path": "/openai/v1/chat/completions", "method": "POST"}
    )
    assert span.name is None


def test_known_passthrough_prefixes_present():
    """Guard the prefix set against accidental edits."""
    assert {"openai", "anthropic", "vertex_ai", "bedrock"} <= PASSTHROUGH_PREFIXES


def test_instrument_fastapi_app_noop_when_gate_off(monkeypatch):
    """With the gate off the mount is a no-op — no instrumentation attached."""
    monkeypatch.delenv("LITELLM_OTEL_V2", raising=False)
    app = fastapi.FastAPI()
    instrument_fastapi_app(app)
    assert getattr(app, "_is_instrumented_by_opentelemetry", False) is False


def test_instrument_fastapi_app_attaches_when_gate_on(monkeypatch):
    """With the gate on the FastAPI app is instrumented for server spans."""
    monkeypatch.setenv("LITELLM_OTEL_V2", "1")
    app = fastapi.FastAPI()

    @app.get("/ping")
    def ping():
        return {"ok": True}

    instrument_fastapi_app(app)
    try:
        assert getattr(app, "_is_instrumented_by_opentelemetry", False) is True
    finally:
        FastAPIInstrumentor.uninstrument_app(app)


def test_ensure_http_semconv_opt_in_appends_when_unset(monkeypatch):
    monkeypatch.delenv(OTEL_SEMCONV_STABILITY_OPT_IN_ENV, raising=False)
    _ensure_http_semconv_opt_in()
    assert os.environ[OTEL_SEMCONV_STABILITY_OPT_IN_ENV] == "http"


def test_ensure_http_semconv_opt_in_preserves_other_categories(monkeypatch):
    """The flag is shared with the gen-ai opt-in, so appending must not clobber it."""
    monkeypatch.setenv(OTEL_SEMCONV_STABILITY_OPT_IN_ENV, "gen_ai_latest_experimental")
    _ensure_http_semconv_opt_in()
    tokens = os.environ[OTEL_SEMCONV_STABILITY_OPT_IN_ENV].split(",")
    assert "gen_ai_latest_experimental" in tokens
    assert "http" in tokens


@pytest.mark.parametrize("existing", ["http", "http/dup", "gen_ai,http/dup"])
def test_ensure_http_semconv_opt_in_respects_operator_http_mode(monkeypatch, existing):
    """An operator who already picked an http mode is left untouched (no double-add)."""
    monkeypatch.setenv(OTEL_SEMCONV_STABILITY_OPT_IN_ENV, existing)
    _ensure_http_semconv_opt_in()
    assert os.environ[OTEL_SEMCONV_STABILITY_OPT_IN_ENV] == existing


def _reset_semconv_stability():
    """Snapshot and clear the once-initialized HTTP semconv mode so an
    ``instrument_app`` call re-reads the env var as a fresh process would.
    Returns a restore callable for teardown."""
    from opentelemetry.instrumentation import _semconv

    cls = _semconv._OpenTelemetrySemanticConventionStability
    prev_initialized = cls._initialized
    prev_mapping = dict(cls._OTEL_SEMCONV_STABILITY_SIGNAL_MAPPING)

    cls._initialized = False
    cls._OTEL_SEMCONV_STABILITY_SIGNAL_MAPPING = {}

    def restore():
        cls._initialized = prev_initialized
        cls._OTEL_SEMCONV_STABILITY_SIGNAL_MAPPING = prev_mapping

    return restore


def _server_span_for_status(app, exporter, path):
    TestClient(app, raise_server_exceptions=False).get(path)
    server_spans = [
        s for s in exporter.get_finished_spans() if s.kind is SpanKind.SERVER
    ]
    assert server_spans, "expected a SERVER span for the failing request"
    return server_spans[0].attributes or {}


def _failing_app_with_exporter():
    app = fastapi.FastAPI()

    @app.get("/boom")
    def boom():
        return fastapi.Response(status_code=503)

    from opentelemetry.sdk.trace import TracerProvider

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return app, provider, exporter


def test_server_span_carries_error_type_on_5xx(monkeypatch):
    """The fix: with the http semconv opt-in active (which the mount applies),
    a 5xx server span carries the OTel ``error.type`` attribute."""
    monkeypatch.delenv(OTEL_SEMCONV_STABILITY_OPT_IN_ENV, raising=False)
    _ensure_http_semconv_opt_in()
    restore = _reset_semconv_stability()
    app, provider, exporter = _failing_app_with_exporter()
    FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)
    try:
        attrs = _server_span_for_status(app, exporter, "/boom")
        assert attrs.get("error.type") == "503"
        assert attrs.get("http.response.status_code") == 503
    finally:
        FastAPIInstrumentor.uninstrument_app(app)
        restore()


def test_server_span_missing_error_type_without_opt_in(monkeypatch):
    """Negative control: in the default semconv mode the same 5xx carries no
    ``error.type`` and only the legacy ``http.status_code`` - exactly the gap
    the opt-in closes."""
    monkeypatch.delenv(OTEL_SEMCONV_STABILITY_OPT_IN_ENV, raising=False)
    restore = _reset_semconv_stability()
    app, provider, exporter = _failing_app_with_exporter()
    FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)
    try:
        attrs = _server_span_for_status(app, exporter, "/boom")
        assert "error.type" not in attrs
        assert attrs.get("http.status_code") == 503
    finally:
        FastAPIInstrumentor.uninstrument_app(app)
        restore()
