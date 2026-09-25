"""V2 entrypoint: the FastAPI instrumentation proxy_server mounts at app creation
(gated by LITELLM_OTEL_V2). The mount logic lives in
``litellm.integrations.otel.mount``; this exercises both that module's public
surface and the server-span + shared-provider behavior it produces.
"""


from datetime import datetime, timezone

import pytest


pytest.importorskip("opentelemetry")
pytest.importorskip("opentelemetry.instrumentation.fastapi")
fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)
from opentelemetry import trace  # noqa: E402
from opentelemetry.trace import SpanKind  # noqa: E402

from litellm.integrations.otel.model.config import (  # noqa: E402
    OpenTelemetryV2Config,
    is_otel_v2_enabled,
)
from litellm.integrations.otel.logger import OpenTelemetryV2  # noqa: E402
from litellm.integrations.otel.mount import (  # noqa: E402
    PASSTHROUGH_PREFIXES,
    _passthrough_span_name_hook,
    instrument_fastapi_app,
)
from litellm.integrations.otel.plumbing.context import (  # noqa: E402
    request_root_http_route,
    set_request_root_span,
)


@pytest.fixture(autouse=True)
def _reset_request_root_span():
    """Clear the root-span anchor around every test. Production gets a fresh
    contextvar copy per request task; the test process shares one context."""
    from litellm.integrations.otel.plumbing import context as _otel_context

    _otel_context._request_root_span.set(None)
    _otel_context._mcp_message_transport_span.set(None)
    yield
    _otel_context._request_root_span.set(None)
    _otel_context._mcp_message_transport_span.set(None)


@pytest.fixture(autouse=True)
def _clear_otel_v2_flag_cache():
    is_otel_v2_enabled.cache_clear()
    yield
    is_otel_v2_enabled.cache_clear()


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
    is_otel_v2_enabled.cache_clear()
    assert is_otel_v2_enabled() is False
    monkeypatch.setenv("LITELLM_OTEL_V2", "1")
    is_otel_v2_enabled.cache_clear()
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


def test_llm_span_route_is_read_off_the_server_span(monkeypatch):
    """``request_root_http_route`` answers with the SERVER span's own ``http.route``.

    Driven through ``instrument_fastapi_app`` and the same
    ``create_litellm_proxy_request_started_span`` call the proxy makes per request,
    so breaking either the mount or the anchor capture fails this."""
    monkeypatch.setenv("LITELLM_OTEL_V2", "1")
    is_otel_v2_enabled.cache_clear()
    app = fastapi.FastAPI()
    seen = {}

    def _anchor_then_read(key):
        logger.create_litellm_proxy_request_started_span(start_time=datetime.now(timezone.utc), headers=None)
        seen[key] = request_root_http_route()

    @app.post("/engines/{model:path}/chat/completions")
    async def engines(model: str):
        _anchor_then_read("templated")
        return {}

    @app.post("/openai/{endpoint:path}")
    async def openai_passthrough(endpoint: str):
        _anchor_then_read("passthrough")
        return {}

    logger = OpenTelemetryV2(config=OpenTelemetryV2Config(exporter="in_memory"))
    exporter = InMemorySpanExporter()
    logger._tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    # instrument_fastapi_app passes no provider, so it binds to the OTel global the
    # way the proxy does once proxy_startup_event publishes one. set_tracer_provider
    # is a once-per-process door, so place it directly and let monkeypatch undo it.
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", logger._tracer_provider)
    instrument_fastapi_app(app)

    client = TestClient(app)
    client.post("/engines/gpt-4o-mini/chat/completions")
    client.post("/openai/v1/responses/resp_abc123")

    routes = {
        (s.attributes or {})["http.route"] for s in exporter.get_finished_spans() if s.kind is SpanKind.SERVER
    }
    # a parameterized route keeps its template; the passthrough hook rewrote the
    # catch-all to the literal path, and both spans have to follow their own span
    assert routes == {"/engines/{model:path}/chat/completions", "/openai/v1/responses/resp_abc123"}
    assert seen["templated"] == "/engines/{model:path}/chat/completions"
    assert seen["passthrough"] == "/openai/v1/responses/resp_abc123"


def test_server_span_route_survives_the_span_ending():
    """The LLM span closes in an async callback that can run after the server span
    has ended, so the attribute has to still be readable then."""
    from opentelemetry.sdk.trace import TracerProvider

    span = TracerProvider().get_tracer("t").start_span("POST /v1/responses/{response_id}")
    span.set_attribute("http.route", "/v1/responses/{response_id}")
    set_request_root_span(span)
    span.end()

    assert request_root_http_route() == "/v1/responses/{response_id}"


def test_no_server_span_means_no_route():
    """An SDK call has no anchored server span, so the attribute is omitted rather
    than reported as empty."""
    assert request_root_http_route() is None


def test_blank_route_on_the_server_span_is_omitted():
    """An excluded or unmatched path leaves the server span without a usable route.
    Report nothing rather than a span attribute whose value is the empty string."""
    from opentelemetry.sdk.trace import TracerProvider

    span = TracerProvider().get_tracer("t").start_span("GET")
    span.set_attribute("http.route", "")
    set_request_root_span(span)

    assert request_root_http_route() is None


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
