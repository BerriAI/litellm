"""V2 entrypoint: the FastAPI instrumentation proxy_server mounts at app creation
(gated by LITELLM_OTEL_V2). The mount logic lives in
``litellm.integrations.otel.mount``; this exercises both that module's public
surface and the server-span + shared-provider behavior it produces.
"""


import contextvars
from datetime import datetime, timezone

import pytest


pytest.importorskip("opentelemetry")
pytest.importorskip("opentelemetry.instrumentation.fastapi")
fastapi = pytest.importorskip("fastapi")

from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from starlette.routing import Route  # noqa: E402
import httpx  # noqa: E402
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor  # noqa: E402
from opentelemetry.instrumentation.propagators import (  # noqa: E402
    ResponsePropagator,
    get_global_response_propagator,
    set_global_response_propagator,
)
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)
from opentelemetry import trace  # noqa: E402
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF  # noqa: E402
from opentelemetry.trace import SpanKind  # noqa: E402

from litellm.constants import LITELLM_UI_ALLOW_HEADERS  # noqa: E402

from litellm.integrations.otel.model.config import (  # noqa: E402
    OpenTelemetryV2Config,
    is_otel_v2_enabled,
)
from litellm.integrations.otel.logger import OpenTelemetryV2  # noqa: E402
from litellm.integrations.otel.mount import (  # noqa: E402
    PASSTHROUGH_PREFIXES,
    _passthrough_span_name_hook,
    instrument_fastapi_app,
    trace_response_expose_headers,
    trace_response_headers,
)
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (  # noqa: E402
    HttpPassThroughEndpointHelpers,
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
def _reset_response_propagator():
    set_global_response_propagator(None)
    yield
    set_global_response_propagator(None)


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


def test_span_hook_does_not_break_requests_when_the_span_provider_fails():
    class _FailingSpan:
        def is_recording(self) -> bool:
            raise RuntimeError("span provider unavailable")

        def update_name(self, name: str) -> None:
            raise AssertionError(name)

        def set_attribute(self, key: str, value: str) -> None:
            raise AssertionError((key, value))

    assert _passthrough_span_name_hook(_FailingSpan(), {}) is None


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


_UPSTREAM_TRACERESPONSE = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"


class _TupleHeadersApp:
    async def __call__(self, scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": ((b"x-litellm-call-id", b"call-tuple"),)})
        await send({"type": "http.response.body", "body": b"ok"})


def _traced_app(monkeypatch, tracer_provider):
    from litellm.proxy.proxy_server import otel_unhandled_exception_handler

    monkeypatch.setenv("LITELLM_OTEL_V2", "1")
    is_otel_v2_enabled.cache_clear()
    app = fastapi.FastAPI()
    app.add_exception_handler(Exception, otel_unhandled_exception_handler)

    @app.get("/success")
    async def success():
        return fastapi.Response(headers={"x-litellm-call-id": "call-success"})

    @app.get("/failure")
    async def failure():
        raise fastapi.HTTPException(status_code=400, headers={"x-litellm-call-id": "call-failure"})

    async def chunks():
        yield b"data: first\n\n"
        yield b"data: [DONE]\n\n"

    @app.get("/stream")
    async def stream():
        return StreamingResponse(chunks(), headers={"x-litellm-call-id": "call-stream"})

    async def relay_upstream():
        upstream = httpx.Headers({"traceresponse": _UPSTREAM_TRACERESPONSE})
        return fastapi.Response(headers=HttpPassThroughEndpointHelpers.get_response_headers(upstream, "call-relayed"))

    app.add_api_route("/relayed", relay_upstream)
    app.add_api_route("/relayed/health", relay_upstream)

    @app.get("/relayed-untraced")
    async def relay_untraced_upstream():
        upstream = httpx.Headers({"access-control-expose-headers": "x-upstream-request-id"})
        return fastapi.Response(headers=HttpPassThroughEndpointHelpers.get_response_headers(upstream, "call-untraced"))

    app.router.routes.append(Route("/tuple-headers", _TupleHeadersApp()))

    @app.get("/crash")
    async def crash(request: fastapi.Request):
        request.state.parent_otel_span = trace.get_current_span()
        raise RuntimeError("unhandled")

    @app.websocket("/ws")
    async def websocket_route(websocket: fastapi.WebSocket):
        await websocket.accept()
        await websocket.send_text("ok")
        await websocket.close()

    instrument_fastapi_app(app, tracer_provider=tracer_provider)
    return app


def _exporting_provider():
    logger = OpenTelemetryV2(config=OpenTelemetryV2Config(exporter="in_memory"))
    exporter = InMemorySpanExporter()
    logger._tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    return logger._tracer_provider, exporter


def _server_spans_by_route(exporter):
    return {(s.attributes or {})["http.route"]: s for s in exporter.get_finished_spans() if s.kind is SpanKind.SERVER}


def test_every_traced_response_names_its_own_server_span(monkeypatch):
    provider, exporter = _exporting_provider()
    client = TestClient(_traced_app(monkeypatch, provider), raise_server_exceptions=False)

    responses = {path: client.get(path) for path in ("/success", "/failure", "/stream", "/tuple-headers", "/crash")}

    server_spans = _server_spans_by_route(exporter)
    assert responses["/failure"].status_code == 400
    assert responses["/tuple-headers"].status_code == 200
    assert responses["/crash"].status_code == 500
    assert responses["/stream"].text == "data: first\n\ndata: [DONE]\n\n"
    for path, response in responses.items():
        context = server_spans[path].context
        assert response.headers.get_list("traceresponse") == [f"00-{context.trace_id:032x}-{context.span_id:016x}-01"]


@pytest.mark.parametrize("path", ["/relayed", "/relayed/health"])
def test_a_relayed_upstream_traceresponse_is_returned_once_and_unchanged(monkeypatch, path):
    provider, _ = _exporting_provider()

    response = TestClient(_traced_app(monkeypatch, provider)).get(path)

    assert response.headers.get_list("traceresponse") == [_UPSTREAM_TRACERESPONSE]


@pytest.mark.parametrize(
    ("caller_headers", "linked"),
    [({}, True), ({"traceparent": f"00-{_UPSTREAM_TRACERESPONSE.split('-')[1]}-00f067aa0ba902b7-01"}, False)],
)
def test_a_relayed_traceresponse_from_a_restarted_trace_is_linked(monkeypatch, caller_headers, linked):
    provider, exporter = _exporting_provider()

    TestClient(_traced_app(monkeypatch, provider)).get("/relayed", headers=caller_headers)

    links = _server_spans_by_route(exporter)["/relayed"].links
    upstream_ids = [f"00-{link.context.trace_id:032x}-{link.context.span_id:016x}-01" for link in links]
    assert upstream_ids == ([_UPSTREAM_TRACERESPONSE] if linked else [])


def test_traceresponse_leaves_access_control_expose_headers_to_cors(monkeypatch):
    provider, _ = _exporting_provider()
    client = TestClient(_traced_app(monkeypatch, provider), raise_server_exceptions=False)

    responses = [client.get(path) for path in ("/success", "/crash", "/relayed-untraced")]

    assert all(response.headers.get_list("traceresponse") for response in responses)
    exposed = [response.headers.get_list("access-control-expose-headers") for response in responses]
    assert exposed == [[], [], ["x-upstream-request-id"]]


def test_server_span_carries_the_call_id_its_response_returns(monkeypatch):
    provider, exporter = _exporting_provider()
    client = TestClient(_traced_app(monkeypatch, provider))

    responses = {path: client.get(path) for path in ("/success", "/failure", "/stream", "/relayed", "/tuple-headers")}

    server_spans = _server_spans_by_route(exporter)
    for path, response in responses.items():
        assert (server_spans[path].attributes or {})["litellm.call_id"] == response.headers["x-litellm-call-id"]


def test_traceresponse_keeps_the_caller_trace(monkeypatch):
    provider, _ = _exporting_provider()
    caller_trace_id = "4bf92f3577b34da6a3ce929d0e0e4736"

    response = TestClient(_traced_app(monkeypatch, provider)).get(
        "/success", headers={"traceparent": f"00-{caller_trace_id}-00f067aa0ba902b7-01"}
    )

    assert response.headers["traceresponse"].split("-")[1] == caller_trace_id


def test_unsampled_requests_get_no_traceresponse(monkeypatch):
    client = TestClient(_traced_app(monkeypatch, TracerProvider(sampler=ALWAYS_OFF)), raise_server_exceptions=False)

    responses = [client.get(path) for path in ("/success", "/crash")]

    assert [response.headers.get_list("traceresponse") for response in responses] == [[], []]


def test_websocket_handshakes_carry_no_traceresponse(monkeypatch):
    provider, _ = _exporting_provider()

    with TestClient(_traced_app(monkeypatch, provider)).websocket_connect("/ws") as websocket:
        assert websocket.receive_text() == "ok"
        handshake_header_names = [name for name, _ in websocket.extra_headers or ()]

    assert b"traceresponse" not in handshake_header_names


def test_gate_off_responses_carry_no_traceresponse(monkeypatch):
    monkeypatch.delenv("LITELLM_OTEL_V2", raising=False)
    app = fastapi.FastAPI()

    @app.get("/ping")
    def ping():
        return {"ok": True}

    instrument_fastapi_app(app)

    assert "traceresponse" not in TestClient(app).get("/ping").headers


def test_failed_instrumentation_registers_no_response_propagator(monkeypatch):
    monkeypatch.setenv("LITELLM_OTEL_V2", "1")
    app = fastapi.FastAPI()

    with TestClient(app):
        instrument_fastapi_app(app)

    assert get_global_response_propagator() is None


def test_an_operator_response_propagator_is_kept(monkeypatch):
    class _OperatorPropagator(ResponsePropagator):
        def inject(self, carrier, context=None, setter=None):
            setter.set(carrier, "x-operator-trace", "kept")

    set_global_response_propagator(_OperatorPropagator())
    provider, _ = _exporting_provider()

    response = TestClient(_traced_app(monkeypatch, provider)).get("/relayed")

    assert response.headers["x-operator-trace"] == "kept"
    assert response.headers["traceresponse"] == _UPSTREAM_TRACERESPONSE


def test_browsers_can_read_traceresponse_through_cors(monkeypatch):
    provider, _ = _exporting_provider()
    app = _traced_app(monkeypatch, provider)
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], expose_headers=[*LITELLM_UI_ALLOW_HEADERS, *trace_response_expose_headers()]
    )

    response = TestClient(app).get("/success", headers={"origin": "https://ui.example.com"})

    exposed = {name.strip() for name in response.headers["access-control-expose-headers"].split(",")}
    assert "traceresponse" in exposed
    assert response.headers["traceresponse"]


def test_cors_leaves_traceresponse_unexposed_with_the_gate_off(monkeypatch):
    monkeypatch.delenv("LITELLM_OTEL_V2", raising=False)
    is_otel_v2_enabled.cache_clear()

    assert "traceresponse" not in [*LITELLM_UI_ALLOW_HEADERS, *trace_response_expose_headers()]


def test_other_instrumented_apps_in_the_process_get_no_traceresponse(monkeypatch):
    provider, _ = _exporting_provider()
    litellm_client = TestClient(_traced_app(monkeypatch, provider))
    operator_app = fastapi.FastAPI()

    @operator_app.get("/operator")
    async def operator_route():
        return {"ok": True}

    FastAPIInstrumentor.instrument_app(operator_app, tracer_provider=provider)
    try:
        operator_response = TestClient(operator_app).get("/operator")
        litellm_response = litellm_client.get("/success")
    finally:
        FastAPIInstrumentor.uninstrument_app(operator_app)

    assert "traceresponse" not in operator_response.headers
    assert litellm_response.headers["traceresponse"]


def test_the_unhandled_error_helper_names_the_span_it_is_given_in_any_context(monkeypatch):
    provider, _ = _exporting_provider()
    _traced_app(monkeypatch, provider)
    span = provider.get_tracer(__name__).start_span("POST /v1/chat/completions", kind=SpanKind.SERVER)
    context = span.get_span_context()

    headers = contextvars.Context().run(trace_response_headers, span)

    assert headers["traceresponse"] == f"00-{context.trace_id:032x}-{context.span_id:016x}-01"
