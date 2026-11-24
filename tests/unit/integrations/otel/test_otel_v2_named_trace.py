"""Proxy requests that name a trace (``trace_id`` in ``metadata`` / ``litellm_metadata``) share that OTel trace on V2."""

from hashlib import sha256

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("opentelemetry.instrumentation.fastapi")

from fastapi.testclient import TestClient  # noqa: E402  # optional dependencies are skipped first
from opentelemetry import trace  # noqa: E402
from opentelemetry.instrumentation.propagators import set_global_response_propagator  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402
from opentelemetry.trace import SpanKind  # noqa: E402

from litellm.integrations.otel.logger import OpenTelemetryV2  # noqa: E402
from litellm.integrations.otel.model.config import OpenTelemetryV2Config, is_otel_v2_enabled  # noqa: E402
from litellm.integrations.otel.mount import instrument_fastapi_app  # noqa: E402
from litellm.integrations.otel.plumbing.context import resolve_request_span_context  # noqa: E402

UUID = "0b6f3a1c-9d2e-4f5a-8b7c-1d2e3f4a5b6c"
UUID_HEX = UUID.replace("-", "")
OTHER_HEX = "4bf92f3577b34da6a3ce929d0e0e4736"
PATHS = ("/v1/chat/completions", "/v1/responses", "/v1/messages")


@pytest.fixture(autouse=True)
def _isolate():
    from litellm.integrations.otel.plumbing import context as _otel_context

    _otel_context._request_root_span.set(None)
    is_otel_v2_enabled.cache_clear()
    yield
    _otel_context._request_root_span.set(None)
    set_global_response_propagator(None)
    is_otel_v2_enabled.cache_clear()


@pytest.fixture
def proxy(monkeypatch):
    """A V2-instrumented app whose LLM routes echo the body they read and start the request's LLM span and a
    detached root span, as the proxy does for the call and for its spend tracking."""
    monkeypatch.setenv("LITELLM_OTEL_V2", "1")
    is_otel_v2_enabled.cache_clear()
    logger = OpenTelemetryV2(config=OpenTelemetryV2Config(exporter="in_memory"))
    provider = logger._tracer_provider
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    app = fastapi.FastAPI()

    async def llm_route(request: fastapi.Request):
        body = await request.body()
        tracer.start_span("llm", context=resolve_request_span_context()).end()
        tracer.start_span("detached", context=trace.set_span_in_context(trace.INVALID_SPAN)).end()
        return fastapi.Response(content=body, media_type=request.headers.get("content-type"))

    for path in PATHS:
        app.add_api_route(path, llm_route, methods=["POST"])
    instrument_fastapi_app(app, tracer_provider=provider)
    return TestClient(app), exporter


def _trace_ids(exporter, name=None):
    return [
        f"{span.context.trace_id:032x}"
        for span in exporter.get_finished_spans()
        if (span.kind is SpanKind.SERVER if name is None else span.name == name)
    ]


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize(
    "body, expected",
    [
        pytest.param({"metadata": {"trace_id": UUID}}, UUID_HEX, id="metadata-uuid"),
        pytest.param({"metadata": {"trace_id": UUID_HEX.upper()}}, UUID_HEX, id="metadata-32-hex"),
        pytest.param({"litellm_metadata": {"trace_id": UUID}}, UUID_HEX, id="litellm_metadata-uuid"),
        pytest.param({"metadata": {"existing_trace_id": UUID}}, UUID_HEX, id="existing_trace_id"),
        pytest.param(
            {"metadata": {"trace_id": OTHER_HEX, "existing_trace_id": UUID}}, UUID_HEX, id="existing_trace_id-wins"
        ),
        pytest.param(
            {"metadata": {"trace_id": UUID}, "litellm_metadata": {"trace_id": OTHER_HEX}}, UUID_HEX, id="metadata-wins"
        ),
        pytest.param(
            {"metadata": {"trace_id": "order-42"}}, sha256(b"order-42").digest()[:16].hex(), id="not-hex-is-hashed"
        ),
        pytest.param(
            {"metadata": {"trace_id": "0" * 32}}, sha256(b"0" * 32).digest()[:16].hex(), id="all-zero-is-hashed"
        ),
    ],
)
def test_requests_naming_one_trace_share_that_otel_trace(proxy, path, body, expected):
    client, exporter = proxy

    responses = [client.post(path, json={"model": "m", **body}) for _ in range(2)]

    assert [r.json() for r in responses] == [{"model": "m", **body}] * 2
    assert _trace_ids(exporter) == [expected, expected]
    assert _trace_ids(exporter, "llm") == [expected, expected]
    assert [r.headers["traceresponse"].split("-")[1] for r in responses] == [expected, expected]


def test_a_langfuse_trace_id_header_names_the_trace(proxy):
    client, exporter = proxy

    for _ in range(2):
        client.post(PATHS[0], json={"model": "m"}, headers={"langfuse_trace_id": UUID})

    assert _trace_ids(exporter) == [UUID_HEX, UUID_HEX]


def test_requests_naming_different_traces_stay_apart(proxy):
    client, exporter = proxy

    client.post(PATHS[1], json={"litellm_metadata": {"trace_id": UUID}})
    client.post(PATHS[1], json={"litellm_metadata": {"trace_id": OTHER_HEX}})

    assert _trace_ids(exporter) == [UUID_HEX, OTHER_HEX]


def test_traceparent_wins_over_a_named_trace(proxy):
    client, exporter = proxy

    client.post(
        PATHS[1],
        json={"litellm_metadata": {"trace_id": UUID}},
        headers={"traceparent": f"00-{OTHER_HEX}-00f067aa0ba902b7-01"},
    )

    assert _trace_ids(exporter) == [OTHER_HEX]
    assert _trace_ids(exporter, "llm") == [OTHER_HEX]
    assert OTHER_HEX not in _trace_ids(exporter, "detached")


def test_work_the_request_detaches_keeps_its_own_trace(proxy):
    client, exporter = proxy

    client.post(PATHS[0], json={"metadata": {"trace_id": UUID}})
    client.post(PATHS[0], json={"metadata": {"trace_id": UUID}})

    detached = _trace_ids(exporter, "detached")
    assert len(set(detached)) == 2
    assert UUID_HEX not in detached


@pytest.mark.parametrize(
    "request_kwargs",
    [
        pytest.param({"json": {"model": "m"}}, id="no-trace-id"),
        pytest.param({"json": {"metadata": {"session_id": UUID}}}, id="session-only"),
        pytest.param({"json": {"metadata": {"trace_id": ""}}}, id="empty"),
        pytest.param({"json": {"metadata": {"trace_id": 7}}}, id="not-a-string"),
        pytest.param({"json": {"metadata": "trace_id"}}, id="metadata-not-an-object"),
        pytest.param({"json": ["trace_id"]}, id="body-not-an-object"),
        pytest.param({"json": {"model": "m"}, "headers": {"x-litellm-session-id": UUID}}, id="session-header"),
        pytest.param(
            {"content": b'{"metadata": {"trace_id": ', "headers": {"content-type": "application/json"}},
            id="invalid-json",
        ),
        pytest.param(
            {"content": b'{"metadata": {"trace_id": "%s"}}' % UUID.encode(), "headers": {"content-type": "text/plain"}},
            id="not-json-content-type",
        ),
    ],
)
def test_requests_naming_no_trace_each_get_their_own(proxy, request_kwargs):
    client, exporter = proxy

    responses = [client.post(PATHS[0], **request_kwargs) for _ in range(2)]

    assert [r.status_code for r in responses] == [200, 200]
    assert responses[0].content == client.build_request("POST", PATHS[0], **request_kwargs).content
    server = _trace_ids(exporter)
    assert len(set(server)) == 2
    assert UUID_HEX not in server


def test_a_body_sent_in_chunks_reaches_the_route_whole(proxy):
    client, exporter = proxy
    payload = b'{"metadata": {"trace_id": "%s"}, "input": "%s"}' % (UUID.encode(), b"x" * 200_000)

    def chunks():
        for start in range(0, len(payload), 4096):
            yield payload[start : start + 4096]

    response = client.post(PATHS[1], content=chunks(), headers={"content-type": "application/json"})

    assert response.content == payload
    assert _trace_ids(exporter) == [UUID_HEX]
