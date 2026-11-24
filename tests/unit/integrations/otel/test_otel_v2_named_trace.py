"""Proxy requests that name a trace (``trace_id`` in ``metadata`` / ``litellm_metadata``) share that OTel trace on V2."""

import json
from hashlib import sha256
from typing import Final

import pytest

fastapi: Final = pytest.importorskip("fastapi")
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

UUID: Final = "0b6f3a1c-9d2e-4f5a-8b7c-1d2e3f4a5b6c"
UUID_HEX: Final = UUID.replace("-", "")
OTHER_HEX: Final = "4bf92f3577b34da6a3ce929d0e0e4736"
PATHS: Final = ("/v1/chat/completions", "/v1/responses", "/v1/messages")


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
    logger: Final = OpenTelemetryV2(config=OpenTelemetryV2Config(exporter="in_memory"))
    provider: Final = logger._tracer_provider
    exporter: Final = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer: Final = provider.get_tracer("test")
    app: Final = fastapi.FastAPI()

    async def llm_route(request: fastapi.Request):
        body: Final = await request.body()
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

    responses: Final = [client.post(path, json={"model": "m", **body}) for _ in range(2)]

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

    detached: Final = _trace_ids(exporter, "detached")
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

    responses: Final = [client.post(PATHS[0], **request_kwargs) for _ in range(2)]

    assert [r.status_code for r in responses] == [200, 200]
    assert responses[0].content == client.build_request("POST", PATHS[0], **request_kwargs).content
    server: Final = _trace_ids(exporter)
    assert len(set(server)) == 2
    assert UUID_HEX not in server


def test_a_body_sent_in_chunks_reaches_the_route_whole(proxy):
    client, exporter = proxy
    payload: Final = b'{"metadata": {"trace_id": "%s"}, "input": "%s"}' % (UUID.encode(), b"x" * 200_000)

    def chunks():
        for start in range(0, len(payload), 4096):
            yield payload[start : start + 4096]

    response: Final = client.post(PATHS[1], content=chunks(), headers={"content-type": "application/json"})

    assert response.content == payload
    assert _trace_ids(exporter) == [UUID_HEX]


def test_a_body_too_deep_to_parse_still_reaches_the_route(proxy):
    client, exporter = proxy
    payload: Final = b'{"metadata": {"trace_id": "%s"}, "input": %s' % (UUID.encode(), b"[" * 200_000)

    response: Final = client.post(PATHS[0], content=payload, headers={"content-type": "application/json"})

    assert response.status_code == 200
    assert response.content == payload
    assert UUID_HEX not in _trace_ids(exporter)


def test_a_body_naming_no_trace_is_never_parsed(proxy, monkeypatch):
    from litellm.integrations.otel.plumbing import named_trace

    def never_called(body: bytes) -> object:
        raise AssertionError("parsed a body that names no trace")

    monkeypatch.setattr(named_trace, "_loads", never_called)
    client, exporter = proxy

    response: Final = client.post(PATHS[1], json={"model": "m", "input": "x" * 10_000})

    assert response.status_code == 200
    assert len(_trace_ids(exporter)) == 1


def test_the_json_content_type_is_matched_in_any_case(proxy):
    client, exporter = proxy
    payload: Final = b'{"litellm_metadata": {"trace_id": "%s"}}' % UUID.encode()

    client.post(PATHS[1], content=payload, headers={"content-type": "Application/JSON; charset=utf-8"})

    assert _trace_ids(exporter) == [UUID_HEX]


@pytest.mark.parametrize("declare_length", [True, False])
def test_a_body_over_the_lookahead_cap_is_not_held_and_reaches_the_route_whole(proxy, monkeypatch, declare_length):
    from litellm.integrations.otel.plumbing import named_trace

    monkeypatch.setattr(named_trace, "_MAX_LOOKAHEAD_BYTES", 10_000)
    client, exporter = proxy
    payload: Final = b'{"metadata": {"trace_id": "%s"}, "input": "%s"}' % (UUID.encode(), b"x" * 50_000)

    def chunks():
        for start in range(0, len(payload), 4096):
            yield payload[start : start + 4096]

    response: Final = client.post(
        PATHS[1],
        content=payload if declare_length else chunks(),
        headers={"content-type": "application/json"},
    )

    assert response.content == payload
    assert UUID_HEX not in _trace_ids(exporter)


@pytest.mark.asyncio
async def test_reading_stops_at_the_lookahead_cap(monkeypatch):
    from litellm.integrations.otel.plumbing import named_trace

    monkeypatch.setattr(named_trace, "_MAX_LOOKAHEAD_BYTES", 10)
    sent: Final = [{"type": "http.request", "body": b"x" * 8, "more_body": True} for _ in range(5)]
    pulled: Final = []

    async def receive():
        pulled.append(sent[len(pulled)])
        return pulled[-1]

    scope: Final = {"type": "http", "method": "POST", "headers": [(b"content-type", b"application/json")]}
    named, replay = await named_trace.read_named_trace(scope, receive)

    assert named is None
    assert len(pulled) == 2
    assert [await replay() for _ in sent] == sent


@pytest.mark.parametrize("coding", ["gzip", "x-gzip", "zstd"])
def test_a_compressed_body_names_its_trace(proxy, coding):
    import gzip

    from litellm.proxy.middleware.request_body_decompression_middleware import AVAILABLE_CODECS

    if coding == "zstd":
        if "zstd" not in AVAILABLE_CODECS:
            pytest.skip("zstd is not installed")
        try:
            from compression import zstd
        except ImportError:
            from backports import zstd
        compress = zstd.compress
    else:
        compress = gzip.compress
    client, exporter = proxy
    payload: Final = compress(json.dumps({"model": "m", "litellm_metadata": {"trace_id": UUID}}).encode())

    for _ in range(2):
        response: Final = client.post(
            PATHS[1], content=payload, headers={"content-type": "application/json", "content-encoding": coding}
        )
        assert response.content == payload

    assert set(_trace_ids(exporter)) == {UUID_HEX}


@pytest.mark.parametrize(
    "coding, payload",
    [
        ("gzip", b"not gzip at all"),
        ("br", b'{"metadata": {"trace_id": "%s"}}' % UUID.encode()),
        (
            "gzip",
            __import__("gzip").compress(
                b'{"metadata": {"trace_id": "%s"}, "input": "%s"}' % (UUID.encode(), b"x" * 50_000)
            ),
        ),
    ],
    ids=["damaged", "unknown coding", "decodes past the cap"],
)
def test_a_compressed_body_that_cannot_be_read_still_reaches_the_route(proxy, monkeypatch, coding, payload):
    from litellm.integrations.otel.plumbing import named_trace

    monkeypatch.setattr(named_trace, "_MAX_LOOKAHEAD_BYTES", 10_000)
    client, exporter = proxy

    response: Final = client.post(
        PATHS[1], content=payload, headers={"content-type": "application/json", "content-encoding": coding}
    )

    assert response.content == payload
    assert UUID_HEX not in _trace_ids(exporter)


@pytest.mark.asyncio
async def test_a_body_sent_in_many_tiny_pieces_is_counted_once_per_piece():
    from litellm.integrations.otel.plumbing import named_trace

    body_reads: Final = []

    class Piece(dict):
        def get(self, key, default=None):
            if key == "body":
                body_reads.append(key)
            return super().get(key, default)

    sent: Final = [Piece(type="http.request", body=b"x", more_body=index < 999) for index in range(1000)]
    remaining: Final = iter(sent)

    async def receive():
        return next(remaining)

    scope: Final = {"type": "http", "method": "POST", "headers": [(b"content-type", b"application/json")]}
    named, replay = await named_trace.read_named_trace(scope, receive)

    assert named is None
    assert len(body_reads) == 2 * len(sent), "each piece is read once to count it and once to join the body"
    assert [await replay() for _ in sent] == sent
