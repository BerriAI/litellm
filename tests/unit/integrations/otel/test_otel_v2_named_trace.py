"""Proxy requests that name a trace (``trace_id`` in ``metadata`` / ``litellm_metadata``) share that OTel trace on V2."""

import asyncio
import json
import logging
import time
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
from litellm.integrations.otel.plumbing import named_trace, providers  # noqa: E402
from litellm.integrations.otel.plumbing.context import resolve_request_span_context  # noqa: E402
from litellm.proxy.litellm_pre_call_utils import add_otel_trace_id_to_request  # noqa: E402

UUID: Final = "0b6f3a1c-9d2e-4f5a-8b7c-1d2e3f4a5b6c"
UUID_HEX: Final = UUID.replace("-", "")
OTHER_HEX: Final = "4bf92f3577b34da6a3ce929d0e0e4736"
PATHS: Final = ("/v1/chat/completions", "/v1/responses", "/v1/messages")


@pytest.fixture(autouse=True)
def _isolate():
    from litellm.integrations.otel.plumbing import context as _otel_context

    _otel_context._request_root_span.set(None)
    is_otel_v2_enabled.cache_clear()
    named_trace._warn_once.cache_clear()
    yield
    _otel_context._request_root_span.set(None)
    set_global_response_propagator(None)
    is_otel_v2_enabled.cache_clear()


@pytest.fixture
def proxy(monkeypatch):
    """A V2-instrumented app whose LLM routes echo the body they read and start the request's LLM span and a
    detached root span, as the proxy does for the call and for its spend tracking."""
    return _instrumented(monkeypatch)


async def _spend_session(request: fastapi.Request):
    data: Final = await request.json()
    add_otel_trace_id_to_request(data, "metadata", trace.get_current_span())
    return {
        "litellm_trace_id": data.get("litellm_trace_id"),
        "server_trace": f"{trace.get_current_span().get_span_context().trace_id:032x}",
    }


def _instrumented(monkeypatch, route=None, **instrument_kwargs):
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
        received_at: Final = getattr(request.state, "litellm_received_at", None)
        return fastapi.Response(
            content=body,
            media_type=request.headers.get("content-type"),
            headers={} if received_at is None else {"x-received-at": f"{received_at.timestamp()}"},
        )

    for path in PATHS:
        app.add_api_route(path, route or llm_route, methods=["POST"])
    instrument_fastapi_app(app, tracer_provider=provider, **instrument_kwargs)
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


@pytest.mark.parametrize(
    ("path", "own", "other"),
    [(PATHS[0], "metadata", "litellm_metadata"), (PATHS[1], "litellm_metadata", "metadata")],
)
def test_the_routes_own_metadata_field_names_the_trace_over_the_other(proxy, path, own, other):
    client, exporter = proxy

    client.post(path, json={own: {"trace_id": UUID}, other: {"trace_id": OTHER_HEX}})

    assert _trace_ids(exporter) == [UUID_HEX]


def test_only_a_trace_id_the_proxy_generated_is_flagged_random(proxy):
    client, exporter = proxy

    client.post(PATHS[0], json={"metadata": {"trace_id": UUID}})
    client.post(PATHS[0], json={"model": "m"})

    named, unnamed = [span for span in exporter.get_finished_spans() if span.kind is SpanKind.SERVER]
    assert (named.context.trace_flags.random_trace_id, unnamed.context.trace_flags.random_trace_id) == (False, True)
    assert all(
        span.context.trace_flags.random_trace_id is (span.context.trace_id != int(UUID_HEX, 16))
        for span in exporter.get_finished_spans()
    )


def test_a_langfuse_trace_id_header_names_the_trace(proxy):
    client, exporter = proxy

    for _ in range(2):
        client.post(PATHS[0], json={"model": "m"}, headers={"langfuse_trace_id": UUID})

    assert _trace_ids(exporter) == [UUID_HEX, UUID_HEX]


@pytest.mark.parametrize(
    "headers, expected",
    [
        pytest.param({"langfuse_trace_id": UUID}, UUID_HEX, id="langfuse-header-wins"),
        pytest.param(
            [("langfuse_trace_id", UUID), ("langfuse_trace_id", OTHER_HEX)], UUID_HEX, id="first-repeated-header-wins"
        ),
        pytest.param({"x-litellm-trace-id": UUID}, sha256(b"body-trace").digest()[:16].hex(), id="litellm-header-unread"),
    ],
)
def test_only_the_langfuse_trace_id_header_wins_over_the_body(proxy, headers, expected):
    client, exporter = proxy

    client.post(PATHS[1], json={"litellm_metadata": {"trace_id": "body-trace"}}, headers=headers)

    assert _trace_ids(exporter) == [expected]


def test_requests_naming_different_traces_stay_apart(proxy):
    client, exporter = proxy

    client.post(PATHS[1], json={"litellm_metadata": {"trace_id": UUID}})
    client.post(PATHS[1], json={"litellm_metadata": {"trace_id": OTHER_HEX}})

    assert _trace_ids(exporter) == [UUID_HEX, OTHER_HEX]


@pytest.mark.parametrize(
    "traceparent",
    [f"00-{'0' * 32}-00f067aa0ba902b7-01", "not-a-traceparent", f"00-{OTHER_HEX}-{'0' * 16}-01"],
    ids=["all-zero-trace", "malformed", "all-zero-parent"],
)
def test_a_traceparent_carrying_no_valid_trace_leaves_the_named_trace(proxy, traceparent):
    client, exporter = proxy

    client.post(PATHS[1], json={"litellm_metadata": {"trace_id": UUID}}, headers={"traceparent": traceparent})

    assert _trace_ids(exporter) == [UUID_HEX]


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
        pytest.param({"json": {"model": "m"}, "headers": {"x-litellm-trace-id": UUID}}, id="litellm-trace-header"),
        pytest.param({"json": {"metadata": {"existing_trace_id": UUID}}}, id="existing_trace_id"),
        pytest.param(
            {"json": {"metadata": {"trace_id": UUID, "existing_trace_id": OTHER_HEX}}}, id="trace_id-joining-a-trace"
        ),
        pytest.param(
            {"json": {"litellm_metadata": {"trace_id": UUID}}, "headers": {"langfuse_existing_trace_id": OTHER_HEX}},
            id="existing-header",
        ),
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


def test_the_time_spent_reading_a_body_for_its_trace_name_counts_from_when_the_request_arrived(proxy):
    client, exporter = proxy
    payload: Final = b'{"litellm_metadata": {"trace_id": "%s"}}' % UUID.encode()
    upload_seconds: Final = 0.2
    rest_sent_at: Final = []

    def slow_upload():
        yield payload[:10]
        time.sleep(upload_seconds)
        rest_sent_at.append(time.time())
        yield payload[10:]

    response: Final = client.post(PATHS[1], content=slow_upload(), headers={"content-type": "application/json"})

    (server,) = [span for span in exporter.get_finished_spans() if span.kind is SpanKind.SERVER]
    assert response.content == payload
    assert f"{server.context.trace_id:032x}" == UUID_HEX
    assert server.end_time - server.start_time >= upload_seconds * 1e9
    assert float(response.headers["x-received-at"]) <= rest_sent_at[0] - upload_seconds
    assert abs(float(response.headers["x-received-at"]) * 1e9 - server.start_time) < 1e6


@pytest.mark.parametrize(
    "request_kwargs",
    [
        pytest.param({"json": {"metadata": {}}, "headers": {"langfuse_trace_id": UUID}}, id="header"),
        pytest.param({"json": {"metadata": {}, "litellm_metadata": {"trace_id": UUID}}}, id="other-metadata-field"),
    ],
)
def test_a_named_trace_does_not_become_the_requests_spend_session(monkeypatch, request_kwargs):
    client, _ = _instrumented(monkeypatch, route=_spend_session)

    responses: Final = [client.post(PATHS[0], **request_kwargs).json() for _ in range(2)]

    assert responses == [{"litellm_trace_id": None, "server_trace": UUID_HEX}] * 2


def test_a_request_naming_no_trace_keeps_its_server_trace_as_its_spend_session(monkeypatch):
    client, _ = _instrumented(monkeypatch, route=_spend_session)

    response: Final = client.post(PATHS[0], json={"metadata": {}}).json()

    assert response["litellm_trace_id"] == response["server_trace"]


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"metadata": {"trace_id": UUID, "existing_trace_id": OTHER_HEX}}, id="existing_trace_id"),
        pytest.param({"metadata": {"trace_id": UUID}, "input": "x" * 200}, id="over-the-lookahead-cap"),
    ],
)
def test_a_request_whose_name_is_skipped_warns_once_and_logs_each_later_one_at_debug(proxy, monkeypatch, caplog, body):
    monkeypatch.setattr(named_trace, "_MAX_LOOKAHEAD_BYTES", 150)
    client, exporter = proxy

    with caplog.at_level(logging.DEBUG, logger="LiteLLM"):
        for _ in range(3):
            client.post(PATHS[0], json=body)

    skipped: Final = [record for record in caplog.records if "OTel V2" in record.getMessage()]
    assert [record.levelno for record in skipped].count(logging.WARNING) == 1
    assert [record.levelno for record in skipped].count(logging.DEBUG) == 3
    assert UUID_HEX not in _trace_ids(exporter)


def _sdk_logger():
    config: Final = OpenTelemetryV2Config(exporter="in_memory")
    exporter: Final = InMemorySpanExporter()
    return OpenTelemetryV2(config=config, tracer_provider=providers.build_tracer_provider(config, exporter=exporter)), exporter


async def _sdk_call_traces(monkeypatch, *metadatas, parent=None):
    import litellm

    logger, exporter = _sdk_logger()
    monkeypatch.setattr(litellm, "callbacks", [logger])
    for metadata in metadatas:
        with trace.use_span(parent, end_on_exit=False) if parent is not None else _no_span():
            await litellm.acompletion(
                model="gpt-4o", messages=[{"role": "user", "content": "hi"}], mock_response="ok", metadata=metadata
            )
    for _ in range(200):
        if len(exporter.get_finished_spans()) >= len(metadatas):
            break
        await asyncio.sleep(0.01)
    return [f"{span.context.trace_id:032x}" for span in exporter.get_finished_spans() if span.parent is None or parent]


def _no_span():
    import contextlib

    return contextlib.nullcontext()


@pytest.mark.asyncio
async def test_sdk_calls_naming_one_trace_share_that_otel_trace(monkeypatch):
    assert await _sdk_call_traces(monkeypatch, {"trace_id": UUID}, {"trace_id": UUID}) == [UUID_HEX, UUID_HEX]


@pytest.mark.asyncio
async def test_sdk_calls_naming_no_trace_each_get_their_own(monkeypatch):
    traces: Final = await _sdk_call_traces(monkeypatch, {}, {})

    assert len(set(traces)) == 2


@pytest.mark.asyncio
async def test_an_sdk_call_under_an_app_span_stays_in_the_app_trace(monkeypatch):
    app_span: Final = trace.NonRecordingSpan(
        trace.SpanContext(int(OTHER_HEX, 16), 0x00F067AA0BA902B7, is_remote=False, trace_flags=trace.TraceFlags(1))
    )

    assert set(await _sdk_call_traces(monkeypatch, {"trace_id": UUID}, parent=app_span)) == {OTHER_HEX}


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


@pytest.mark.parametrize(
    ("lookahead_cap", "max_body_bytes"),
    [(10, None), (None, 10), (10, 1000)],
    ids=["lookahead cap", "app size limit", "lookahead cap under a larger app size limit"],
)
@pytest.mark.asyncio
async def test_reading_stops_at_the_lookahead_cap(monkeypatch, lookahead_cap, max_body_bytes):
    from litellm.integrations.otel.plumbing import named_trace

    if lookahead_cap is not None:
        monkeypatch.setattr(named_trace, "_MAX_LOOKAHEAD_BYTES", lookahead_cap)
    sent: Final = [{"type": "http.request", "body": b"x" * 8, "more_body": True} for _ in range(5)]
    pulled: Final = []

    async def receive():
        pulled.append(sent[len(pulled)])
        return pulled[-1]

    scope: Final = {"type": "http", "method": "POST", "headers": [(b"content-type", b"application/json")]}
    async with named_trace.read_named_trace(scope, receive, named_trace.LookaheadBudget(), max_body_bytes) as (
        named,
        replay,
    ):
        assert named is None
        assert len(pulled) == 2
        assert await replay() == {"type": "http.request", "body": b"x" * 16, "more_body": True}
        assert [await replay() for _ in sent[2:]] == sent[2:]


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
async def test_a_body_sent_in_many_tiny_pieces_is_held_as_one_message():
    from litellm.integrations.otel.plumbing import named_trace

    body: Final = b'{"metadata": {"trace_id": "%s"}}' % UUID.encode()
    sent: Final = [
        {"type": "http.request", "body": body[index : index + 1], "more_body": index < len(body) - 1}
        for index in range(len(body))
    ]
    left: Final = iter([*sent, {"type": "http.disconnect"}])

    async def receive():
        return next(left)

    scope: Final = {"type": "http", "method": "POST", "headers": [(b"content-type", b"application/json")]}
    async with named_trace.read_named_trace(scope, receive, named_trace.LookaheadBudget()) as (named, replay):
        assert named == int(UUID_HEX, 16)
        assert await replay() == {"type": "http.request", "body": body, "more_body": False}
        assert await replay() == {"type": "http.disconnect"}


@pytest.mark.asyncio
async def test_a_client_that_leaves_mid_body_is_still_seen_leaving():
    from litellm.integrations.otel.plumbing import named_trace

    left: Final = iter([{"type": "http.request", "body": b"{", "more_body": True}, {"type": "http.disconnect"}])

    async def receive():
        return next(left)

    scope: Final = {"type": "http", "method": "POST", "headers": [(b"content-type", b"application/json")]}
    async with named_trace.read_named_trace(scope, receive, named_trace.LookaheadBudget()) as (named, replay):
        assert named is None
        assert await replay() == {"type": "http.request", "body": b"{", "more_body": True}
        assert await replay() == {"type": "http.disconnect"}


def test_a_plain_body_labelled_gzip_names_its_trace(proxy):
    client, exporter = proxy
    payload: Final = json.dumps({"model": "m", "litellm_metadata": {"trace_id": UUID}}).encode()

    for _ in range(2):
        response = client.post(
            PATHS[1], content=payload, headers={"content-type": "application/json", "content-encoding": "gzip"}
        )
        assert response.content == payload

    assert set(_trace_ids(exporter)) == {UUID_HEX}


def test_a_compressed_body_expanding_past_the_proxy_ratio_names_no_trace(proxy, monkeypatch):
    import gzip

    from litellm import constants

    monkeypatch.setattr(constants, "MAX_REQUEST_BODY_DECOMPRESSION_RATIO", 2)
    client, exporter = proxy
    payload: Final = gzip.compress(b'{"metadata": {"trace_id": "%s"}, "input": "%s"}' % (UUID.encode(), b"x" * 50_000))

    response = client.post(
        PATHS[1], content=payload, headers={"content-type": "application/json", "content-encoding": "gzip"}
    )

    assert response.content == payload
    assert UUID_HEX not in _trace_ids(exporter)


JSON_POST: Final = {"type": "http", "method": "POST", "headers": [(b"content-type", b"application/json")]}
GZIP_JSON_POST: Final = {**JSON_POST, "headers": [*JSON_POST["headers"], (b"content-encoding", b"gzip")]}


def _named_body(trace_id: str = UUID) -> bytes:
    return b'{"metadata": {"trace_id": "%s"}}' % trace_id.encode()


def _sends(*messages):
    left: Final = iter(messages)

    async def receive():
        return next(left)

    return receive


@pytest.mark.asyncio
async def test_partial_uploads_together_hold_no_more_than_the_budget():
    import asyncio

    from litellm.integrations.otel.plumbing import named_trace

    budget: Final = named_trace.LookaheadBudget(max_bytes=100)
    rest_sent: Final = asyncio.Event()
    first_body: Final = _named_body(OTHER_HEX)

    async def stalled_upload():
        if not rest_sent.is_set():
            await rest_sent.wait()
            return {"type": "http.request", "body": first_body[60:], "more_body": False}
        raise AssertionError("read past the end of the body")

    first_pieces: Final = iter([{"type": "http.request", "body": first_body[:60], "more_body": True}])

    async def first_receive():
        piece = next(first_pieces, None)
        return piece if piece is not None else await stalled_upload()

    async def first_request():
        async with named_trace.read_named_trace(JSON_POST, first_receive, budget) as (named, _):
            return named

    first: Final = asyncio.create_task(first_request())
    await asyncio.sleep(0)
    second_body: Final = _named_body()
    second_pieces: Final = [
        {"type": "http.request", "body": second_body[:60], "more_body": True},
        {"type": "http.request", "body": second_body[60:], "more_body": False},
    ]
    second_pulled: Final = []

    async def second_receive():
        second_pulled.append(second_pieces[len(second_pulled)])
        return second_pulled[-1]

    async with named_trace.read_named_trace(JSON_POST, second_receive, budget) as (named, replay):
        assert named is None
        assert len(second_pulled) == 1
        assert await replay() == second_pieces[0]
        assert await replay() == second_pieces[1]

    rest_sent.set()
    assert await first == int(OTHER_HEX, 16)
    async with named_trace.read_named_trace(
        JSON_POST, _sends({"type": "http.request", "body": _named_body(), "more_body": False}), budget
    ) as (named, _):
        assert named == int(UUID_HEX, 16)


@pytest.mark.asyncio
async def test_a_body_the_app_never_read_gives_its_budget_back_when_the_request_ends():
    from litellm.integrations.otel.plumbing import named_trace

    body: Final = _named_body()
    budget: Final = named_trace.LookaheadBudget(max_bytes=len(body))

    for _ in range(3):
        async with named_trace.read_named_trace(
            JSON_POST, _sends({"type": "http.request", "body": body, "more_body": False}), budget
        ) as (named, _):
            assert named == int(UUID_HEX, 16)


@pytest.mark.asyncio
async def test_a_body_the_app_has_read_gives_its_budget_back_while_the_request_runs():
    from litellm.integrations.otel.plumbing import named_trace

    sent: Final = {"type": "http.request", "body": _named_body(), "more_body": False}
    budget: Final = named_trace.LookaheadBudget(max_bytes=len(sent["body"]))

    async with named_trace.read_named_trace(JSON_POST, _sends(sent), budget) as (named, replay):
        assert await replay() == sent
        async with named_trace.read_named_trace(JSON_POST, _sends(sent), budget) as (later, _):
            assert (named, later) == (int(UUID_HEX, 16), int(UUID_HEX, 16))


@pytest.mark.asyncio
async def test_a_compressed_body_names_no_trace_when_the_budget_cannot_cover_its_decoding():
    import gzip

    from litellm import constants
    from litellm.integrations.otel.plumbing import named_trace

    sent: Final = {"type": "http.request", "body": gzip.compress(_named_body()), "more_body": False}
    decoded_limit: Final = len(sent["body"]) * constants.MAX_REQUEST_BODY_DECOMPRESSION_RATIO
    short: Final = named_trace.LookaheadBudget(max_bytes=len(sent["body"]) + decoded_limit - 1)
    enough: Final = named_trace.LookaheadBudget(max_bytes=len(sent["body"]) + decoded_limit)

    async with named_trace.read_named_trace(GZIP_JSON_POST, _sends(sent), short) as (named, replay):
        assert named is None
        assert await replay() == sent
    for _ in range(2):
        async with named_trace.read_named_trace(GZIP_JSON_POST, _sends(sent), enough) as (named, _):
            assert named == int(UUID_HEX, 16)


@pytest.mark.asyncio
async def test_a_compressed_body_is_not_held_up_by_another_requests_decoding(monkeypatch):
    import gzip
    import threading

    import anyio

    from litellm.integrations.otel.plumbing import named_trace

    slow_body: Final = _named_body(OTHER_HEX)
    slow_decoding: Final = threading.Event()
    read_json: Final = named_trace._json_object

    def json_object(body):
        if body == slow_body:
            slow_decoding.wait(10)
        return read_json(body)

    monkeypatch.setattr(named_trace, "_json_object", json_object)
    budget: Final = named_trace.LookaheadBudget()

    async def decode(body):
        sent: Final = {"type": "http.request", "body": gzip.compress(body), "more_body": False}
        async with named_trace.read_named_trace(GZIP_JSON_POST, _sends(sent), budget) as (named, _):
            return named

    try:
        async with anyio.create_task_group() as group:
            group.start_soon(decode, slow_body)
            await anyio.sleep(0.05)
            with anyio.fail_after(5):
                assert await decode(_named_body()) == int(UUID_HEX, 16)
            slow_decoding.set()
    finally:
        slow_decoding.set()


@pytest.mark.asyncio
async def test_a_body_declared_over_the_apps_size_limit_is_never_read():
    from litellm.integrations.otel.plumbing import named_trace

    body: Final = _named_body()
    scope: Final = {**JSON_POST, "headers": [*JSON_POST["headers"], (b"content-length", b"%d" % len(body))]}

    async def receive():
        raise AssertionError("read a body the app turns away by its declared length")

    async with named_trace.read_named_trace(scope, receive, named_trace.LookaheadBudget(), len(body) - 1) as (
        named,
        replay,
    ):
        assert named is None
        assert replay is receive
    sent: Final = {"type": "http.request", "body": body, "more_body": False}
    async with named_trace.read_named_trace(scope, _sends(sent), named_trace.LookaheadBudget(), len(body)) as (
        named,
        _,
    ):
        assert named == int(UUID_HEX, 16)


@pytest.mark.asyncio
async def test_a_compressed_body_decoding_past_the_apps_size_limit_names_no_trace():
    import gzip

    from litellm.integrations.otel.plumbing import named_trace

    body: Final = b'{"metadata": {"trace_id": "%s"}, "input": "%s"}' % (UUID.encode(), b"x" * 1000)
    sent: Final = {"type": "http.request", "body": gzip.compress(body), "more_body": False}

    async with named_trace.read_named_trace(
        GZIP_JSON_POST, _sends(sent), named_trace.LookaheadBudget(), len(body) - 1
    ) as (
        named,
        replay,
    ):
        assert named is None
        assert await replay() == sent
    async with named_trace.read_named_trace(GZIP_JSON_POST, _sends(sent), named_trace.LookaheadBudget(), len(body)) as (
        named,
        _,
    ):
        assert named == int(UUID_HEX, 16)


def test_the_apps_size_limit_for_each_request_bounds_its_lookahead(monkeypatch):
    body: Final = _named_body()
    client, exporter = _instrumented(
        monkeypatch, max_request_body_bytes=lambda scope: len(body) - 1 if scope["path"] == PATHS[1] else None
    )

    over_limit: Final = client.post(PATHS[1], content=body, headers={"content-type": "application/json"})
    within_limit: Final = client.post(PATHS[0], content=body, headers={"content-type": "application/json"})

    assert over_limit.content == within_limit.content == body
    assert _trace_ids(exporter).count(UUID_HEX) == 1
