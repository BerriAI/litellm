"""Proxy requests that name a trace (``trace_id`` in ``metadata`` / ``litellm_metadata``) share that trace in Langfuse on
V2, while every other exporter keeps each request's own trace."""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from types import MappingProxyType
from hashlib import sha256
from typing import Final

import pytest

fastapi: Final = pytest.importorskip("fastapi")
pytest.importorskip("opentelemetry.instrumentation.fastapi")

from fastapi.testclient import TestClient  # noqa: E402  # optional dependencies are skipped first
from opentelemetry import trace  # noqa: E402
from opentelemetry.instrumentation.propagators import set_global_response_propagator  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402
from opentelemetry.trace import SpanKind  # noqa: E402

from litellm.integrations.otel.logger import OpenTelemetryV2  # noqa: E402
from litellm.integrations.otel.model.config import (  # noqa: E402
    ExporterOwner,
    ExporterSpec,
    OpenTelemetryV2Config,
    is_otel_v2_enabled,
)
from litellm.integrations.otel.model.destination import OtelDestination  # noqa: E402
from litellm.integrations.otel.model.semconv import GenAI, LiteLLM  # noqa: E402
from litellm.integrations.otel.mount import instrument_fastapi_app  # noqa: E402
from litellm.integrations.otel.plumbing import named_trace  # noqa: E402
from litellm.integrations.otel.plumbing.context import (  # noqa: E402
    reset_request_destinations,
    resolve_request_span_context,
    set_request_destinations,
)
from litellm.integrations.otel.plumbing.providers import (  # noqa: E402
    TenantFanOutSpanProcessor,
    parse_headers,
    register_exporter_factory,
)
from litellm.proxy.auth.user_api_key_auth import _ensure_litellm_received_at_on_request_state  # noqa: E402
from litellm.proxy.litellm_pre_call_utils import add_otel_trace_id_to_request  # noqa: E402

UUID: Final = "0b6f3a1c-9d2e-4f5a-8b7c-1d2e3f4a5b6c"
UUID_HEX: Final = UUID.replace("-", "")
OTHER_HEX: Final = "4bf92f3577b34da6a3ce929d0e0e4736"
PATHS: Final = ("/v1/chat/completions", "/v1/responses", "/v1/messages")
NAMED: Final = named_trace.TraceNaming(int(UUID_HEX, 16))
NOT_NAMED: Final = named_trace.TraceNaming()
EXPORTER_KIND: Final = "lit11186_named_trace"


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


@dataclass(frozen=True)
class _Exported:
    langfuse: InMemorySpanExporter
    otel: InMemorySpanExporter


@dataclass(frozen=True)
class _Proxy:
    client: TestClient
    langfuse: InMemorySpanExporter
    otel: InMemorySpanExporter


@pytest.fixture
def proxy(monkeypatch):
    """A V2-instrumented app exporting to Langfuse and to a plain OTLP collector, whose LLM routes echo the body
    they read and start the request's LLM span and a detached root span, as the proxy does for the call and for its
    spend tracking."""
    return _instrumented(monkeypatch)


def _logger(**logger_kwargs):
    exporters: Final = {}
    register_exporter_factory(EXPORTER_KIND, lambda spec: exporters.setdefault(spec.owner, InMemorySpanExporter()))
    config: Final = OpenTelemetryV2Config(
        exporters=[
            ExporterSpec(
                kind=EXPORTER_KIND,
                endpoint="http://langfuse.local",
                owner=ExporterOwner.LANGFUSE_OTEL,
                use_simple_processor=True,
            ),
            ExporterSpec(kind=EXPORTER_KIND, endpoint="http://collector.local", use_simple_processor=True),
        ]
    )
    logger: Final = OpenTelemetryV2(config=config, **logger_kwargs)
    return logger, _Exported(exporters[ExporterOwner.LANGFUSE_OTEL], exporters[None])


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
    logger, exported = _logger()
    provider: Final = logger._tracer_provider
    tracer: Final = provider.get_tracer("test")
    app: Final = fastapi.FastAPI()

    async def llm_route(request: fastapi.Request):
        body: Final = await request.body()
        tracer.start_span("llm", context=resolve_request_span_context()).end()
        tracer.start_span("detached", context=trace.set_span_in_context(trace.INVALID_SPAN)).end()
        return fastapi.Response(content=body, media_type=request.headers.get("content-type"))

    for path in PATHS:
        app.add_api_route(path, route or llm_route, methods=["POST"])
    instrument_fastapi_app(app, tracer_provider=provider, **instrument_kwargs)
    return _Proxy(TestClient(app), exported.langfuse, exported.otel)


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
def test_requests_naming_one_trace_share_that_langfuse_trace(proxy, path, body, expected):
    client, exporter = proxy.client, proxy.langfuse

    responses: Final = [client.post(path, json={"model": "m", **body}) for _ in range(2)]

    assert [r.json() for r in responses] == [{"model": "m", **body}] * 2
    assert _trace_ids(exporter) == [expected, expected]
    assert _trace_ids(exporter, "llm") == [expected, expected]


def test_every_other_exporter_and_the_traceresponse_keep_each_requests_own_trace(proxy):
    responses: Final = [proxy.client.post(PATHS[0], json={"metadata": {"trace_id": UUID}}) for _ in range(2)]

    server: Final = _trace_ids(proxy.otel)
    assert len(set(server)) == 2
    assert UUID_HEX not in server
    assert _trace_ids(proxy.otel, "llm") == server
    assert [r.headers["traceresponse"].split("-")[1] for r in responses] == server
    assert all(LiteLLM.NAMED_TRACE_ID not in span.attributes for span in proxy.otel.get_finished_spans())


def test_every_span_of_a_named_langfuse_trace_links_to_the_same_span_in_the_trace_every_other_exporter_sees(proxy):
    proxy.client.post(PATHS[0], json={"metadata": {"trace_id": UUID}})

    seen: Final = {span.context.span_id: span.context for span in proxy.otel.get_finished_spans()}
    langfuse: Final = proxy.langfuse.get_finished_spans()
    named: Final = [span for span in langfuse if span.context.trace_id == int(UUID_HEX, 16)]
    (server,) = [span for span in named if span.kind is SpanKind.SERVER]
    (llm,) = [span for span in named if span.name == "llm"]
    assert (llm.parent.trace_id, llm.parent.span_id) == (int(UUID_HEX, 16), server.context.span_id)
    assert [[link.context for link in span.links] for span in named] == [[seen[span.context.span_id]] for span in named]
    assert [span.links for span in langfuse if span not in named] == [()]
    assert all(LiteLLM.NAMED_TRACE_ID not in span.attributes for span in langfuse)


def _named_call_through_a_destination(
    callback_name: str, span_scope: str = "full"
) -> tuple[InMemorySpanExporter, InMemorySpanExporter]:
    destination: Final = InMemorySpanExporter()
    other: Final = InMemorySpanExporter()
    provider: Final = TracerProvider()
    provider.add_span_processor(named_trace.NamedTraceSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(other))
    provider.add_span_processor(
        TenantFanOutSpanProcessor(
            processor_factory=lambda _destination: SimpleSpanProcessor(destination),
            operator_sinks=MappingProxyType({}),
        )
    )
    tracer: Final = provider.get_tracer("test")
    token: Final = set_request_destinations(
        (
            OtelDestination(
                endpoint="http://key.local/api/public/otel",
                headers=MappingProxyType({"Authorization": "Basic a2V5"}),
                callback_name=callback_name,
                span_scope=span_scope,
            ),
        )
    )
    try:
        with named_trace.named_trace_scope(int(UUID_HEX, 16)), tracer.start_as_current_span("request"):
            tracer.start_span("chat m", attributes={GenAI.OPERATION_NAME: "chat"}).end()
    finally:
        reset_request_destinations(token)
    return destination, other


@pytest.mark.parametrize("span_scope", ["full", "llm_only"])
def test_every_span_a_langfuse_destination_gets_in_a_named_trace_links_to_the_trace_other_exporters_see(span_scope):
    destination, other = _named_call_through_a_destination("langfuse_otel", span_scope)

    seen: Final = {span.context.span_id: span.context for span in other.get_finished_spans()}
    exported: Final = destination.get_finished_spans()
    assert {span.context.trace_id for span in exported} == {int(UUID_HEX, 16)}
    assert [[link.context for link in span.links] for span in exported] == [
        [seen[span.context.span_id]] for span in exported
    ]


def test_a_destination_other_than_langfuse_gets_the_requests_own_trace_with_nothing_about_the_named_one():
    destination, _other = _named_call_through_a_destination("arize")

    exported: Final = destination.get_finished_spans()

    assert len(exported) == 2
    assert int(UUID_HEX, 16) not in {span.context.trace_id for span in exported}
    assert all(LiteLLM.NAMED_TRACE_ID not in span.attributes for span in exported)


@pytest.mark.parametrize(
    ("path", "own", "other"),
    [(PATHS[0], "metadata", "litellm_metadata"), (PATHS[1], "litellm_metadata", "metadata")],
)
def test_the_routes_own_metadata_field_names_the_trace_over_the_other(proxy, path, own, other):
    client, exporter = proxy.client, proxy.langfuse

    client.post(path, json={own: {"trace_id": UUID}, other: {"trace_id": OTHER_HEX}})

    assert _trace_ids(exporter) == [UUID_HEX]


def test_only_a_trace_id_the_proxy_generated_is_flagged_random(proxy):
    client, exporter = proxy.client, proxy.langfuse

    client.post(PATHS[0], json={"metadata": {"trace_id": UUID}})
    client.post(PATHS[0], json={"model": "m"})

    named, unnamed = [span for span in exporter.get_finished_spans() if span.kind is SpanKind.SERVER]
    assert (named.context.trace_flags.random_trace_id, unnamed.context.trace_flags.random_trace_id) == (False, True)
    assert all(
        span.context.trace_flags.random_trace_id is (span.context.trace_id != int(UUID_HEX, 16))
        for span in exporter.get_finished_spans()
    )
    assert all(span.context.trace_flags.random_trace_id for span in proxy.otel.get_finished_spans())


def test_a_langfuse_trace_id_header_names_the_trace(proxy):
    client, exporter = proxy.client, proxy.langfuse

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
        pytest.param(
            {"x-litellm-trace-id": UUID}, sha256(b"body-trace").digest()[:16].hex(), id="litellm-header-unread"
        ),
    ],
)
def test_only_the_langfuse_trace_id_header_wins_over_the_body(proxy, headers, expected):
    client, exporter = proxy.client, proxy.langfuse

    client.post(PATHS[1], json={"litellm_metadata": {"trace_id": "body-trace"}}, headers=headers)

    assert _trace_ids(exporter) == [expected]


def test_requests_naming_different_traces_stay_apart(proxy):
    client, exporter = proxy.client, proxy.langfuse

    client.post(PATHS[1], json={"litellm_metadata": {"trace_id": UUID}})
    client.post(PATHS[1], json={"litellm_metadata": {"trace_id": OTHER_HEX}})

    assert _trace_ids(exporter) == [UUID_HEX, OTHER_HEX]


@pytest.mark.parametrize(
    "traceparent",
    [f"00-{'0' * 32}-00f067aa0ba902b7-01", "not-a-traceparent", f"00-{OTHER_HEX}-{'0' * 16}-01"],
    ids=["all-zero-trace", "malformed", "all-zero-parent"],
)
def test_a_traceparent_carrying_no_valid_trace_leaves_the_named_trace(proxy, traceparent):
    client, exporter = proxy.client, proxy.langfuse

    client.post(PATHS[1], json={"litellm_metadata": {"trace_id": UUID}}, headers={"traceparent": traceparent})

    assert _trace_ids(exporter) == [UUID_HEX]


def test_traceparent_wins_over_a_named_trace(proxy):
    client, exporter = proxy.client, proxy.langfuse

    client.post(
        PATHS[1],
        json={"litellm_metadata": {"trace_id": UUID}},
        headers={"traceparent": f"00-{OTHER_HEX}-00f067aa0ba902b7-01"},
    )

    assert _trace_ids(exporter) == [OTHER_HEX]
    assert _trace_ids(exporter, "llm") == [OTHER_HEX]
    assert OTHER_HEX not in _trace_ids(exporter, "detached")


def test_work_the_request_detaches_keeps_its_own_trace(proxy):
    client, exporter = proxy.client, proxy.langfuse

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
            {
                "content": b'{"metadata": {"trace_id": "%s"}}' % UUID.encode(),
                "headers": {"content-type": "application/x-protobuf"},
            },
            id="binary-upload",
        ),
    ],
)
def test_requests_naming_no_trace_each_get_their_own(proxy, request_kwargs):
    client, exporter = proxy.client, proxy.langfuse

    responses: Final = [client.post(PATHS[0], **request_kwargs) for _ in range(2)]

    assert [r.status_code for r in responses] == [200, 200]
    assert responses[0].content == client.build_request("POST", PATHS[0], **request_kwargs).content
    server: Final = _trace_ids(exporter)
    assert len(set(server)) == 2
    assert UUID_HEX not in server


async def _queued_route(request: fastapi.Request):
    await asyncio.sleep(QUEUED_SECONDS)
    received_at: Final = _ensure_litellm_received_at_on_request_state(request)
    handled_at: Final = time.time()
    return fastapi.Response(
        content=await request.body(),
        headers={"x-received-at": f"{received_at.timestamp()}", "x-handled-at": f"{handled_at}"},
    )


QUEUED_SECONDS: Final = 0.2


def test_a_request_counts_the_time_spent_reading_its_body_but_not_the_time_it_waited_before_the_route(monkeypatch):
    rig: Final = _instrumented(monkeypatch, route=_queued_route)
    client, exporter = rig.client, rig.langfuse
    payload: Final = b'{"litellm_metadata": {"trace_id": "%s"}}' % UUID.encode()
    upload_seconds: Final = 0.2

    def slow_upload():
        yield payload[:10]
        time.sleep(upload_seconds)
        yield payload[10:]

    response: Final = client.post(PATHS[1], content=slow_upload(), headers={"content-type": "application/json"})

    (server,) = [span for span in exporter.get_finished_spans() if span.kind is SpanKind.SERVER]
    received_at: Final = float(response.headers["x-received-at"])
    assert response.content == payload
    assert f"{server.context.trace_id:032x}" == UUID_HEX
    assert server.end_time - server.start_time >= (upload_seconds + QUEUED_SECONDS) * 1e9
    assert float(response.headers["x-handled-at"]) - received_at >= upload_seconds
    assert received_at * 1e9 - server.start_time >= QUEUED_SECONDS * 1e9


@pytest.mark.parametrize(
    "request_kwargs",
    [
        pytest.param({"json": {"metadata": {}}}, id="no-name"),
        pytest.param({"json": {"metadata": {}}, "headers": {"langfuse_trace_id": UUID}}, id="header"),
        pytest.param({"json": {"metadata": {}, "litellm_metadata": {"trace_id": UUID}}}, id="other-metadata-field"),
    ],
)
def test_a_requests_spend_session_is_its_own_server_trace_whatever_trace_it_names(monkeypatch, request_kwargs):
    client: Final = _instrumented(monkeypatch, route=_spend_session).client

    responses: Final = [client.post(PATHS[0], **request_kwargs).json() for _ in range(2)]

    assert [r["litellm_trace_id"] for r in responses] == [r["server_trace"] for r in responses]
    assert len({r["server_trace"] for r in responses}) == 2


@pytest.mark.parametrize(
    "body, reason",
    [
        pytest.param(
            {"metadata": {"trace_id": UUID, "existing_trace_id": OTHER_HEX}},
            "existing_trace_id",
            id="existing_trace_id",
        ),
        pytest.param({"metadata": {"trace_id": UUID}, "input": "x" * 200}, "body_too_large", id="over-the-size-limit"),
    ],
)
def test_a_request_whose_name_is_skipped_says_why_on_its_root_and_warns_once(monkeypatch, caplog, body, reason):
    rig: Final = _instrumented(monkeypatch, max_request_body_bytes=lambda _scope: 150)

    with caplog.at_level(logging.DEBUG, logger="LiteLLM"):
        for _ in range(3):
            rig.client.post(PATHS[0], json=body)

    skipped: Final = [record for record in caplog.records if "OTel V2" in record.getMessage()]
    assert [record.levelno for record in skipped].count(logging.WARNING) == 1
    assert [record.levelno for record in skipped].count(logging.DEBUG) == 3
    assert UUID_HEX not in _trace_ids(rig.langfuse)
    assert [
        span.attributes.get(LiteLLM.TRACE_ID_SKIPPED)
        for span in rig.langfuse.get_finished_spans()
        if span.parent is None
    ] == [None, reason] * 3
    assert all(LiteLLM.TRACE_ID_SKIPPED not in span.attributes for span in rig.otel.get_finished_spans())


def test_requests_skipped_for_different_content_encodings_warn_once(proxy, caplog):
    client: Final = proxy.client
    payload: Final = b'{"metadata": {"trace_id": "%s"}}' % UUID.encode()

    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        for coding in ("x-first", "x-second", "x-third"):
            client.post(
                PATHS[1], content=payload, headers={"content-type": "application/json", "content-encoding": coding}
            )

    assert len([record for record in caplog.records if "OTel V2" in record.getMessage()]) == 1


async def _sdk_call_traces(monkeypatch, *metadatas, parent=None):
    import litellm

    logger, exported = _logger()
    monkeypatch.setattr(litellm, "callbacks", [logger])
    for metadata in metadatas:
        with trace.use_span(parent, end_on_exit=False) if parent is not None else _no_span():
            await litellm.acompletion(
                model="gpt-4o", messages=[{"role": "user", "content": "hi"}], mock_response="ok", metadata=metadata
            )
    for _ in range(200):
        if len(exported.langfuse.get_finished_spans()) >= len(metadatas):
            break
        await asyncio.sleep(0.01)
    return tuple(
        [f"{span.context.trace_id:032x}" for span in exporter.get_finished_spans() if span.parent is None or parent]
        for exporter in (exported.langfuse, exported.otel)
    )


def _no_span():
    import contextlib

    return contextlib.nullcontext()


@pytest.mark.asyncio
async def test_sdk_calls_naming_one_trace_share_that_langfuse_trace(monkeypatch):
    langfuse, otel = await _sdk_call_traces(monkeypatch, {"trace_id": UUID}, {"trace_id": UUID})

    assert langfuse == [UUID_HEX, UUID_HEX]
    assert len(set(otel)) == 2


@pytest.mark.asyncio
async def test_sdk_calls_naming_no_trace_each_get_their_own(monkeypatch):
    langfuse, _ = await _sdk_call_traces(monkeypatch, {}, {})

    assert len(set(langfuse)) == 2


@pytest.mark.asyncio
async def test_an_sdk_call_under_an_app_span_stays_in_the_app_trace(monkeypatch):
    app_span: Final = trace.NonRecordingSpan(
        trace.SpanContext(int(OTHER_HEX, 16), 0x00F067AA0BA902B7, is_remote=False, trace_flags=trace.TraceFlags(1))
    )

    langfuse, otel = await _sdk_call_traces(monkeypatch, {"trace_id": UUID}, parent=app_span)

    assert set(langfuse) == set(otel) == {OTHER_HEX}


def test_a_body_sent_in_chunks_reaches_the_route_whole(proxy):
    client, exporter = proxy.client, proxy.langfuse
    payload: Final = b'{"metadata": {"trace_id": "%s"}, "input": "%s"}' % (UUID.encode(), b"x" * 200_000)

    def chunks():
        for start in range(0, len(payload), 4096):
            yield payload[start : start + 4096]

    response: Final = client.post(PATHS[1], content=chunks(), headers={"content-type": "application/json"})

    assert response.content == payload
    assert _trace_ids(exporter) == [UUID_HEX]


def test_a_body_too_deep_to_parse_still_reaches_the_route(proxy):
    client, exporter = proxy.client, proxy.langfuse
    payload: Final = b'{"metadata": {"trace_id": "%s"}, "input": %s' % (UUID.encode(), b"[" * 200_000)

    response: Final = client.post(PATHS[0], content=payload, headers={"content-type": "application/json"})

    assert response.status_code == 200
    assert response.content == payload
    assert UUID_HEX not in _trace_ids(exporter)


def test_the_json_content_type_is_matched_in_any_case(proxy):
    client, exporter = proxy.client, proxy.langfuse
    payload: Final = b'{"litellm_metadata": {"trace_id": "%s"}}' % UUID.encode()

    client.post(PATHS[1], content=payload, headers={"content-type": "Application/JSON; charset=utf-8"})

    assert _trace_ids(exporter) == [UUID_HEX]


@pytest.mark.parametrize(
    "request_kwargs",
    [
        pytest.param({"content": b'{"metadata": {"trace_id": "%s"}}' % UUID.encode()}, id="no-content-type"),
        pytest.param(
            {"content": b'{"metadata": {"trace_id": "%s"}}' % UUID.encode(), "headers": {"content-type": "text/plain"}},
            id="text-content-type",
        ),
        pytest.param({"data": {"model": "m", "metadata": json.dumps({"trace_id": UUID})}}, id="urlencoded-form"),
        pytest.param(
            {
                "data": {"model": "m", "metadata": json.dumps({"trace_id": UUID})},
                "files": {"file": ("speech.wav", b"RIFF" + b"\x00" * 4096, "audio/wav")},
            },
            id="multipart-form",
        ),
    ],
)
def test_a_body_the_proxy_parses_as_json_or_as_a_form_names_its_trace(proxy, request_kwargs):
    requests: Final = [proxy.client.build_request("POST", PATHS[0], **request_kwargs) for _ in range(2)]

    responses: Final = [proxy.client.send(request) for request in requests]

    assert [response.content for response in responses] == [request.read() for request in requests]
    assert _trace_ids(proxy.langfuse) == [UUID_HEX, UUID_HEX]


@pytest.mark.parametrize(
    "form",
    [
        pytest.param({"metadata": "{not json"}, id="metadata-not-json"),
        pytest.param({"metadata": json.dumps({"session_id": "trace_id"})}, id="metadata-naming-no-trace"),
        pytest.param({"trace_id": UUID}, id="no-metadata-field"),
    ],
)
def test_a_form_whose_metadata_names_no_trace_still_reaches_the_route(proxy, form):
    request: Final = proxy.client.build_request("POST", PATHS[0], data=form)

    response: Final = proxy.client.send(request)

    assert response.content == request.read()
    assert UUID_HEX not in _trace_ids(proxy.langfuse)


@pytest.mark.parametrize(
    ("content", "content_type"),
    [
        pytest.param(b"garbage trace_id garbage", "multipart/form-data; boundary=XyZ", id="bad-multipart-body"),
        pytest.param(b"garbage trace_id garbage", "multipart/form-data; boundary=" + "b" * 300, id="boundary-too-long"),
        pytest.param(
            b"model=m&metadata=" + b"%5B" * 100_000 + b"trace_id",
            "application/x-www-form-urlencoded",
            id="deep-metadata",
        ),
        pytest.param(
            b"model=m&metadata=%7B%22trace_id%22%3A" + b"9" * 5000 + b"%7D",
            "application/x-www-form-urlencoded",
            id="metadata-integer-too-long",
        ),
    ],
)
def test_a_form_that_does_not_parse_still_reaches_the_route(proxy, content, content_type):
    response: Final = proxy.client.post(PATHS[0], content=content, headers={"content-type": content_type})

    assert response.content == content
    assert len(_trace_ids(proxy.langfuse)) == 1


@pytest.mark.parametrize("declare_length", [True, False])
def test_a_body_over_the_apps_size_limit_is_not_held_and_reaches_the_route_whole(monkeypatch, declare_length):
    rig: Final = _instrumented(monkeypatch, max_request_body_bytes=lambda _scope: 10_000)
    client, exporter = rig.client, rig.langfuse
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
    ("piece_bytes", "max_body_bytes", "pieces_read"),
    [
        (8, 10, 2),
        (named_trace._MAX_LOOKAHEAD_BYTES // 4, None, 5),
        (named_trace._MAX_LOOKAHEAD_BYTES // 4, named_trace._MAX_LOOKAHEAD_BYTES * 2, 5),
    ],
    ids=["app size limit", "lookahead cap", "lookahead cap under a larger app size limit"],
)
@pytest.mark.asyncio
async def test_reading_stops_past_the_lookahead_cap(piece_bytes, max_body_bytes, pieces_read):
    sent: Final = [{"type": "http.request", "body": b"x" * piece_bytes, "more_body": True} for _ in range(6)]
    pulled: Final = []

    async def receive():
        pulled.append(sent[len(pulled)])
        return pulled[-1]

    async with named_trace.read_named_trace(JSON_POST, receive, named_trace.LookaheadBudget(), max_body_bytes) as (
        named,
        replay,
    ):
        assert named == named_trace.TraceNaming(skipped="body_too_large")
        assert len(pulled) == pieces_read
        assert await replay() == {"type": "http.request", "body": b"x" * piece_bytes * pieces_read, "more_body": True}
        assert [await replay() for _ in sent[pieces_read:]] == sent[pieces_read:]


@pytest.mark.parametrize(
    "headers, expected",
    [
        pytest.param((), named_trace.TraceNaming(skipped="timed_out"), id="body-name"),
        pytest.param(((b"langfuse_trace_id", UUID.encode()),), NAMED, id="header-name"),
    ],
)
@pytest.mark.asyncio
async def test_a_body_still_arriving_at_the_deadline_is_handed_on_with_what_was_read(headers, expected):
    import anyio

    body: Final = _named_body()
    first: Final = {"type": "http.request", "body": body[:10], "more_body": True}
    rest: Final = {"type": "http.request", "body": body[10:], "more_body": False}
    arrived: Final = anyio.Event()
    pieces: Final = iter([first])

    async def stalled_receive():
        piece = next(pieces, None)
        if piece is not None:
            return piece
        await arrived.wait()
        return rest

    scope: Final = {**JSON_POST, "headers": [*JSON_POST["headers"], *headers]}
    with anyio.fail_after(5):
        async with named_trace.read_named_trace(
            scope, stalled_receive, named_trace.LookaheadBudget(), deadline_seconds=0.05
        ) as (named, replay):
            assert named == expected
            assert await replay() == first
            arrived.set()
            assert await replay() == rest


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
    client, exporter = proxy.client, proxy.langfuse
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
def test_a_compressed_body_that_cannot_be_read_still_reaches_the_route(monkeypatch, coding, payload):
    rig: Final = _instrumented(monkeypatch, max_request_body_bytes=lambda _scope: 10_000)
    client, exporter = rig.client, rig.langfuse

    response: Final = client.post(
        PATHS[1], content=payload, headers={"content-type": "application/json", "content-encoding": coding}
    )

    assert response.content == payload
    assert UUID_HEX not in _trace_ids(exporter)


@pytest.mark.asyncio
async def test_a_body_sent_in_many_tiny_pieces_is_held_as_one_message():
    body: Final = b'{"metadata": {"trace_id": "%s"}}' % UUID.encode()
    sent: Final = [
        {"type": "http.request", "body": body[index : index + 1], "more_body": index < len(body) - 1}
        for index in range(len(body))
    ]
    left: Final = iter([*sent, {"type": "http.disconnect"}])

    async def receive():
        return next(left)

    async with named_trace.read_named_trace(JSON_POST, receive, named_trace.LookaheadBudget()) as (named, replay):
        assert named == NAMED
        assert await replay() == {"type": "http.request", "body": body, "more_body": False}
        assert await replay() == {"type": "http.disconnect"}


@pytest.mark.asyncio
async def test_a_client_that_leaves_mid_body_is_still_seen_leaving():
    left: Final = iter([{"type": "http.request", "body": b"{", "more_body": True}, {"type": "http.disconnect"}])

    async def receive():
        return next(left)

    async with named_trace.read_named_trace(JSON_POST, receive, named_trace.LookaheadBudget()) as (named, replay):
        assert named == NOT_NAMED
        assert await replay() == {"type": "http.request", "body": b"{", "more_body": True}
        assert await replay() == {"type": "http.disconnect"}


def test_a_plain_body_labelled_gzip_names_its_trace(proxy):
    client, exporter = proxy.client, proxy.langfuse
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
    client, exporter = proxy.client, proxy.langfuse
    payload: Final = gzip.compress(b'{"metadata": {"trace_id": "%s"}, "input": "%s"}' % (UUID.encode(), b"x" * 50_000))

    response = client.post(
        PATHS[1], content=payload, headers={"content-type": "application/json", "content-encoding": "gzip"}
    )

    assert response.content == payload
    assert UUID_HEX not in _trace_ids(exporter)
    (server,) = [span for span in proxy.langfuse.get_finished_spans() if span.kind is SpanKind.SERVER]
    assert server.attributes[LiteLLM.TRACE_ID_SKIPPED] == "decoded_too_large"


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
        assert named == named_trace.TraceNaming(skipped="budget_spent")
        assert len(second_pulled) == 1
        assert await replay() == second_pieces[0]
        assert await replay() == second_pieces[1]

    rest_sent.set()
    assert await first == named_trace.TraceNaming(int(OTHER_HEX, 16))
    async with named_trace.read_named_trace(
        JSON_POST, _sends({"type": "http.request", "body": _named_body(), "more_body": False}), budget
    ) as (named, _):
        assert named == NAMED


@pytest.mark.asyncio
async def test_a_body_the_app_never_read_gives_its_budget_back_when_the_request_ends():
    body: Final = _named_body()
    budget: Final = named_trace.LookaheadBudget(max_bytes=len(body))

    for _ in range(3):
        async with named_trace.read_named_trace(
            JSON_POST, _sends({"type": "http.request", "body": body, "more_body": False}), budget
        ) as (named, _):
            assert named == NAMED


@pytest.mark.asyncio
async def test_a_body_the_app_has_read_gives_its_budget_back_while_the_request_runs():
    sent: Final = {"type": "http.request", "body": _named_body(), "more_body": False}
    budget: Final = named_trace.LookaheadBudget(max_bytes=len(sent["body"]))

    async with named_trace.read_named_trace(JSON_POST, _sends(sent), budget) as (named, replay):
        assert await replay() == sent
        async with named_trace.read_named_trace(JSON_POST, _sends(sent), budget) as (later, _):
            assert (named, later) == (NAMED, NAMED)


@pytest.mark.asyncio
async def test_a_compressed_body_names_no_trace_when_the_budget_cannot_cover_its_decoding():
    import gzip

    from litellm import constants

    sent: Final = {"type": "http.request", "body": gzip.compress(_named_body()), "more_body": False}
    decoded_limit: Final = len(sent["body"]) * constants.MAX_REQUEST_BODY_DECOMPRESSION_RATIO
    short: Final = named_trace.LookaheadBudget(max_bytes=len(sent["body"]) + decoded_limit - 1)
    enough: Final = named_trace.LookaheadBudget(max_bytes=len(sent["body"]) + decoded_limit)

    async with named_trace.read_named_trace(GZIP_JSON_POST, _sends(sent), short) as (named, replay):
        assert named == named_trace.TraceNaming(skipped="budget_spent")
        assert await replay() == sent
    for _ in range(2):
        async with named_trace.read_named_trace(GZIP_JSON_POST, _sends(sent), enough) as (named, _):
            assert named == NAMED


@pytest.mark.asyncio
async def test_a_compressed_body_is_not_held_up_by_another_requests_decoding(monkeypatch):
    import gzip
    import threading

    import anyio

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
                assert await decode(_named_body()) == NAMED
            slow_decoding.set()
    finally:
        slow_decoding.set()


@pytest.mark.asyncio
async def test_a_body_declared_over_the_apps_size_limit_is_never_read():
    body: Final = _named_body()
    scope: Final = {**JSON_POST, "headers": [*JSON_POST["headers"], (b"content-length", b"%d" % len(body))]}

    async def receive():
        raise AssertionError("read a body the app turns away by its declared length")

    async with named_trace.read_named_trace(scope, receive, named_trace.LookaheadBudget(), len(body) - 1) as (
        named,
        replay,
    ):
        assert named == named_trace.TraceNaming(skipped="body_too_large")
        assert replay is receive
    sent: Final = {"type": "http.request", "body": body, "more_body": False}
    async with named_trace.read_named_trace(scope, _sends(sent), named_trace.LookaheadBudget(), len(body)) as (
        named,
        _,
    ):
        assert named == NAMED


@pytest.mark.asyncio
async def test_a_compressed_body_decoding_past_the_apps_size_limit_names_no_trace():
    import gzip

    body: Final = b'{"metadata": {"trace_id": "%s"}, "input": "%s"}' % (UUID.encode(), b"x" * 1000)
    sent: Final = {"type": "http.request", "body": gzip.compress(body), "more_body": False}

    async with named_trace.read_named_trace(
        GZIP_JSON_POST, _sends(sent), named_trace.LookaheadBudget(), len(body) - 1
    ) as (
        named,
        replay,
    ):
        assert named == named_trace.TraceNaming(skipped="decoded_too_large")
        assert await replay() == sent
    async with named_trace.read_named_trace(GZIP_JSON_POST, _sends(sent), named_trace.LookaheadBudget(), len(body)) as (
        named,
        _,
    ):
        assert named == NAMED


def test_the_apps_size_limit_for_each_request_bounds_its_lookahead(monkeypatch):
    body: Final = _named_body()
    rig: Final = _instrumented(
        monkeypatch, max_request_body_bytes=lambda scope: len(body) - 1 if scope["path"] == PATHS[1] else None
    )
    client, exporter = rig.client, rig.langfuse

    over_limit: Final = client.post(PATHS[1], content=body, headers={"content-type": "application/json"})
    within_limit: Final = client.post(PATHS[0], content=body, headers={"content-type": "application/json"})

    assert over_limit.content == within_limit.content == body
    assert _trace_ids(exporter).count(UUID_HEX) == 1


def _llm_call_kwargs(credentials):
    return {
        "standard_logging_object": {
            "call_type": "acompletion",
            "custom_llm_provider": "openai",
            "model": "gpt-4o",
            "litellm_call_id": "call-lit11186",
            "status": "success",
            "metadata": {},
            "hidden_params": {},
        },
        "litellm_params": {"metadata": {}},
        "standard_callback_dynamic_params": credentials,
        "api_call_start_time": time.time(),
    }


@pytest.mark.parametrize("opened_at_pre_call", [True, False], ids=["opened-at-pre-call", "opened-at-success"])
def test_a_call_sent_to_a_request_level_langfuse_account_joins_the_trace_its_request_named(opened_at_pre_call):
    routed: Final = {}
    register_exporter_factory(EXPORTER_KIND, lambda spec: routed.setdefault(spec.headers, InMemorySpanExporter()))
    logger: Final = OpenTelemetryV2(
        config=OpenTelemetryV2Config(
            exporters=[
                ExporterSpec(
                    kind=EXPORTER_KIND,
                    endpoint="http://langfuse.local",
                    owner=ExporterOwner.LANGFUSE_OTEL,
                    use_simple_processor=True,
                )
            ]
        ),
        callback_name="langfuse_otel",
    )
    with named_trace.named_trace_scope(int(UUID_HEX, 16)):
        request_span: Final = logger._tracer_provider.get_tracer("test").start_span("request")
    kwargs: Final = _llm_call_kwargs({"langfuse_public_key": "pk-team", "langfuse_secret_key": "sk-team"})

    with trace.use_span(request_span, end_on_exit=False):
        if opened_at_pre_call:
            logger.log_pre_api_call(model="gpt-4o", messages=[], kwargs=kwargs)
        asyncio.run(logger.async_log_success_event(kwargs, None, None, None))
    request_span.end()

    operator: Final = routed.pop(None)
    ((team_headers, team),) = routed.items()
    (llm,) = team.get_finished_spans()
    assert "authorization" in parse_headers(team_headers)
    assert (llm.parent, f"{llm.context.trace_id:032x}") == (None, UUID_HEX)
    assert [span.name for span in operator.get_finished_spans()] == ["request"]
    assert request_span.get_span_context() in [link.context for link in llm.links]
