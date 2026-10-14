from collections.abc import Sequence
from typing import Final

import httpx
import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode, format_trace_id
from opentelemetry.trace.span import SpanContext

from litellm.proxy.control_plane.common.tracing import build_provider, build_telemetry, tracing_enabled
from litellm.proxy.control_plane.settings import ControlPlaneSettings
from tests.unit.proxy.control_plane.support import MASTER_KEY, response, rig

PROXY_ROUTE: Final = "GET /api/v1/dataplanes/{dataplane_id}/proxy/{path:path}"
USER_INFO: Final = "https://eu.test/user/info"


def _settings(**overrides: object) -> ControlPlaneSettings:
    return ControlPlaneSettings.model_validate({"LITELLM_MASTER_KEY": MASTER_KEY, **overrides})


def _by_kind(spans: Sequence[ReadableSpan], kind: SpanKind) -> list[ReadableSpan]:
    return [span for span in spans if span.kind == kind]


def _ctx(span: ReadableSpan) -> SpanContext:
    assert span.context is not None
    return span.context


def _attr(span: ReadableSpan, key: str) -> object:
    return None if span.attributes is None else span.attributes.get(key)


def test_one_proxy_call_is_one_trace_from_server_span_to_dataplane_header() -> None:
    exporter = InMemorySpanExporter()
    client, transport, minter = rig({USER_INFO: [response(200, {"ok": True})]}, build_provider(_settings(), exporter))
    reply = client.get("/api/v1/dataplanes/eu/proxy/user/info?secret=1", headers=minter.headers("eu"))
    assert reply.status_code == 200, reply.text
    spans = exporter.get_finished_spans()
    (server,), (dataplane_call,) = _by_kind(spans, SpanKind.SERVER), _by_kind(spans, SpanKind.CLIENT)
    assert server.name == PROXY_ROUTE and dataplane_call.name == "GET /user/info"
    assert dataplane_call.parent is not None and dataplane_call.parent.span_id == _ctx(server).span_id
    _, _, headers = transport.calls[0]
    assert headers is not None and headers["traceparent"].split("-")[1:3] == [
        format_trace_id(_ctx(dataplane_call).trace_id),
        format(_ctx(dataplane_call).span_id, "016x"),
    ]
    assert headers["Authorization"] == "Bearer sk-eu", "the credential survives header injection"


def test_client_span_carries_the_dataplane_and_never_the_query_string() -> None:
    exporter = InMemorySpanExporter()
    client, _, minter = rig({USER_INFO: [response(200, {})]}, build_provider(_settings(), exporter))
    client.get("/api/v1/dataplanes/eu/proxy/user/info?token=sk-leak", headers=minter.headers("eu"))
    (dataplane_call,) = _by_kind(exporter.get_finished_spans(), SpanKind.CLIENT)
    attributes = dict(dataplane_call.attributes or {})
    assert attributes["peer.service"] == "eu" and attributes["server.address"] == "eu.test"
    assert attributes["url.path"] == "/user/info" and attributes["http.response.status_code"] == 200
    assert attributes["litellm.control_plane.credential"] == "service_token"
    assert "sk-leak" not in repr(attributes) and "url.full" not in attributes and "url.query" not in attributes


def test_a_virtual_key_in_the_forwarded_path_never_reaches_the_span() -> None:
    exporter = InMemorySpanExporter()
    url = "https://eu.test/key/sk-live-123/reset_spend"
    client, _, minter = rig({url: [response(200, {})]}, build_provider(_settings(), exporter))
    client.get("/api/v1/dataplanes/eu/proxy/key/sk-live-123/reset_spend", headers=minter.headers("eu"))
    (dataplane_call,) = _by_kind(exporter.get_finished_spans(), SpanKind.CLIENT)
    assert dataplane_call.name == "GET /key/sk-[redacted]/reset_spend"
    assert _attr(dataplane_call, "url.path") == "/key/sk-[redacted]/reset_spend"
    assert "sk-live-123" not in repr(dataplane_call.attributes)


def test_the_server_span_keeps_neither_the_virtual_key_nor_the_query_string() -> None:
    exporter = InMemorySpanExporter()
    url = "https://eu.test/key/sk-live-123/reset_spend"
    client, _, minter = rig({url: [response(200, {})]}, build_provider(_settings(), exporter))
    client.get("/api/v1/dataplanes/eu/proxy/key/sk-live-123/reset_spend?token=sk-q", headers=minter.headers("eu"))
    spans: Final = exporter.get_finished_spans()
    (server,) = _by_kind(spans, SpanKind.SERVER)
    attributes: Final = repr(tuple(span.attributes for span in spans))
    assert "sk-live-123" not in attributes and "sk-q" not in attributes and "token=" not in attributes
    redacted_path: Final = "/api/v1/dataplanes/eu/proxy/key/sk-[redacted]/reset_spend"
    assert _attr(server, "http.target") == redacted_path
    assert _attr(server, "url.path") == redacted_path
    assert _attr(server, "url.query") == ""
    expected_url: Final = "http://testserver/api/v1/dataplanes/eu/proxy/key/sk-[redacted]/reset_spend"
    assert _attr(server, "http.url") == expected_url
    assert _attr(server, "url.full") == expected_url


def test_server_span_names_the_caller_but_not_the_email() -> None:
    exporter = InMemorySpanExporter()
    client, _, minter = rig({USER_INFO: [response(200, {})]}, build_provider(_settings(), exporter))
    client.get("/api/v1/dataplanes/eu/proxy/user/info", headers=minter.headers("eu"))
    (server,) = _by_kind(exporter.get_finished_spans(), SpanKind.SERVER)
    assert _attr(server, "enduser.id") == "alice" and _attr(server, "enduser.role") == ("viewer",)
    assert _attr(server, "litellm.control_plane.token_kind") == "session"
    assert "alice@x.test" not in repr(server.attributes)


def test_identity_attributes_can_be_turned_off() -> None:
    exporter = InMemorySpanExporter()
    client, _, minter = rig(
        {USER_INFO: [response(200, {})]}, build_provider(_settings(), exporter), otel_identity_attributes=False
    )
    client.get("/api/v1/dataplanes/eu/proxy/user/info", headers=minter.headers("eu"))
    (server,) = _by_kind(exporter.get_finished_spans(), SpanKind.SERVER)
    assert _attr(server, "enduser.id") is None and _attr(server, "litellm.control_plane.token_kind") == "session"


def test_each_retry_is_its_own_client_span_with_a_resend_count() -> None:
    exporter = InMemorySpanExporter()
    outcomes: list[httpx.Response | Exception] = [httpx.ConnectError("boom"), response(503), response(200, {})]
    client, _, minter = rig({USER_INFO: outcomes}, build_provider(_settings(), exporter))
    reply = client.get("/api/v1/dataplanes/eu/proxy/user/info", headers=minter.headers("eu"))
    assert reply.status_code == 200, reply.text
    attempts = _by_kind(exporter.get_finished_spans(), SpanKind.CLIENT)
    assert [_attr(span, "http.request.resend_count") for span in attempts] == [None, 1, 2]
    assert [span.status.status_code for span in attempts] == [StatusCode.ERROR, StatusCode.ERROR, StatusCode.UNSET]
    assert [_attr(span, "error.type") for span in attempts] == ["ConnectError", "503", None]
    assert len({_ctx(span).trace_id for span in attempts}) == 1


def test_health_is_not_traced() -> None:
    exporter = InMemorySpanExporter()
    client, _, _ = rig({}, build_provider(_settings(), exporter))
    assert client.get("/healthz").status_code == 200
    assert exporter.get_finished_spans() == ()


def test_without_a_provider_nothing_is_recorded_and_no_trace_header_is_sent() -> None:
    client, transport, minter = rig({USER_INFO: [response(200, {})]}, otel_enabled=False)
    assert client.get("/api/v1/dataplanes/eu/proxy/user/info", headers=minter.headers("eu")).status_code == 200
    _, _, headers = transport.calls[0]
    assert headers is not None and "traceparent" not in headers


@pytest.mark.parametrize(
    "env_name",
    (
        "OTEL_ENDPOINT",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_TRACES_ENDPOINT",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    ),
)
def test_tracing_enables_for_each_endpoint_alias(env_name: str) -> None:
    assert tracing_enabled(_settings(), {env_name: "http://collector:4318"}) is True


def test_tracing_is_disabled_without_an_endpoint() -> None:
    assert tracing_enabled(_settings(), {}) is False


def test_explicitly_disabled_tracing_wins_over_an_endpoint() -> None:
    assert tracing_enabled(_settings(otel_enabled=False), {"OTEL_ENDPOINT": "http://collector:4318"}) is False


def test_explicitly_enabled_tracing_needs_no_endpoint() -> None:
    assert tracing_enabled(_settings(otel_enabled=True), {}) is True


def test_disabled_telemetry_has_no_provider_to_instrument_with() -> None:
    assert build_telemetry(_settings(otel_enabled=False)).provider is None


def test_app_shutdown_flushes_the_server_span() -> None:
    exporter: Final = InMemorySpanExporter()
    tracer_provider: Final = TracerProvider()
    tracer_provider.add_span_processor(BatchSpanProcessor(exporter, schedule_delay_millis=60_000))
    client, _, minter = rig({USER_INFO: [response(200, {"ok": True})]}, tracer_provider)
    with client:
        reply: Final = client.get(
            "/api/v1/dataplanes/eu/proxy/user/info",
            headers=minter.headers("eu"),
        )
    assert reply.status_code == 200, reply.text
    server_spans: Final = _by_kind(exporter.get_finished_spans(), SpanKind.SERVER)
    assert tuple(span.name for span in server_spans) == (PROXY_ROUTE,)
