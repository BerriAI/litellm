import asyncio
import importlib.abc
import sys
from collections.abc import Iterator, Mapping, Sequence
from importlib.machinery import ModuleSpec
from types import ModuleType
from typing import Final

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.trace import Span, SpanKind, StatusCode, format_trace_id, use_span
from opentelemetry.trace.span import SpanContext

from litellm.proxy.control_plane.common.tracing import (
    ClientSpan,
    build_provider,
    build_telemetry,
    redact_server_span,
    tracing_enabled,
)
from litellm.proxy.control_plane.settings import ControlPlaneSettings
from tests.unit.proxy.control_plane.support import DATAPLANES, MASTER_KEY, Licensed, response, rig

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


def _recording_server_span(attributes: Mapping[str, str] | None = None) -> tuple[Span, InMemorySpanExporter]:
    exporter: Final = InMemorySpanExporter()
    span: Final = build_provider(_settings(), exporter).get_tracer("test").start_span("server")
    if attributes is not None:
        for key, value in attributes.items():
            span.set_attribute(key, value)
    return span, exporter


def test_one_proxy_call_is_one_trace_from_server_span_to_dataplane_header() -> None:
    exporter: Final = InMemorySpanExporter()
    client, transport, minter = rig({USER_INFO: [response(200, {"ok": True})]}, build_provider(_settings(), exporter))
    reply: Final = client.get("/api/v1/dataplanes/eu/proxy/user/info?secret=1", headers=minter.headers("eu"))
    assert reply.status_code == 200, reply.text
    spans: Final = exporter.get_finished_spans()
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
    exporter: Final = InMemorySpanExporter()
    client, _, minter = rig({USER_INFO: [response(200, {})]}, build_provider(_settings(), exporter))
    client.get("/api/v1/dataplanes/eu/proxy/user/info?token=sk-leak", headers=minter.headers("eu"))
    (dataplane_call,) = _by_kind(exporter.get_finished_spans(), SpanKind.CLIENT)
    attributes: Final = dict(dataplane_call.attributes or {})
    assert attributes["peer.service"] == "eu" and attributes["server.address"] == "eu.test"
    assert attributes["url.path"] == "/user/info" and attributes["http.response.status_code"] == 200
    assert attributes["litellm.control_plane.credential"] == "service_token"
    assert "sk-leak" not in repr(attributes) and "url.full" not in attributes and "url.query" not in attributes


def test_a_virtual_key_in_the_forwarded_path_never_reaches_the_span() -> None:
    exporter: Final = InMemorySpanExporter()
    url: Final = "https://eu.test/key/sk-live-123/reset_spend"
    client, _, minter = rig({url: [response(200, {})]}, build_provider(_settings(), exporter))
    client.get("/api/v1/dataplanes/eu/proxy/key/sk-live-123/reset_spend", headers=minter.headers("eu"))
    (dataplane_call,) = _by_kind(exporter.get_finished_spans(), SpanKind.CLIENT)
    assert dataplane_call.name == "GET /key/sk-[redacted]/reset_spend"
    assert _attr(dataplane_call, "url.path") == "/key/sk-[redacted]/reset_spend"
    assert "sk-live-123" not in repr(dataplane_call.attributes)


def test_the_server_span_keeps_neither_the_virtual_key_nor_the_query_string() -> None:
    exporter: Final = InMemorySpanExporter()
    url: Final = "https://eu.test/key/sk-live-123/reset_spend"
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


def test_server_span_redacts_double_encoded_virtual_keys() -> None:
    span, exporter = _recording_server_span(
        {
            "http.url": "https://instrumented.test/original?token=sk-live-123",
            "url.full": "https://instrumented.test/original?token=sk-live-123",
        }
    )
    redact_server_span(
        span,
        {
            "path": "/api/v1/dataplanes/eu/proxy/key/%73k-live-123/secret/reset_spend",
            "scheme": "https",
            "server": ("proxy.test", 443),
        },
    )
    span.end()
    (recorded,) = exporter.get_finished_spans()
    attributes: Final = recorded.attributes or {}
    redacted_path: Final = "/api/v1/dataplanes/eu/proxy/key/sk-[redacted]/reset_spend"
    expected_url: Final = f"https://instrumented.test{redacted_path}"
    assert attributes["http.target"] == redacted_path
    assert attributes["url.path"] == redacted_path
    assert attributes["http.url"] == expected_url
    assert attributes["url.full"] == expected_url
    assert "sk-live-123" not in repr(attributes)


def test_server_span_without_an_instrumented_url_leaves_full_url_unset() -> None:
    span, exporter = _recording_server_span()
    redact_server_span(span, {"path": "/healthz", "scheme": "https", "server": ("proxy.test", 443)})
    span.end()
    (recorded,) = exporter.get_finished_spans()
    attributes: Final = recorded.attributes or {}
    assert "http.url" not in attributes and "url.full" not in attributes


def test_server_span_drops_fragments_from_redacted_urls() -> None:
    instrumented_url: Final = "https://instrumented.test/original#%23&key=sk-live-123"
    span, exporter = _recording_server_span({"http.url": instrumented_url, "url.full": instrumented_url})
    redact_server_span(span, {"path": "/healthz"})
    span.end()
    (recorded,) = exporter.get_finished_spans()
    attributes: Final = recorded.attributes or {}
    expected_url: Final = "https://instrumented.test/healthz"
    assert attributes["http.url"] == expected_url
    assert attributes["url.full"] == expected_url
    assert "key=" not in attributes["http.url"] and "key=" not in attributes["url.full"]


def test_double_encoded_key_is_redacted_on_the_real_server_span() -> None:
    exporter: Final = InMemorySpanExporter()
    target: Final = "https://eu.test/key/sk-live-123/reset_spend"
    client, _, minter = rig({target: [response(200, {})]}, build_provider(_settings(), exporter))
    reply: Final = client.get(
        "/api/v1/dataplanes/eu/proxy/key/%2573k-live-123/reset_spend",
        headers=minter.headers("eu"),
    )
    assert reply.status_code == 200, reply.text
    (server,) = _by_kind(exporter.get_finished_spans(), SpanKind.SERVER)
    assert "k-live-123" not in repr(server.attributes)


@pytest.mark.parametrize("error", [ValueError("boom"), asyncio.CancelledError()])
def test_client_span_records_exceptions(error: BaseException) -> None:
    exporter: Final = InMemorySpanExporter()
    span: Final = build_provider(_settings(), exporter).get_tracer("test").start_span("client")
    with pytest.raises(type(error)):
        with ClientSpan(span, use_span(span, end_on_exit=True)):
            raise error
    (recorded,) = exporter.get_finished_spans()
    assert recorded.status.status_code == StatusCode.ERROR
    assert _attr(recorded, ERROR_TYPE) == f"{type(error).__module__}.{type(error).__qualname__}"
    if isinstance(error, asyncio.CancelledError):
        assert recorded.events == ()
    else:
        assert tuple(event.name for event in recorded.events) == ("exception",)


def test_server_instrumentation_redacts_captured_request_headers() -> None:
    environ: Final = {
        "OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_SERVER_REQUEST": ".*",
        "OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_SANITIZE_FIELDS": "X-Request-Id",
    }
    exporter: Final = InMemorySpanExporter()
    provider: Final = build_provider(_settings(), exporter)
    telemetry: Final = build_telemetry(_settings(otel_enabled=True), provider, environ=environ)
    app: Final = FastAPI()

    @app.get("/headers")
    def headers() -> dict[str, bool]:
        return {"ok": True}

    telemetry.instrument(app)
    response: Final = TestClient(app).get(
        "/headers",
        headers={
            "Authorization": "Bearer auth-secret",
            "Cookie": "session=cookie-secret",
            "X-Api-Key": "api-key-secret",
            "X-Request-Id": "request-id-secret",
        },
    )
    assert response.status_code == 200
    (server,) = _by_kind(exporter.get_finished_spans(), SpanKind.SERVER)
    attributes: Final = server.attributes or {}
    header_attributes: Final = (
        "http.request.header.authorization",
        "http.request.header.cookie",
        "http.request.header.x_api_key",
        "http.request.header.x_request_id",
    )
    assert all("[REDACTED]" in repr(attributes.get(name)) for name in header_attributes)
    assert all(
        secret not in repr(attributes)
        for secret in ("auth-secret", "cookie-secret", "api-key-secret", "request-id-secret")
    )


@pytest.fixture
def without_otel_sdk() -> Iterator[None]:
    modules: Final = sys.modules.copy()
    meta_path: Final = tuple(sys.meta_path)

    class BlockSdk(importlib.abc.MetaPathFinder):
        def find_spec(
            self, fullname: str, path: Sequence[str] | None, target: ModuleType | None = None
        ) -> ModuleSpec | None:
            if fullname == "opentelemetry.sdk" or fullname.startswith("opentelemetry.sdk."):
                raise ImportError(fullname)
            return None

    sys.meta_path = [BlockSdk(), *meta_path]
    for name in tuple(sys.modules):
        if (
            name == "opentelemetry.sdk"
            or name.startswith("opentelemetry.sdk.")
            or name == "litellm.proxy.control_plane"
            or name.startswith("litellm.proxy.control_plane.")
            or name == "litellm.integrations.otel.plumbing.context"
        ):
            sys.modules.pop(name)
    try:
        yield
    finally:
        sys.modules.clear()
        sys.modules.update(modules)
        sys.meta_path = list(meta_path)


def test_control_plane_app_builds_without_the_otel_sdk_when_tracing_is_disabled(
    without_otel_sdk: None,
) -> None:
    from litellm.proxy.control_plane.app import create_app
    from litellm.proxy.control_plane.container import build_container

    settings: Final = _settings(otel_enabled=False)
    app: Final = create_app(build_container(settings), Licensed())
    reply: Final = TestClient(app).get("/healthz")
    assert reply.status_code == 200, reply.text
    assert reply.json() == {"status": "ok"}


def test_inbound_baggage_cannot_stamp_control_plane_spans_and_traceparent_is_honored() -> None:
    exporter: Final = InMemorySpanExporter()
    client, transport, minter = rig({USER_INFO: [response(200, {})]}, build_provider(_settings(), exporter))
    trace_id: Final = "1234567890abcdef1234567890abcdef"
    span_id: Final = "0123456789abcdef"
    reply: Final = client.get(
        "/api/v1/dataplanes/eu/proxy/user/info",
        headers={
            **minter.headers("eu"),
            "baggage": "litellm.team.id=spoofed",
            "traceparent": f"00-{trace_id}-{span_id}-01",
        },
    )
    assert reply.status_code == 200
    spans: Final = exporter.get_finished_spans()
    (server,) = _by_kind(spans, SpanKind.SERVER)
    (dataplane_call,) = _by_kind(spans, SpanKind.CLIENT)
    assert format_trace_id(_ctx(server).trace_id) == trace_id
    assert _attr(server, "litellm.team.id") is None
    assert _attr(dataplane_call, "litellm.team.id") is None
    _, _, outbound_headers = transport.calls[0]
    assert outbound_headers is not None and outbound_headers["traceparent"].split("-")[1] == trace_id


def test_server_span_names_the_caller_but_not_the_email() -> None:
    exporter: Final = InMemorySpanExporter()
    client, _, minter = rig({USER_INFO: [response(200, {})]}, build_provider(_settings(), exporter))
    client.get("/api/v1/dataplanes/eu/proxy/user/info", headers=minter.headers("eu"))
    (server,) = _by_kind(exporter.get_finished_spans(), SpanKind.SERVER)
    assert _attr(server, "enduser.id") == "alice" and _attr(server, "enduser.role") == ("viewer",)
    assert _attr(server, "litellm.control_plane.token_kind") == "session"
    assert "alice@x.test" not in repr(server.attributes)


def test_identity_attributes_can_be_turned_off() -> None:
    exporter: Final = InMemorySpanExporter()
    client, _, minter = rig(
        {USER_INFO: [response(200, {})]}, build_provider(_settings(), exporter), otel_identity_attributes=False
    )
    client.get("/api/v1/dataplanes/eu/proxy/user/info", headers=minter.headers("eu"))
    (server,) = _by_kind(exporter.get_finished_spans(), SpanKind.SERVER)
    assert _attr(server, "enduser.id") is None and _attr(server, "litellm.control_plane.token_kind") == "session"


def test_each_retry_is_its_own_client_span_with_a_resend_count() -> None:
    exporter: Final = InMemorySpanExporter()
    outcomes: Final = [httpx.ConnectError("boom"), response(503), response(200, {})]
    client, _, minter = rig({USER_INFO: outcomes}, build_provider(_settings(), exporter))
    reply: Final = client.get("/api/v1/dataplanes/eu/proxy/user/info", headers=minter.headers("eu"))
    assert reply.status_code == 200, reply.text
    attempts: Final = _by_kind(exporter.get_finished_spans(), SpanKind.CLIENT)
    assert [_attr(span, "http.request.resend_count") for span in attempts] == [None, 1, 2]
    assert [span.status.status_code for span in attempts] == [StatusCode.ERROR, StatusCode.ERROR, StatusCode.UNSET]
    assert [_attr(span, "error.type") for span in attempts] == ["ConnectError", "503", None]
    assert len({_ctx(span).trace_id for span in attempts}) == 1


def test_client_span_marks_a_4xx_reply_as_an_error() -> None:
    exporter: Final = InMemorySpanExporter()
    client, _, minter = rig({USER_INFO: [response(400)]}, build_provider(_settings(), exporter))

    reply: Final = client.get("/api/v1/dataplanes/eu/proxy/user/info", headers=minter.headers("eu"))

    assert reply.status_code == 400
    (dataplane_call,) = _by_kind(exporter.get_finished_spans(), SpanKind.CLIENT)
    assert dataplane_call.status.status_code == StatusCode.ERROR
    assert _attr(dataplane_call, "error.type") == "400"


def test_health_is_not_traced() -> None:
    exporter: Final = InMemorySpanExporter()
    client, _, _ = rig({}, build_provider(_settings(), exporter))
    assert client.get("/healthz").status_code == 200
    assert exporter.get_finished_spans() == ()


def test_server_exclusions_match_only_the_exact_url_path() -> None:
    exporter: Final = InMemorySpanExporter()
    docs_url: Final = "https://docs-prod.test/user/info"
    dataplanes: Final = {
        **DATAPLANES,
        "docs-prod": {"url": "https://docs-prod.test", "service_token": {"token": "sk-docs"}},
    }
    client, _, minter = rig(
        {docs_url: [response(200, {"ok": True})]},
        build_provider(_settings(), exporter),
        dataplanes=dataplanes,
    )

    assert client.get("/api/v1/docs").status_code == 200
    assert _by_kind(exporter.get_finished_spans(), SpanKind.SERVER) == []
    assert client.get("/api/v1/openapi.json").status_code == 200
    assert _by_kind(exporter.get_finished_spans(), SpanKind.SERVER) == []
    reply: Final = client.get(
        "/api/v1/dataplanes/docs-prod/proxy/user/info?probe=1",
        headers=minter.headers("docs-prod"),
    )

    assert reply.status_code == 200, reply.text
    (server,) = _by_kind(exporter.get_finished_spans(), SpanKind.SERVER)
    assert server.name == PROXY_ROUTE
    assert _attr(server, "http.url") == "http://testserver/api/v1/dataplanes/docs-prod/proxy/user/info"


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


def test_explicit_enable_without_an_endpoint_exports_spans_to_console(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    endpoint_aliases: Final = (
        "OTEL_EXPORTER",
        "OTEL_EXPORTER_OTLP_PROTOCOL",
        "OTEL_ENDPOINT",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_TRACES_ENDPOINT",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "OTEL_LOGS_ENDPOINT",
        "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
    )
    for alias in endpoint_aliases:
        monkeypatch.delenv(alias, raising=False)

    telemetry: Final = build_telemetry(_settings(otel_enabled=True), environ={})
    assert telemetry.provider is not None
    telemetry.provider.get_tracer("test").start_span("console-default").end()
    telemetry.shutdown()

    assert "console-default" in capfd.readouterr().out


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
