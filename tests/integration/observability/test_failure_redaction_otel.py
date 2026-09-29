import json
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from google.protobuf.json_format import MessageToDict
from integration._support.client import Gateway, JsonValue, eventually, gateway_from_environment, object_value
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest


def _provider(request: Request) -> Reply:
    try:
        body: Final = json.loads(request.body)
    except json.JSONDecodeError:
        return Reply(status=404, body=b"{}")
    text: Final = body["messages"][-1]["content"]
    return Reply(
        status=400,
        body=json.dumps(
            {"error": {"type": "invalid_request_error", "message": f"Unsupported content: {text}"}}
        ).encode(),
    )


def _decode(body: bytes) -> dict[str, JsonValue]:
    if body[:1] == b"{":
        return json.loads(body)
    request: Final = ExportTraceServiceRequest()
    request.ParseFromString(body)
    return object_value(MessageToDict(request))


@dataclass(frozen=True, slots=True)
class Spans:
    wire: Wire
    batches: list[Request]

    def all(self) -> tuple[dict[str, JsonValue], ...]:
        self.batches.extend(self.wire.drain())
        return tuple(
            span
            for batch in self.batches
            for resource in _decode(batch.body).get("resourceSpans", ())
            for scope in resource.get("scopeSpans", ())
            for span in scope.get("spans", ())
        )

    def named(self, name: str, model: str) -> tuple[dict[str, JsonValue], ...]:
        return tuple(span for span in self.all() if span.get("name") == name and model in json.dumps(span))

    def in_trace(self, name: str, trace_id: str) -> tuple[dict[str, JsonValue], ...]:
        return tuple(span for span in self.all() if span.get("name") == name and span.get("traceId") == trace_id)


def _span_attributes(span: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        str(attribute["key"]): object_value(attribute["value"]).get("stringValue")
        or object_value(attribute["value"]).get("intValue")
        for attribute in span.get("attributes", ())
        if isinstance(attribute, dict)
    }


def _exception_events(span: Mapping[str, JsonValue]) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        object_value(event)
        for event in span.get("events", ())
        if isinstance(event, dict) and event.get("name") == "exception"
    )


def _exception_texts(span: Mapping[str, JsonValue]) -> str:
    return json.dumps(_exception_events(span))


def _error_attribute(span: Mapping[str, JsonValue]) -> str:
    attributes: Final = _span_attributes(span)
    return str(attributes.get("error.message", ""))


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: Gateway
    provider: Wire
    sink: Spans


@contextmanager
def _otel_rig(root: Path, provider: Wire, sink: Wire, v2: bool, global_on: bool) -> Iterator[Rig]:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    settings: Final[dict[str, JsonValue]] = {"callbacks": ["otel"]}
    if global_on:
        settings["turn_off_message_logging"] = True
    config["litellm_settings"].update(settings)
    if v2:
        config["callback_settings"] = {
            "otel": {"exporter": "http/json", "endpoint": sink.url, "mapper_names": ["genai"]}
        }
    path: Final = root / "otel_failure.yaml"
    path.write_text(yaml.safe_dump(config))
    env: Final = (
        {"LITELLM_OTEL_V2": "1", "OTEL_BSP_SCHEDULE_DELAY": "300"}
        if v2
        else {
            "OTEL_EXPORTER": "http/json",
            "OTEL_EXPORTER_OTLP_ENDPOINT": sink.url,
            "OTEL_BSP_SCHEDULE_DELAY": "300",
        }
    )
    with (
        gateway_from_environment() as gateway,
        owned_proxy(gateway, root, env, config=path, workers=2) as proxy,
    ):
        yield Rig(proxy, provider, Spans(sink, []))  # mutable-ok: drain consumes, polls keep earlier batches


@pytest.fixture(scope="module")
def provider() -> Iterator[Wire]:
    with wire_server(_provider) as wire:
        yield wire


@pytest.fixture(scope="module")
def sink() -> Iterator[Wire]:
    with wire_server(lambda _: Reply()) as wire:
        yield wire


@pytest.fixture(scope="module")
def rig_v1_on(tmp_path_factory: pytest.TempPathFactory, provider: Wire, sink: Wire) -> Iterator[Rig]:
    with _otel_rig(tmp_path_factory.mktemp("otel_v1_on"), provider, sink, v2=False, global_on=True) as booted:
        yield booted


@pytest.fixture(scope="module")
def rig_v2_on(tmp_path_factory: pytest.TempPathFactory, provider: Wire, sink: Wire) -> Iterator[Rig]:
    with _otel_rig(tmp_path_factory.mktemp("otel_v2_on"), provider, sink, v2=True, global_on=True) as booted:
        yield booted


@pytest.fixture(scope="module")
def rig_v2_off(tmp_path_factory: pytest.TempPathFactory, provider: Wire, sink: Wire) -> Iterator[Rig]:
    with _otel_rig(tmp_path_factory.mktemp("otel_v2_off"), provider, sink, v2=True, global_on=False) as booted:
        yield booted


def _secret() -> str:
    return "otel-secret-" + uuid.uuid4().hex


def _fail(rig: Rig, model: str, secret: str, **kwargs: JsonValue) -> httpx.Response:
    headers: Final = kwargs.pop("headers", None)
    key: Final = kwargs.pop("key", None)
    return rig.proxy.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": secret}], **kwargs},
        headers=headers if isinstance(headers, dict) else None,
        key=key if isinstance(key, str) else None,
    )


def _llm_span(rig: Rig, call_id: str) -> dict[str, JsonValue]:
    def found() -> tuple[dict[str, JsonValue], ...]:
        return tuple(
            span
            for span in rig.sink.all()
            if call_id in json.dumps(span) and str(span.get("name", "")).startswith(("chat ", "litellm_request"))
        )

    return eventually(found, lambda values: len(values) >= 1, seconds=260)[0]


_SERVER_SPAN_NAMES: Final = ("Received Proxy Server Request", "POST /v1/chat/completions", "POST /v1/messages")


def _server_span(rig: Rig, call_id: str) -> dict[str, JsonValue]:
    trace_id: Final = str(_llm_span(rig, call_id)["traceId"])

    def found() -> tuple[dict[str, JsonValue], ...]:
        return tuple(
            span
            for span in rig.sink.all()
            if span.get("traceId") == trace_id and str(span.get("name", "")) in _SERVER_SPAN_NAMES
        )

    spans: Final = eventually(found, lambda values: len(values) >= 1, seconds=60, return_last_on_timeout=True)
    assert spans, [(span.get("name"), span.get("traceId")) for span in rig.sink.all()]
    return spans[0]


def _span_for(rig: Rig, name: str, model: str) -> dict[str, JsonValue]:
    spans: Final = eventually(lambda: rig.sink.named(name, model), lambda values: len(values) >= 1, seconds=260)
    return spans[0]


def _auth_exception_span(rig: Rig) -> dict[str, JsonValue]:
    def found() -> tuple[dict[str, JsonValue], ...]:
        return tuple(
            span for span in rig.sink.all() if str(span.get("name", "")).startswith("auth") and _exception_events(span)
        )

    return eventually(found, lambda values: len(values) >= 1, seconds=260)[0]


# C6: OTEL v1 failure span redaction under global on
@pytest.mark.timeout(320)
def test_c6_v1_provider_error_spans_redacted(rig_v1_on: Rig) -> None:
    secret: Final = _secret()
    with rig_v1_on.proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=rig_v1_on.provider.url + "/v1", api_key="synthetic-provider-key")
        response: Final = _fail(rig_v1_on, model, secret)
        assert response.status_code == 400, response.text
        call_id: Final = response.headers["x-litellm-call-id"]
        assert any(secret.encode() in request.body for request in rig_v1_on.provider.drain())
        server: Final = _server_span(rig_v1_on, call_id)
        assert _error_attribute(server) == "redacted-by-litellm", _span_attributes(server)
        assert secret not in _exception_texts(server), _exception_texts(server)[:600]
        request_span: Final = _llm_span(rig_v1_on, call_id)
        assert _error_attribute(request_span) == "redacted-by-litellm", _span_attributes(request_span)
        assert secret not in _exception_texts(request_span), _exception_texts(request_span)[:600]


# C7: OTEL v2 request + server spans redacted under global on
@pytest.mark.timeout(320)
def test_c7_v2_provider_error_spans_redacted(rig_v2_on: Rig) -> None:
    secret: Final = _secret()
    with rig_v2_on.proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=rig_v2_on.provider.url + "/v1", api_key="synthetic-provider-key")
        response: Final = _fail(rig_v2_on, model, secret)
        assert response.status_code == 400, response.text
        call_id: Final = response.headers["x-litellm-call-id"]
        server: Final = _server_span(rig_v2_on, call_id)
        assert _error_attribute(server) == "redacted-by-litellm", _span_attributes(server)
        assert secret not in _exception_texts(server), _exception_texts(server)[:600]


# C8: v2 server span restamp keeps request opt-in under global off
@pytest.mark.timeout(320)
def test_c8_v2_header_opt_in_restamped_server_span_redacted(rig_v2_off: Rig) -> None:
    secret: Final = _secret()
    with rig_v2_off.proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=rig_v2_off.provider.url + "/v1", api_key="synthetic-provider-key")
        response: Final = _fail(rig_v2_off, model, secret, headers={"x-litellm-enable-message-redaction": "true"})
        assert response.status_code == 400, response.text
        call_id: Final = response.headers["x-litellm-call-id"]
        server: Final = _server_span(rig_v2_off, call_id)
        assert _error_attribute(server) == "redacted-by-litellm", _span_attributes(server)
        assert secret not in _exception_texts(server), _exception_texts(server)[:600]


# C9: v2 server span keeps permitted opt-out raw under global on
@pytest.mark.timeout(320)
def test_c9_v2_permitted_opt_out_keeps_server_span_raw(rig_v2_on: Rig) -> None:
    secret: Final = _secret()
    with rig_v2_on.proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=rig_v2_on.provider.url + "/v1", api_key="synthetic-provider-key")
        key: Final = scenario.key(models=[model], metadata={"allow_client_message_redaction_opt_out": True})
        response: Final = _fail(
            rig_v2_on, model, secret, key=key, headers={"litellm-disable-message-redaction": "true"}
        )
        assert response.status_code == 400, response.text
        call_id: Final = response.headers["x-litellm-call-id"]
        server: Final = _server_span(rig_v2_on, call_id)
        assert secret in _error_attribute(server), _span_attributes(server)


# C10: v2 auth phase span honors request opt-in under global off
@pytest.mark.timeout(320)
def test_c10_v2_auth_span_honors_header_opt_in(rig_v2_off: Rig) -> None:
    with rig_v2_off.proxy.scenario() as scenario:
        allowed: Final = scenario.model(api_base=rig_v2_off.provider.url + "/v1", api_key="synthetic-provider-key")
        denied: Final = scenario.model(api_base=rig_v2_off.provider.url + "/v1", api_key="synthetic-provider-key")
        key: Final = scenario.key(models=[allowed])
        response: Final = rig_v2_off.proxy.request(
            "POST",
            "/v1/chat/completions",
            {"model": denied, "messages": [{"role": "user", "content": "hi"}]},
            key=key,
            headers={"x-litellm-enable-message-redaction": "true"},
        )
        assert response.status_code in (400, 401, 403, 404), response.text
        auth: Final = _auth_exception_span(rig_v2_off)
        assert allowed not in _exception_texts(auth), _exception_texts(auth)[:600]
        assert "redacted-by-litellm" in _exception_texts(auth), _exception_texts(auth)[:600]


# C11: v2 auth phase span redacts under global on even with unpermitted disable header
@pytest.mark.timeout(320)
def test_c11_v2_auth_span_redacts_under_global_on(rig_v2_on: Rig) -> None:
    with rig_v2_on.proxy.scenario() as scenario:
        allowed: Final = scenario.model(api_base=rig_v2_on.provider.url + "/v1", api_key="synthetic-provider-key")
        denied: Final = scenario.model(api_base=rig_v2_on.provider.url + "/v1", api_key="synthetic-provider-key")
        key: Final = scenario.key(models=[allowed])
        response: Final = rig_v2_on.proxy.request(
            "POST",
            "/v1/chat/completions",
            {"model": denied, "messages": [{"role": "user", "content": "hi"}]},
            key=key,
            headers={"litellm-disable-message-redaction": "true"},
        )
        assert response.status_code in (400, 401, 403, 404), response.text
        auth: Final = _auth_exception_span(rig_v2_on)
        assert allowed not in _exception_texts(auth), _exception_texts(auth)[:600]
        assert "redacted-by-litellm" in _exception_texts(auth), _exception_texts(auth)[:600]
