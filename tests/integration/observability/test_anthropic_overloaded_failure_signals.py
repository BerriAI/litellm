import gzip
import json
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import unquote

import httpx
import pytest
import yaml
from integration._support.anthropic_sse import error_body
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    JsonValue,
    eventually,
    gateway_from_environment,
    object_value,
    string_value,
)
from integration._support.otlp_sink import Span, SpanSinks, recorded_spans, spans_for_trace
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.prometheus_series import Sample, scrape
from integration._support.wire import Reply, Request, Wire, wire_server

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

_MODEL: Final = "claude-sonnet-4-5-20250929"
_BEDROCK_MODEL: Final = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
_BEDROCK_503: Final = b'{"message": "Bedrock is unable to process your request."}'
_MESSAGES_ANTHROPIC: Final = "audit-signals-messages-anthropic"
_MESSAGES_BEDROCK: Final = "audit-signals-messages-bedrock"
_CHAT_ANTHROPIC: Final = "audit-signals-chat-anthropic"
_TRACED_MESSAGES: Final = "audit-signals-traced-messages"
_TRACED_CHAT: Final = "audit-signals-traced-chat"
_DATADOG_MESSAGES_ANTHROPIC: Final = "audit-signals-datadog-messages-anthropic"
_DATADOG_MESSAGES_BEDROCK: Final = "audit-signals-datadog-messages-bedrock"
_DATADOG_CHAT_ANTHROPIC: Final = "audit-signals-datadog-chat-anthropic"
_GUARDED_MESSAGES: Final = "audit-signals-guarded-messages"
_GUARDED_CHAT: Final = "audit-signals-guarded-chat"
_AKTO_DOWN: Final = "audit-signals-akto-down"
_AKTO_UNREACHABLE: Final = "Akto guardrail service unreachable"
_LATE_PASS_THROUGH_BACKEND: Final = "claude-late-pass-through"
_KEEPALIVE_SECONDS: Final = 0.2
_LATE_ANSWER_SECONDS: Final = 1.0
_OVERLOADED: Final = Reply(status=529, body=error_body(529, "Overloaded"))
_SERVER: Final = 2

AuditConfigWriter = Callable[[Path, Mapping[str, JsonValue]], Path]


@dataclass(frozen=True, slots=True)
class _Rig:
    proxy: Gateway
    wire: Wire
    spans: str
    datadog: Wire


def _respond(request: Request) -> Reply:
    assert request.method == "POST", request
    if unquote(request.target).startswith("/model/"):
        return Reply(status=503, body=_BEDROCK_503)
    if json.loads(request.body).get("model") == _LATE_PASS_THROUGH_BACKEND:
        time.sleep(_LATE_ANSWER_SECONDS)
    return _OVERLOADED


def _datadog_intake(request: Request) -> Reply:
    assert request.target == "/api/v2/logs", request.target
    return Reply(status=202, body=b"{}")


def _config(directory: Path, tracing: Path, wire: Wire) -> Path:
    configuration: Final = JSON_OBJECT.validate_python(yaml.safe_load(tracing.read_text()))
    anthropic: Final = {"model": f"anthropic/{_MODEL}", "api_base": wire.url, "api_key": "synthetic-anthropic-key"}
    bedrock: Final = {
        "model": f"bedrock/{_BEDROCK_MODEL}",
        "api_base": wire.url,
        "api_key": "synthetic-bedrock-bearer",
        "aws_region_name": "us-east-1",
    }
    deployments: Final = (
        (_MESSAGES_ANTHROPIC, anthropic),
        (_MESSAGES_BEDROCK, bedrock),
        (_CHAT_ANTHROPIC, anthropic),
        (_TRACED_MESSAGES, anthropic),
        (_TRACED_CHAT, anthropic),
        (_DATADOG_MESSAGES_ANTHROPIC, anthropic),
        (_DATADOG_MESSAGES_BEDROCK, bedrock),
        (_DATADOG_CHAT_ANTHROPIC, anthropic),
        (_GUARDED_MESSAGES, anthropic),
        (_GUARDED_CHAT, anthropic),
    )
    akto_down: Final = {
        "guardrail_name": _AKTO_DOWN,
        "litellm_params": {
            "guardrail": "akto",
            "mode": "pre_call",
            "akto_base_url": "http://127.0.0.1:9",
            "akto_api_key": "synthetic-akto-key",
            "unreachable_fallback": "fail_closed",
        },
    }
    path: Final = directory / "anthropic-overloaded-failure-signals.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                **configuration,
                "model_list": [{"model_name": name, "litellm_params": params} for name, params in deployments],
                "router_settings": {**object_value(configuration.get("router_settings", {})), "num_retries": 0},
                "litellm_settings": {
                    **object_value(configuration["litellm_settings"]),
                    "sse_keepalive_ping_interval_seconds": _KEEPALIVE_SECONDS,
                },
                "guardrails": [akto_down],
            }
        )
    )
    return path


@pytest.fixture(scope="module")
def rig(
    tmp_path_factory: pytest.TempPathFactory, audit_sinks: SpanSinks, otel_audit_config: AuditConfigWriter
) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("anthropic-overloaded-failure-signals")
    multiprocess: Final = directory / "prometheus"
    multiprocess.mkdir()
    tracing: Final = otel_audit_config(directory, {"callbacks": ["otel", "prometheus", "datadog"]})
    with (
        wire_server(_respond) as wire,
        wire_server(_datadog_intake) as datadog,
        gateway_from_environment() as shared,
        owned_proxy(
            shared,
            directory,
            {
                "OTEL_EXPORTER": "http/json",
                "OTEL_ENDPOINT": audit_sinks.operator,
                "PROMETHEUS_MULTIPROC_DIR": str(multiprocess),
                "DD_API_KEY": "synthetic-datadog-key",
                "DD_SITE": "datadog.invalid",
                "DD_BASE_URL": datadog.url,
                "DD_BATCH_SIZE": "1",
                "ANTHROPIC_API_BASE": wire.url,
                "ANTHROPIC_API_KEY": "synthetic-anthropic-key",
            },
            config=_config(directory, tracing, wire),
            workers=2,
        ) as owned,
    ):
        yield _Rig(owned, wire, audit_sinks.operator, datadog)


@pytest.fixture(scope="module")
def v2_rig(
    tmp_path_factory: pytest.TempPathFactory, audit_sinks: SpanSinks, otel_audit_config: AuditConfigWriter
) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("anthropic-overloaded-failure-signals-v2")
    tracing: Final = otel_audit_config(directory, {"callbacks": ["otel"]})
    with (
        wire_server(_respond) as wire,
        wire_server(_datadog_intake) as datadog,
        gateway_from_environment() as shared,
        owned_proxy(
            shared,
            directory,
            {"OTEL_EXPORTER": "http/json", "OTEL_ENDPOINT": audit_sinks.operator, "LITELLM_OTEL_V2": "1"},
            config=_config(directory, tracing, wire),
            workers=2,
        ) as owned,
    ):
        yield _Rig(owned, wire, audit_sinks.operator, datadog)


def _send(
    rig: _Rig,
    route: str,
    model: str,
    trace_id: str | None = None,
    *,
    key: str | None = None,
    extra: Mapping[str, JsonValue] | None = None,
) -> httpx.Response:
    traceparent: Final = {} if trace_id is None else {"traceparent": f"00-{trace_id}-{uuid.uuid4().hex[:16]}-01"}
    return rig.proxy.request(
        "POST",
        route,
        {
            "model": model,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": f"signals-{uuid.uuid4().hex}"}],
            "num_retries": 0,
            **(extra or {}),
        },
        key=key,
        headers={"anthropic-version": "2023-06-01", **traceparent},
    )


def _series(samples: Sequence[Sample], name: str, model: str) -> tuple[Sample, ...]:
    return tuple(sample for sample in samples if sample.name == name and sample.labels.get("requested_model") == model)


def _until(rig: _Rig, name: str, model: str) -> tuple[Sample, ...]:
    return eventually(lambda: _series(scrape(rig.proxy), name, model), bool, seconds=30)


@pytest.mark.parametrize(
    ("route", "model", "answered", "deployment_status", "deployment_class"),
    (
        pytest.param("/v1/messages", _MESSAGES_ANTHROPIC, 529, "500", "InternalServerError", id="messages-anthropic"),
        pytest.param("/v1/messages", _MESSAGES_BEDROCK, 529, "503", "ServiceUnavailableError", id="messages-bedrock"),
        pytest.param("/v1/chat/completions", _CHAT_ANTHROPIC, 500, "500", "InternalServerError", id="chat-anthropic"),
    ),
)
def test_the_failed_request_metric_counts_the_answered_status_and_the_deployment_metric_the_deployment_status(
    rig: _Rig, route: str, model: str, answered: int, deployment_status: str, deployment_class: str
) -> None:
    rig.wire.drain()
    response: Final = _send(rig, route, model)
    assert response.status_code == answered, response.text
    assert len(rig.wire.drain()) == 1
    failed: Final = _until(rig, "litellm_proxy_failed_requests_metric_total", model)
    assert [(sample.labels["exception_status"], sample.labels["route"], sample.value) for sample in failed] == [
        (str(answered), route, 1.0)
    ], failed
    deployment: Final = _until(rig, "litellm_deployment_failure_responses_total", model)
    assert [(sample.labels["exception_status"], sample.value) for sample in deployment] == [(deployment_status, 1.0)], (
        deployment
    )
    assert all(deployment_class in sample.labels["exception_class"] for sample in deployment), deployment


def _server_spans(rig: _Rig, since: int, trace_id: str) -> tuple[Span, ...]:
    traced: Final = spans_for_trace(recorded_spans(rig.spans, since)[1], trace_id)
    return tuple(span for span in traced if span["kind"] == _SERVER)


@pytest.mark.parametrize(
    ("route", "model", "answered"),
    (
        pytest.param("/v1/messages", _TRACED_MESSAGES, 529, id="messages"),
        pytest.param("/v1/chat/completions", _TRACED_CHAT, 500, id="chat"),
    ),
)
def test_the_server_span_carries_the_status_the_route_answered(
    rig: _Rig, route: str, model: str, answered: int
) -> None:
    rig.wire.drain()
    since, _ = recorded_spans(rig.spans)
    trace_id: Final = uuid.uuid4().hex
    response: Final = _send(rig, route, model, trace_id)
    assert response.status_code == answered, response.text
    assert len(rig.wire.drain()) == 1
    server: Final = eventually(
        lambda: _server_spans(rig, since, trace_id),
        lambda spans: any("error.code" in span["attributes"] for span in spans),
        seconds=40,
        return_last_on_timeout=True,
    )
    assert [
        (str(span["attributes"].get("http.response.status_code")), str(span["attributes"].get("error.code")))
        for span in server
    ] == [(str(answered), str(answered))], server


def _inflated(request: Request) -> bytes:
    return gzip.decompress(request.body) if request.headers.get("content-encoding") == "gzip" else request.body


def _datadog_events(batches: Sequence[Request]) -> Iterator[JsonValue]:
    for batch in batches:
        yield from json.loads(_inflated(batch))


def _datadog_messages(batches: Sequence[Request]) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        JSON_OBJECT.validate_json(string_value(object_value(event)["message"])) for event in _datadog_events(batches)
    )


def _keyed_by(message: Mapping[str, JsonValue], alias: str) -> bool:
    proxy_side: Final = message.get("user_api_key_dict")
    llm_side: Final = message.get("metadata")
    return any(isinstance(side, dict) and side.get("user_api_key_alias") == alias for side in (proxy_side, llm_side))


def _datadog_status_codes(rig: _Rig, alias: str, expected: int) -> tuple[tuple[str, str], ...]:
    batches: Final[list[Request]] = []  # mutable-ok: drain() consumes the queue, later polls must keep earlier batches

    def delivered() -> tuple[tuple[str, str], ...]:
        batches.extend(rig.datadog.drain())
        return tuple(
            sorted(
                ("proxy", str(message["status_code"]))
                if "user_api_key_dict" in message
                else ("llm", str(object_value(message["error_information"])["error_code"]))
                for message in _datadog_messages(batches)
                if _keyed_by(message, alias)
            )
        )

    return eventually(delivered, lambda codes: len(codes) >= expected, seconds=40, return_last_on_timeout=True)


@pytest.mark.parametrize(
    ("route", "model", "answered", "deployment_status"),
    (
        pytest.param("/v1/messages", _DATADOG_MESSAGES_ANTHROPIC, 529, "500", id="messages-anthropic"),
        pytest.param("/v1/messages", _DATADOG_MESSAGES_BEDROCK, 529, "503", id="messages-bedrock"),
        pytest.param("/v1/chat/completions", _DATADOG_CHAT_ANTHROPIC, 500, "500", id="chat-anthropic"),
    ),
)
def test_the_datadog_proxy_failure_log_carries_the_answered_status_and_the_llm_failure_log_the_deployment_status(
    rig: _Rig, route: str, model: str, answered: int, deployment_status: str
) -> None:
    rig.wire.drain()
    rig.datadog.drain()
    alias: Final = f"audit-signals-datadog-{uuid.uuid4().hex}"
    with rig.proxy.scenario() as scenario:
        response: Final = _send(rig, route, model, key=scenario.key(key_alias=alias, models=[model]))
        assert response.status_code == answered, response.text
        assert len(rig.wire.drain()) == 1
        assert _datadog_status_codes(rig, alias, 2) == (("llm", deployment_status), ("proxy", str(answered)))


@pytest.mark.parametrize(
    ("route", "model", "answered"),
    (
        pytest.param("/v1/messages", _GUARDED_MESSAGES, 529, id="messages"),
        pytest.param("/v1/chat/completions", _GUARDED_CHAT, 503, id="chat"),
    ),
)
def test_a_fail_closed_guardrail_outage_is_answered_and_recorded_with_the_routes_status(
    rig: _Rig, route: str, model: str, answered: int
) -> None:
    rig.wire.drain()
    rig.datadog.drain()
    alias: Final = f"audit-signals-guarded-{uuid.uuid4().hex}"
    with rig.proxy.scenario() as scenario:
        response: Final = _send(
            rig, route, model, key=scenario.key(key_alias=alias, models=[model]), extra={"guardrails": [_AKTO_DOWN]}
        )
        assert response.status_code == answered, response.text
        assert _AKTO_UNREACHABLE in response.text, response.text
        assert rig.wire.drain() == ()
        failed: Final = _until(rig, "litellm_proxy_failed_requests_metric_total", model)
        assert [(sample.labels["exception_status"], sample.labels["route"], sample.value) for sample in failed] == [
            (str(answered), route, 1.0)
        ], failed
        assert _datadog_status_codes(rig, alias, 2) == (("llm", str(answered)), ("proxy", str(answered)))


@pytest.mark.parametrize(
    ("route", "model", "answered"),
    (
        pytest.param("/v1/messages", _TRACED_MESSAGES, 529, id="messages-anthropic"),
        pytest.param("/v1/messages", _MESSAGES_BEDROCK, 529, id="messages-bedrock"),
        pytest.param("/v1/chat/completions", _TRACED_CHAT, 500, id="chat-anthropic"),
    ),
)
def test_the_otel_v2_server_span_carries_the_status_the_route_answered(
    v2_rig: _Rig, route: str, model: str, answered: int
) -> None:
    v2_rig.wire.drain()
    since, _ = recorded_spans(v2_rig.spans)
    trace_id: Final = uuid.uuid4().hex
    response: Final = _send(v2_rig, route, model, trace_id)
    assert response.status_code == answered, response.text
    assert len(v2_rig.wire.drain()) == 1
    server: Final = eventually(
        lambda: _server_spans(v2_rig, since, trace_id),
        lambda spans: any("litellm.provider.error.code" in span["attributes"] for span in spans),
        seconds=40,
        return_last_on_timeout=True,
    )
    assert [str(span["attributes"].get("litellm.provider.error.code")) for span in server] == [str(answered)], server


def test_an_anthropic_pass_through_stream_that_fails_after_the_keepalive_opened_relays_the_upstream_body(
    rig: _Rig,
) -> None:
    rig.wire.drain()
    response: Final = rig.proxy.request(
        "POST",
        "/anthropic/v1/messages",
        {
            "model": _LATE_PASS_THROUGH_BACKEND,
            "max_tokens": 16,
            "stream": True,
            "messages": [{"role": "user", "content": f"signals-{uuid.uuid4().hex}"}],
        },
        headers={"anthropic-version": "2023-06-01"},
    )
    assert response.status_code == 200, response.text
    pings, _, relayed = response.text.rpartition("\n\n")
    assert pings and set(pings.split("\n")) <= {"", ": ping"}, response.text
    assert relayed == _OVERLOADED.body.decode(), response.text
    assert [unquote(request.target) for request in rig.wire.drain()] == ["/v1/messages"]
