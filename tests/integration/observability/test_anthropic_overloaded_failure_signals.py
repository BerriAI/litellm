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
from integration._support.client import Gateway, JsonValue, eventually, gateway_from_environment
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
_OVERLOADED: Final = Reply(status=529, body=error_body(529, "Overloaded"))
_SERVER: Final = 2

AuditConfigWriter = Callable[[Path, Mapping[str, JsonValue]], Path]


@dataclass(frozen=True, slots=True)
class _Rig:
    proxy: Gateway
    wire: Wire
    spans: str


def _respond(request: Request) -> Reply:
    assert request.method == "POST", request
    if unquote(request.target).startswith("/model/"):
        return Reply(status=503, body=_BEDROCK_503)
    return _OVERLOADED


def _config(directory: Path, tracing: Path, wire: Wire) -> Path:
    configuration: Final = yaml.safe_load(tracing.read_text())
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
    )
    path: Final = directory / "anthropic-overloaded-failure-signals.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                **configuration,
                "model_list": [{"model_name": name, "litellm_params": params} for name, params in deployments],
                "router_settings": {**configuration.get("router_settings", {}), "num_retries": 0},
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
    tracing: Final = otel_audit_config(directory, {"callbacks": ["otel", "prometheus"]})
    with (
        wire_server(_respond) as wire,
        gateway_from_environment() as shared,
        owned_proxy(
            shared,
            directory,
            {
                "OTEL_EXPORTER": "http/json",
                "OTEL_ENDPOINT": audit_sinks.operator,
                "PROMETHEUS_MULTIPROC_DIR": str(multiprocess),
            },
            config=_config(directory, tracing, wire),
            workers=2,
        ) as owned,
    ):
        yield _Rig(owned, wire, audit_sinks.operator)


def _send(rig: _Rig, route: str, model: str, trace_id: str | None = None) -> httpx.Response:
    traceparent: Final = {} if trace_id is None else {"traceparent": f"00-{trace_id}-{uuid.uuid4().hex[:16]}-01"}
    return rig.proxy.request(
        "POST",
        route,
        {
            "model": model,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": f"signals-{uuid.uuid4().hex}"}],
            "num_retries": 0,
        },
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
