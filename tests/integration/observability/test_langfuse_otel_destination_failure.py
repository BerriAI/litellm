"""The operator's collector records a failed request the same way whether or not the key or team has its own
``langfuse_otel`` destination, and no key or team destination receives the operator's stack trace, when ``otel`` and
``langfuse_otel`` are both listed and the operator has no Langfuse keys"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterator, Mapping
from typing import Final, Literal

import pytest
from integration._support.client import Gateway, gateway_from_environment, held
from integration._support.otlp_sink import (
    AuditConfigWriter,
    Span,
    SpanSinks,
    langfuse_callback_variables,
    recorded_spans,
    spans_for_trace,
)
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

SERVER_SPAN_KIND: Final = 2
FAILURE_SPAN: Final = "Failed Proxy Server Request"
STACK_TRACE: Final = "error.stack_trace"
ERROR_ATTRIBUTES: Final = ("error.type", "error.code", "http.response.status_code", "metadata.user_api_key_team_id")


def _upstream_error(_request: Request) -> Reply:
    return Reply(
        status=500, body=json.dumps({"error": {"message": "upstream failed", "type": "server_error"}}).encode()
    )


@pytest.fixture(scope="module")
def failing_provider() -> Iterator[Wire]:
    with wire_server(_upstream_error) as wire:
        yield wire


@pytest.fixture(scope="module")
def proxy(
    audit_sinks: SpanSinks, otel_audit_config: AuditConfigWriter, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("langfuse-destination-failure")
    with (
        gateway_from_environment() as gateway,
        owned_proxy_process(
            gateway,
            directory,
            {
                "LITELLM_OTEL_V2": "1",
                "OTEL_BSP_SCHEDULE_DELAY": "300",
                "OTEL_EXPORTER": "otlp_http",
                "OTEL_ENDPOINT": audit_sinks.operator,
                "ARIZE_HTTP_ENDPOINT": audit_sinks.arize + "/v1/traces",
            },
            config=otel_audit_config(directory, {"callbacks": ["otel", "langfuse_otel"]}),
            remove_environment=("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"),
            workers=2,
        ) as owned,
    ):
        yield owned.gateway


def _trace(url: str, since: int, trace_id: str) -> tuple[Span, ...]:
    return spans_for_trace(recorded_spans(url, since)[1], trace_id)


def _has_server_span(spans: tuple[Span, ...]) -> bool:
    return any(span["kind"] == SERVER_SPAN_KIND for span in spans)


def _has_failure_record(spans: tuple[Span, ...]) -> bool:
    return _has_server_span(spans) and any(span["name"] == FAILURE_SPAN for span in spans)


def _fail(proxy: Gateway, key: str, model: str) -> str:
    trace_id: Final = uuid.uuid4().hex
    response: Final = proxy.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": f"failure-{trace_id}"}]},
        key=key,
        headers={"traceparent": f"00-{trace_id}-{uuid.uuid4().hex[:16]}-01"},
    )
    assert response.status_code >= 400, response.text
    return trace_id


def _operator_record(
    sinks: SpanSinks, since: int, trace_id: str, complete: Callable[[tuple[Span, ...]], bool]
) -> tuple[Mapping[str, JsonValue], tuple[str, ...]]:
    held(lambda: _trace(sinks.operator, since, trace_id), complete, holding=3, seconds=40)
    spans: Final = _trace(sinks.operator, since, trace_id)
    server: Final = next(span for span in spans if span["kind"] == SERVER_SPAN_KIND)
    failures: Final = tuple(str(span["name"]) for span in spans if span["name"] == FAILURE_SPAN)
    return {name: server["attributes"].get(name) for name in ERROR_ATTRIBUTES}, failures


@pytest.mark.parametrize("failure", ["upstream_500", "unknown_model"])
@pytest.mark.parametrize("callback_type", ["success", "success_and_failure"])
@pytest.mark.parametrize("level", ["key", "team"])
def test_the_operator_collector_records_a_destination_requests_failure_like_any_other(
    level: Literal["key", "team"],
    callback_type: str,
    failure: Literal["upstream_500", "unknown_model"],
    proxy: Gateway,
    failing_provider: Wire,
    audit_sinks: SpanSinks,
) -> None:
    logging: Final[list[JsonValue]] = [
        {
            "callback_name": "langfuse_otel",
            "callback_type": callback_type,
            "callback_vars": langfuse_callback_variables(audit_sinks),
        }
    ]
    with proxy.scenario() as scenario:
        model: Final = (
            scenario.model(api_base=failing_provider.url + "/v1")
            if failure == "upstream_500"
            else f"not-deployed-{uuid.uuid4().hex}"
        )
        alias: Final = f"lf-destination-failure-{uuid.uuid4().hex}"
        plain_team: Final = scenario.team()
        destination_team: Final = scenario.team(metadata={"logging": logging} if level == "team" else {})
        plain_key: Final = scenario.key(key_alias=f"{alias}-plain", team_id=plain_team)
        destination_key: Final = scenario.key(
            key_alias=alias,
            team_id=destination_team,
            **({"metadata": {"logging": logging}} if level == "key" else {}),
        )
        since: Final = (recorded_spans(audit_sinks.operator)[0], recorded_spans(audit_sinks.tenant)[0])

        plain_trace: Final = _fail(proxy, plain_key, model)
        destination_trace: Final = _fail(proxy, destination_key, model)
        plain: Final = _operator_record(audit_sinks, since[0], plain_trace, _has_failure_record)
        destination: Final = _operator_record(audit_sinks, since[0], destination_trace, _has_server_span)
        held(lambda: _trace(audit_sinks.tenant, since[1], destination_trace), _has_server_span, holding=3, seconds=40)
        tenant: Final = _trace(audit_sinks.tenant, since[1], destination_trace)

    assert plain[1] == (FAILURE_SPAN,), plain
    assert None not in plain[0].values(), plain[0]
    assert plain[0]["metadata.user_api_key_team_id"] == plain_team, plain[0]
    assert destination[1] == plain[1], (destination, plain)
    assert {**destination[0], "metadata.user_api_key_team_id": None} == {
        **plain[0],
        "metadata.user_api_key_team_id": None,
    }, (destination[0], plain[0])
    assert destination[0]["metadata.user_api_key_team_id"] == destination_team, destination[0]
    assert FAILURE_SPAN not in {span["name"] for span in tenant}, [span["name"] for span in tenant]
    assert [span["name"] for span in tenant if STACK_TRACE in span["attributes"]] == [], [
        span["name"] for span in tenant
    ]


@pytest.mark.parametrize("failure", ["upstream_500", "unknown_model"])
def test_an_arize_destination_gets_the_failure_without_the_operators_stack_trace(
    failure: Literal["upstream_500", "unknown_model"],
    proxy: Gateway,
    failing_provider: Wire,
    audit_sinks: SpanSinks,
) -> None:
    logging: Final[list[JsonValue]] = [
        {
            "callback_name": "arize",
            "callback_type": "success",
            "callback_vars": {"arize_api_key": "arize-test-key", "arize_space_id": "arize-test-space"},
        }
    ]
    with proxy.scenario() as scenario:
        model: Final = (
            scenario.model(api_base=failing_provider.url + "/v1")
            if failure == "upstream_500"
            else f"not-deployed-{uuid.uuid4().hex}"
        )
        key: Final = scenario.key(
            key_alias=f"arize-destination-failure-{uuid.uuid4().hex}", metadata={"logging": logging}
        )
        since: Final = recorded_spans(audit_sinks.arize)[0]

        trace_id: Final = _fail(proxy, key, model)
        held(lambda: _trace(audit_sinks.arize, since, trace_id), _has_server_span, holding=3, seconds=40)
        spans: Final = _trace(audit_sinks.arize, since, trace_id)

    server: Final = next(span for span in spans if span["kind"] == SERVER_SPAN_KIND)
    assert "error.type" in server["attributes"], server["attributes"]
    assert [span["name"] for span in spans if STACK_TRACE in span["attributes"]] == [], [span["name"] for span in spans]
