"""Without operator Langfuse keys on OTel v2, Langfuse logging the OTel v2 fan-out does not carry still reaches the
tenant's own Langfuse project: a key's failure-only entry on a failed call, a config ``default_team_settings`` entry,
and a key destination's call the proxy rejected before any provider. A ``success_and_failure`` key entry sends that
project exactly one copy of a failed call."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterator
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

OPERATION: Final = "gen_ai.operation.name"
TAGS: Final = ["agent:orders", "run:r1"]
CONFIG_TEAM: Final = f"lf-config-team-{uuid.uuid4().hex[:8]}"
FAIL: Final = "please-fail"


def _chat(request: Request) -> Reply:
    if FAIL.encode() in request.body:
        return Reply(
            status=500, body=json.dumps({"error": {"message": "upstream boom", "type": "server_error"}}).encode()
        )
    return Reply(
        body=json.dumps(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
            }
        ).encode()
    )


@pytest.fixture(scope="module")
def provider() -> Iterator[Wire]:
    with wire_server(_chat) as wire:
        yield wire


@pytest.fixture(scope="module")
def proxy(
    audit_sinks: SpanSinks,
    otel_audit_config: AuditConfigWriter,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("langfuse-legacy-destinations")
    settings: Final[dict[str, JsonValue]] = {
        "callbacks": [],
        "default_team_settings": [
            {
                "team_id": CONFIG_TEAM,
                "success_callback": ["langfuse_otel"],
                "callback_vars": dict(langfuse_callback_variables(audit_sinks)),
            }
        ],
    }
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
            },
            config=otel_audit_config(directory, settings),
            remove_environment=("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"),
            workers=2,
        ) as owned,
    ):
        yield owned.gateway


def _tenant_spans(url: str, since: int, trace_id: str) -> tuple[Span, ...]:
    return spans_for_trace(recorded_spans(url, since)[1], trace_id)


def _tags(value: object) -> object:
    """The tags a span carries: the legacy logger writes them as a JSON string, OTel v2 as a list."""
    return json.loads(value) if isinstance(value, str) else value


def _settled(spans: tuple[Span, ...]) -> bool:
    """The model span and the request's root span have both arrived."""
    return any(OPERATION in span["attributes"] for span in spans) and any(
        str(span["name"]).startswith("POST ") for span in spans
    )


def _logging_entry(audit_sinks: SpanSinks, callback_type: str) -> list[JsonValue]:
    return [
        {
            "callback_name": "langfuse_otel",
            "callback_type": callback_type,
            "callback_vars": langfuse_callback_variables(audit_sinks),
        }
    ]


def _chat_status(proxy: Gateway, model: str, key: str, trace_id: str, content: str) -> int:
    body: Final[dict[str, JsonValue]] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "metadata": {"tags": [*TAGS], "session_id": f"sess-{trace_id}", "trace_user_id": "user-42"},
    }
    return proxy.request(
        "POST",
        "/v1/chat/completions",
        body,
        key=key,
        headers={"traceparent": f"00-{trace_id}-{uuid.uuid4().hex[:16]}-01"},
    ).status_code


def _trace_fields(spans: tuple[Span, ...]) -> list[tuple[object, object]]:
    return [
        (span["attributes"]["user.id"], _tags(span["attributes"].get("langfuse.trace.tags")))
        for span in spans
        if "user.id" in span["attributes"]
    ]


def _carries_the_trace_fields_once(spans: tuple[Span, ...]) -> bool:
    return _trace_fields(spans) == [("user-42", TAGS)]


def _one_failed_call(spans: tuple[Span, ...]) -> bool:
    return (
        _settled(spans)
        and len([span for span in spans if OPERATION in span["attributes"]]) == 1
        and "litellm_request" not in {span["name"] for span in spans}
    )


def _send(
    proxy: Gateway,
    provider: Wire,
    audit_sinks: SpanSinks,
    setup: Literal["failure", "config_team", "both"],
    expected: Callable[[tuple[Span, ...]], bool],
) -> tuple[int, tuple[Span, ...]]:
    logging_entry: Final = _logging_entry(audit_sinks, "failure" if setup == "failure" else "success_and_failure")
    with proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=provider.url + "/v1", num_retries=0)
        alias: Final = f"lf-legacy-{setup}-{uuid.uuid4().hex}"
        key: Final = (
            scenario.key(key_alias=alias, team_id=scenario.team(team_id=CONFIG_TEAM))
            if setup == "config_team"
            else scenario.key(key_alias=alias, metadata={"logging": logging_entry})
        )
        since: Final = recorded_spans(audit_sinks.tenant)[0]
        trace_id: Final = uuid.uuid4().hex
        content: Final = f"lf-legacy-{setup}-{trace_id}" + ("" if setup == "config_team" else f" {FAIL}")
        status: Final = _chat_status(proxy, model, key, trace_id, content)
        tenant: Final = held(
            lambda: _tenant_spans(audit_sinks.tenant, since, trace_id), expected, holding=3, seconds=40
        )
    return status, tenant


@pytest.mark.parametrize("setup", ["failure", "config_team"])
def test_a_tenant_langfuse_project_without_a_destination_still_gets_the_trace(
    setup: Literal["failure", "config_team"], proxy: Gateway, provider: Wire, audit_sinks: SpanSinks
) -> None:
    status, tenant = _send(proxy, provider, audit_sinks, setup, _carries_the_trace_fields_once)

    assert status == (500 if setup == "failure" else 200)
    assert _trace_fields(tenant) == [("user-42", TAGS)], [span["name"] for span in tenant]


def test_a_key_destinations_call_rejected_before_any_provider_still_gets_the_trace_fields(
    proxy: Gateway, provider: Wire, audit_sinks: SpanSinks
) -> None:
    with proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=provider.url + "/v1", num_retries=0)
        key: Final = scenario.key(
            key_alias=f"lf-legacy-rejected-{uuid.uuid4().hex}",
            metadata={"logging": _logging_entry(audit_sinks, "success_and_failure")},
            rpm_limit=1,
        )
        since: Final = recorded_spans(audit_sinks.tenant)[0]
        statuses: Final = {
            trace_id: _chat_status(proxy, model, key, trace_id, f"lf-legacy-rejected-{trace_id}")
            for trace_id in [uuid.uuid4().hex for _ in range(4)]
        }
        rejected: Final = [trace_id for trace_id, status in statuses.items() if status == 429]
        assert rejected, statuses
        tenant: Final = held(
            lambda: _tenant_spans(audit_sinks.tenant, since, rejected[0]),
            _carries_the_trace_fields_once,
            holding=3,
            seconds=40,
        )

    assert _trace_fields(tenant) == [("user-42", TAGS)], [span["name"] for span in tenant]


def test_a_failed_call_reaches_a_success_and_failure_destination_once(
    proxy: Gateway, provider: Wire, audit_sinks: SpanSinks
) -> None:
    status, tenant = _send(proxy, provider, audit_sinks, "both", _one_failed_call)

    assert status == 500
    assert _one_failed_call(tenant), [span["name"] for span in tenant]
