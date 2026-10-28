"""Without operator Langfuse keys on OTel v2, Langfuse logging that resolves no key or team destination still reaches
the tenant's own Langfuse project: a key's failure-only entry on a failed call, and a config ``default_team_settings``
entry. A ``success_and_failure`` key entry sends that project exactly one copy of a failed call."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from typing import Final, Literal

import pytest
from integration._support.client import Gateway, eventually, gateway_from_environment
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


def _send(
    proxy: Gateway, provider: Wire, audit_sinks: SpanSinks, setup: Literal["failure", "config_team", "both"]
) -> tuple[int, tuple[Span, ...]]:
    logging_entry: Final[list[JsonValue]] = [
        {
            "callback_name": "langfuse_otel",
            "callback_type": "failure" if setup == "failure" else "success_and_failure",
            "callback_vars": langfuse_callback_variables(audit_sinks),
        }
    ]
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
        body: Final[dict[str, JsonValue]] = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "metadata": {"tags": [*TAGS], "session_id": f"sess-{trace_id}", "trace_user_id": "user-42"},
        }
        response: Final = proxy.request(
            "POST",
            "/v1/chat/completions",
            body,
            key=key,
            headers={"traceparent": f"00-{trace_id}-{uuid.uuid4().hex[:16]}-01"},
        )
        tenant: Final = eventually(
            lambda: _tenant_spans(audit_sinks.tenant, since, trace_id),
            _settled if setup == "both" else lambda spans: any(span["attributes"].get("user.id") for span in spans),
            seconds=40,
        )
    return response.status_code, tenant


@pytest.mark.parametrize("setup", ["failure", "config_team"])
def test_a_tenant_langfuse_project_without_a_destination_still_gets_the_trace(
    setup: Literal["failure", "config_team"], proxy: Gateway, provider: Wire, audit_sinks: SpanSinks
) -> None:
    status, tenant = _send(proxy, provider, audit_sinks, setup)

    assert status == (500 if setup == "failure" else 200)
    assert [
        (span["attributes"]["user.id"], _tags(span["attributes"].get("langfuse.trace.tags")))
        for span in tenant
        if "user.id" in span["attributes"]
    ] == [("user-42", TAGS)], [span["name"] for span in tenant]


def test_a_failed_call_reaches_a_success_and_failure_destination_once(
    proxy: Gateway, provider: Wire, audit_sinks: SpanSinks
) -> None:
    status, tenant = _send(proxy, provider, audit_sinks, "both")

    assert status == 500
    assert len([span for span in tenant if OPERATION in span["attributes"]]) == 1, [span["name"] for span in tenant]
    assert "litellm_request" not in {span["name"] for span in tenant}, [span["name"] for span in tenant]
