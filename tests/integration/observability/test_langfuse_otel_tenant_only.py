"""A key's or team's own Langfuse project gets the Langfuse trace fields when the operator has no Langfuse keys,
whether or not ``langfuse_otel`` is listed in ``litellm_settings.callbacks``."""

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


def _chat(request: Request) -> Reply:
    marker: Final = json.loads(request.body)["messages"][-1]["content"]
    return Reply(
        body=json.dumps(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": marker}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
            }
        ).encode()
    )


@pytest.fixture(scope="module")
def provider() -> Iterator[Wire]:
    with wire_server(_chat) as wire:
        yield wire


@pytest.fixture(scope="module", params=[["langfuse_otel"], []], ids=["listed", "not-listed"])
def proxy(
    request: pytest.FixtureRequest,
    audit_sinks: SpanSinks,
    otel_audit_config: AuditConfigWriter,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("langfuse-tenant-only")
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
            config=otel_audit_config(directory, {"callbacks": request.param}),
            remove_environment=("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"),
            workers=2,
        ) as owned,
    ):
        yield owned.gateway


def _model_spans(url: str, since: int, trace_id: str) -> tuple[Span, ...]:
    return tuple(
        span for span in spans_for_trace(recorded_spans(url, since)[1], trace_id) if OPERATION in span["attributes"]
    )


@pytest.mark.parametrize("level", ["key", "team"])
def test_a_tenant_langfuse_project_gets_user_session_and_tags_without_operator_keys(
    level: Literal["key", "team"], proxy: Gateway, provider: Wire, audit_sinks: SpanSinks
) -> None:
    logging: Final[list[JsonValue]] = [
        {
            "callback_name": "langfuse_otel",
            "callback_type": "success",
            "callback_vars": langfuse_callback_variables(audit_sinks),
        }
    ]
    with proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=provider.url + "/v1")
        alias: Final = f"lf-tenant-only-{uuid.uuid4().hex}"
        key: Final = (
            scenario.key(key_alias=alias, metadata={"logging": logging})
            if level == "key"
            else scenario.key(key_alias=alias, team_id=scenario.team(metadata={"logging": logging}))
        )
        since: Final = (recorded_spans(audit_sinks.tenant)[0], recorded_spans(audit_sinks.operator)[0])
        trace_id: Final = uuid.uuid4().hex
        session: Final = f"sess-{trace_id}"
        response: Final = proxy.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"lf-tenant-only-{trace_id}"}],
                "metadata": {"tags": TAGS, "session_id": session, "trace_user_id": "user-42"},
            },
            key=key,
            headers={"traceparent": f"00-{trace_id}-{uuid.uuid4().hex[:16]}-01"},
        )
        assert response.status_code == 200, response.text

        tenant: Final = eventually(
            lambda: _model_spans(audit_sinks.tenant, since[0], trace_id), lambda spans: bool(spans), seconds=40
        )
        operator: Final = eventually(
            lambda: _model_spans(audit_sinks.operator, since[1], trace_id), lambda spans: bool(spans), seconds=40
        )

    assert [
        {name: span["attributes"].get(name) for name in ("user.id", "session.id", "langfuse.trace.tags")}
        for span in tenant
    ] == [{"user.id": "user-42", "session.id": session, "langfuse.trace.tags": TAGS}], [s["name"] for s in tenant]
    assert "langfuse.observation.metadata" in tenant[0]["attributes"], sorted(tenant[0]["attributes"])
    assert "langfuse.observation.metadata" not in operator[0]["attributes"], sorted(operator[0]["attributes"])
