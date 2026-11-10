"""A key's or team's own Langfuse project gets one copy of the call with the Langfuse trace fields when the operator
has no Langfuse keys, whether or not ``langfuse_otel`` is listed in ``litellm_settings.callbacks``."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from typing import Final, Literal, cast

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
LEGACY_SPANS: Final = frozenset({"litellm_request", "raw_gen_ai_request"})


def _chat(_request: Request) -> Reply:
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
            config=otel_audit_config(directory, {"callbacks": cast("list[JsonValue]", request.param)}),
            remove_environment=("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"),
            workers=2,
        ) as owned,
    ):
        yield owned.gateway


def _trace_spans(url: str, since: int, trace_id: str) -> tuple[Span, ...]:
    return spans_for_trace(recorded_spans(url, since)[1], trace_id)


def _model_spans(spans: tuple[Span, ...]) -> tuple[Span, ...]:
    return tuple(span for span in spans if OPERATION in span["attributes"])


def _one_model_span(spans: tuple[Span, ...]) -> bool:
    return len(_model_spans(spans)) == 1 and not LEGACY_SPANS & {str(span["name"]) for span in spans}


@pytest.mark.parametrize("capture", [None, "span_only"])
@pytest.mark.parametrize("level", ["key", "team"])
def test_a_tenant_langfuse_project_gets_user_session_and_tags_without_operator_keys(
    level: Literal["key", "team"], capture: str | None, proxy: Gateway, provider: Wire, audit_sinks: SpanSinks
) -> None:
    logging: Final[list[JsonValue]] = [
        {
            "callback_name": "langfuse_otel",
            "callback_type": "success",
            "callback_vars": {
                **langfuse_callback_variables(audit_sinks),
                **({"capture_message_content": capture} if capture else {}),
            },
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
        body: Final[dict[str, JsonValue]] = {
            "model": model,
            "messages": [{"role": "user", "content": f"lf-tenant-only-{trace_id}"}],
            "metadata": {"tags": [*TAGS], "session_id": session, "trace_user_id": "user-42"},
        }
        response: Final = proxy.request(
            "POST",
            "/v1/chat/completions",
            body,
            key=key,
            headers={"traceparent": f"00-{trace_id}-{uuid.uuid4().hex[:16]}-01"},
        )
        assert response.status_code == 200, response.text

        tenant: Final = _model_spans(
            held(lambda: _trace_spans(audit_sinks.tenant, since[0], trace_id), _one_model_span, holding=3, seconds=40)
        )
        operator: Final = _model_spans(
            held(lambda: _trace_spans(audit_sinks.operator, since[1], trace_id), _one_model_span, holding=3, seconds=40)
        )

    assert [
        {name: span["attributes"].get(name) for name in ("user.id", "session.id", "langfuse.trace.tags")}
        for span in tenant
    ] == [{"user.id": "user-42", "session.id": session, "langfuse.trace.tags": TAGS}], [s["name"] for s in tenant]
    assert "langfuse.observation.metadata" in tenant[0]["attributes"], sorted(tenant[0]["attributes"])
    assert (f"lf-tenant-only-{trace_id}" in json.dumps(tenant[0]["attributes"])) == (capture == "span_only"), sorted(
        tenant[0]["attributes"]
    )
    assert len(operator) == 1, [span["name"] for span in operator]
    assert f"lf-tenant-only-{trace_id}" not in json.dumps(operator[0]["attributes"]), sorted(operator[0]["attributes"])
    assert [
        name for name in operator[0]["attributes"] if name in ("user.id", "session.id") or name.startswith("langfuse.")
    ] == [], sorted(operator[0]["attributes"])


def test_a_tenant_langfuse_project_gets_the_keys_langfuse_environment(
    proxy: Gateway, provider: Wire, audit_sinks: SpanSinks
) -> None:
    logging: Final[list[JsonValue]] = [
        {
            "callback_name": "langfuse_otel",
            "callback_type": "success",
            "callback_vars": {**langfuse_callback_variables(audit_sinks), "langfuse_environment": "prod"},
        }
    ]
    with proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=provider.url + "/v1")
        key: Final = scenario.key(key_alias=f"lf-tenant-env-{uuid.uuid4().hex}", metadata={"logging": logging})
        since: Final = (recorded_spans(audit_sinks.tenant)[0], recorded_spans(audit_sinks.operator)[0])
        trace_id: Final = uuid.uuid4().hex
        response: Final = proxy.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": f"lf-tenant-env-{trace_id}"}]},
            key=key,
            headers={"traceparent": f"00-{trace_id}-{uuid.uuid4().hex[:16]}-01"},
        )
        assert response.status_code == 200, response.text

        tenant: Final = held(
            lambda: _trace_spans(audit_sinks.tenant, since[0], trace_id), _one_model_span, holding=3, seconds=40
        )
        operator: Final = held(
            lambda: _trace_spans(audit_sinks.operator, since[1], trace_id), _one_model_span, holding=3, seconds=40
        )

    assert [span["resource"].get("langfuse.environment") for span in tenant] == ["prod"] * len(tenant), [
        span["name"] for span in tenant
    ]
    assert not any("langfuse.environment" in span["resource"] for span in operator), [span["name"] for span in operator]
