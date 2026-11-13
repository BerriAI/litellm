from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Final, Literal

import pytest
from integration._support.client import Gateway, gateway_from_environment, held
from integration._support.otlp_sink import (
    AuditConfigWriter,
    SpanSinks,
    langfuse_callback_variables,
    recorded_spans,
    spans_for_trace,
)
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue


def _chat(request: Request) -> Reply:
    assert request.target.endswith("/chat/completions"), request.target
    assert b'"messages"' in request.body, request.body
    return Reply(
        body=(
            b'{"id":"chatcmpl-scope","object":"chat.completion","created":1,"model":"gpt-4o-mini",'
            b'"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],'
            b'"usage":{"prompt_tokens":7,"completion_tokens":2,"total_tokens":9}}'
        )
    )


@pytest.fixture(scope="module")
def provider() -> Iterator[Wire]:
    with wire_server(_chat) as wire:
        yield wire


@pytest.fixture(scope="module", params=["llm_only", None], ids=["proxy-llm-only", "proxy-unset"])
def proxy(
    request: pytest.FixtureRequest,
    audit_sinks: SpanSinks,
    otel_audit_config: AuditConfigWriter,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[Gateway, bool]]:
    directory: Final = tmp_path_factory.mktemp("langfuse-destination-scope")
    narrowed: Final = request.param == "llm_only"
    with (
        gateway_from_environment() as gateway,
        owned_proxy_process(
            gateway,
            directory,
            {
                "LITELLM_OTEL_V2": "true",
                **({"LITELLM_OTEL_LANGFUSE_DESTINATION_SPAN_SCOPE": "llm_only"} if narrowed else {}),
            },
            config=otel_audit_config(directory, {}),
            remove_environment=(
                "LANGFUSE_PUBLIC_KEY",
                "LANGFUSE_SECRET_KEY",
                "LANGFUSE_HOST",
                "LITELLM_OTEL_LANGFUSE_SPAN_SCOPE",
                *(("LITELLM_OTEL_LANGFUSE_DESTINATION_SPAN_SCOPE",) if not narrowed else ()),
            ),
            workers=2,
        ) as owned,
    ):
        yield owned.gateway, narrowed


@pytest.mark.parametrize(("level", "scope"), [("key", None), ("key", "full"), ("team", None)])
def test_proxy_default_limits_only_destinations_without_an_explicit_scope(
    level: Literal["key", "team"],
    scope: str | None,
    proxy: tuple[Gateway, bool],
    provider: Wire,
    audit_sinks: SpanSinks,
) -> None:
    gateway, narrowed = proxy
    logging: Final[list[JsonValue]] = [
        {
            "callback_name": "langfuse_otel",
            "callback_type": "success",
            "callback_vars": {
                **langfuse_callback_variables(audit_sinks),
                **({"langfuse_span_scope": scope} if scope else {}),
            },
        }
    ]
    with gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=provider.url + "/v1")
        key: Final = (
            scenario.key(metadata={"logging": logging})
            if level == "key"
            else scenario.key(team_id=scenario.team(metadata={"logging": logging}))
        )
        since: Final = recorded_spans(audit_sinks.tenant)[0]
        trace_id: Final = uuid.uuid4().hex
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "destination span scope"}]},
            key=key,
            headers={"traceparent": f"00-{trace_id}-{uuid.uuid4().hex[:16]}-01"},
        )
        assert response.status_code == 200, response.text
        spans: Final = held(
            lambda: spans_for_trace(recorded_spans(audit_sinks.tenant, since)[1], trace_id),
            lambda found: any("gen_ai.operation.name" in span["attributes"] for span in found),
            holding=3,
            seconds=40,
        )
    if narrowed and scope is None:
        assert len(spans) == 1, [(span["name"], span["kind"]) for span in spans]
        assert "gen_ai.operation.name" in spans[0]["attributes"], spans
        assert not any(span["kind"] == 2 or span["attributes"].get("db.system") == "postgresql" for span in spans)
        return
    assert any(span["kind"] == 2 for span in spans), [(span["name"], span["kind"]) for span in spans]
    assert any("gen_ai.operation.name" in span["attributes"] for span in spans), spans
    assert any(span["attributes"].get("db.system") == "postgresql" for span in spans), spans
