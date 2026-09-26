from __future__ import annotations

import asyncio
import json
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import (
    Gateway,
    Scenario,
    eventually,
    gateway_from_environment,
    object_value,
)
from integration._support.database import read_rows
from integration._support.otlp_sink import (
    ConnectSink,
    GrpcSink,
    Span,
    SpanSinks,
    configure_sink,
    recorded_requests,
    recorded_spans,
    span_class,
    spans_for_trace,
)
from integration._support.process import owned_proxy
from pydantic import JsonValue

AuditConfigWriter = Callable[[Path, Mapping[str, JsonValue]], Path]

ARIZE_VARS: Final[dict[str, str]] = {"arize_space_id": "audit-space", "arize_api_key": "audit-arize-key"}
SPAN_SCOPE_VAR: Final = "otel_span_scope"


@pytest.fixture(scope="module")
def gateway(
    audit_sinks: SpanSinks,
    otel_audit_config: AuditConfigWriter,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("otel-audit-proxy")
    with gateway_from_environment() as base:
        with owned_proxy(
            base,
            directory,
            {"LITELLM_OTEL_V2": "1", "ARIZE_HTTP_ENDPOINT": audit_sinks.arize},
            config=otel_audit_config(directory, {}),
            workers=2,
        ) as candidate:
            yield candidate


def _nonce() -> str:
    return f"otelaudit-{uuid.uuid4().hex}"


def _add_callback(
    gateway: Gateway,
    team_id: str,
    callback_vars: Mapping[str, JsonValue],
    *,
    callback_name: str = "langfuse_otel",
    callback_type: str | None = None,
    key: str | None = None,
) -> httpx.Response:
    body: Final[dict[str, JsonValue]] = {"callback_name": callback_name, "callback_vars": dict(callback_vars)}
    if callback_type is not None:
        body["callback_type"] = callback_type
    return gateway.request("POST", f"/team/{team_id}/callback", body, key=key)


def _key_on_team(scenario: Scenario, team_id: str, **fields: JsonValue) -> str:
    return scenario.key(team_id=team_id, **fields)


def _audit_model(scenario: Scenario, upstream_url: str) -> str:
    return scenario.model(model="openai/audit-chat", api_base=f"{upstream_url}/v1")


def _chat(gateway: Gateway, key: str, model: str, nonce: str, *, stream: bool = False) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": nonce}], **({"stream": True} if stream else {})},
        key=key,
    )


def _response_id(body: JsonValue) -> str | None:
    if isinstance(body, dict):
        value: Final = body.get("id")
        if isinstance(value, str):
            return value
    return None


def _call_id(response: httpx.Response) -> str | None:
    return response.headers.get("x-litellm-call-id")


def _trace_id(sink_url: str, *, call_id: str | None = None, response_id: str | None = None, seconds: float = 40) -> str:
    def look() -> str | None:
        _, spans = recorded_spans(sink_url)
        return next(
            (
                str(span["trace_id"])
                for span in spans
                if (call_id is not None and span["attributes"].get("litellm.call_id") == call_id)
                or (response_id is not None and span["attributes"].get("gen_ai.response.id") == response_id)
            ),
            None,
        )

    found: Final = eventually(look, lambda value: value is not None, seconds=seconds)
    assert found is not None
    return found


def _trace_spans(sink_url: str, trace_id: str, seconds: float = 30) -> tuple[Span, ...]:
    def settled() -> tuple[Span, ...] | None:
        _, spans = recorded_spans(sink_url)
        group: Final = spans_for_trace(spans, trace_id)
        classes: Final = {span_class(span) for span in group}
        return group if "root" in classes and "tenant" in classes else None

    group: Final = eventually(settled, lambda value: value is not None, seconds=seconds)
    assert group is not None
    return group


def _classes(spans: tuple[Span, ...]) -> dict[str, int]:
    return {name: sum(1 for span in spans if span_class(span) == name) for name in ("root", "tenant", "internal")}


def _assert_no_internal(trace_spans: tuple[Span, ...]) -> None:
    counts: Final = _classes(trace_spans)
    names: Final = sorted(str(span["name"]) for span in trace_spans)
    assert counts["root"] == 1 and counts["internal"] == 0 and counts["tenant"] >= 1, (
        f"expected root + tenant spans only, got {counts} with {names}"
    )


def _assert_full(trace_spans: tuple[Span, ...]) -> None:
    counts: Final = _classes(trace_spans)
    names: Final = sorted(str(span["name"]) for span in trace_spans)
    assert counts["root"] == 1 and counts["internal"] >= 1 and counts["tenant"] >= 1, (
        f"expected a full tree with internal spans, got {counts} with {names}"
    )


def _no_internal_flow(
    gateway: Gateway,
    upstream_url: str,
    langfuse_vars: Mapping[str, JsonValue],
    send: Callable[[Gateway, str, str, str], httpx.Response],
    *,
    span_scope: str | None = "no_internal",
    callback_vars: Mapping[str, JsonValue] | None = None,
) -> tuple[httpx.Response, str]:
    with gateway.scenario() as scenario:
        model: Final = _audit_model(scenario, upstream_url)
        team_id: Final = scenario.team()
        vars: Final = {
            **dict(langfuse_vars),
            **(dict(callback_vars) if callback_vars else {}),
            **({SPAN_SCOPE_VAR: span_scope} if span_scope is not None else {}),
        }
        response: Final = _add_callback(gateway, team_id, vars)
        assert response.status_code == 200, f"callback setup failed: {response.status_code} {response.text}"
        key: Final = _key_on_team(scenario, team_id)
        nonce: Final = _nonce()
        traffic: Final = send(gateway, key, model, nonce)
        return traffic, nonce


def _assert_trace_split(audit_sinks: SpanSinks, traffic: httpx.Response) -> None:
    assert traffic.status_code == 200, traffic.text
    call_id: Final = _call_id(traffic)
    response_id: Final = _response_id(traffic.json())
    operator_trace: Final = _trace_id(audit_sinks.operator, call_id=call_id, response_id=response_id)
    _assert_full(_trace_spans(audit_sinks.operator, operator_trace))
    tenant_trace: Final = _trace_id(audit_sinks.tenant, call_id=call_id, response_id=response_id)
    _assert_no_internal(_trace_spans(audit_sinks.tenant, tenant_trace))
    assert tenant_trace == operator_trace


def _assert_upstream_saw(upstream_url: str, nonce: str) -> None:
    observations: Final = httpx.get(f"{upstream_url}/__observations", trust_env=False, timeout=15).json()["requests"]
    matching: Final = [
        entry for entry in observations if nonce in json.dumps(entry.get("body", {}))
    ]
    assert len(matching) >= 1, f"upstream never saw nonce {nonce}"


def _openai_sync_send(gateway: Gateway, key: str, model: str, nonce: str, *, stream: bool = False) -> httpx.Response:
    return _chat(gateway, key, model, nonce, stream=stream)


def _openai_sdk(gateway: Gateway, key: str, model: str, nonce: str) -> tuple[str | None, str | None]:
    import openai

    client: Final = openai.OpenAI(base_url=f"{gateway.client.base_url}".rstrip("/"), api_key=key, timeout=30)
    raw: Final = client.chat.completions.with_raw_response.create(
        model=model, messages=[{"role": "user", "content": nonce}]
    )
    parsed: Final = raw.parse()
    return parsed.id, raw.headers.get("x-litellm-call-id")


def _openai_sdk_async_stream(gateway: Gateway, key: str, model: str, nonce: str) -> tuple[str | None, str | None]:
    import openai

    async def run() -> tuple[str | None, str | None]:
        client: Final = openai.AsyncOpenAI(base_url=f"{gateway.client.base_url}".rstrip("/"), api_key=key, timeout=30)
        raw: Final = await client.chat.completions.with_raw_response.create(
            model=model, messages=[{"role": "user", "content": nonce}], stream=True
        )
        stream: Final = raw.parse()
        last_id: str | None = None
        async for chunk in stream:
            if chunk.id:
                last_id = chunk.id
        return last_id, raw.headers.get("x-litellm-call-id")

    return asyncio.run(run())


def _anthropic_sdk(gateway: Gateway, key: str, model: str, nonce: str) -> tuple[str | None, str | None]:
    import anthropic

    client: Final = anthropic.Anthropic(base_url=f"{gateway.client.base_url}".rstrip("/"), api_key=key, timeout=30)
    raw: Final = client.messages.with_raw_response.create(
        model=model, max_tokens=16, messages=[{"role": "user", "content": nonce}]
    )
    parsed: Final = raw.parse()
    return parsed.id, raw.headers.get("x-litellm-call-id")


def _anthropic_sdk_async_stream(gateway: Gateway, key: str, model: str, nonce: str) -> tuple[str | None, str | None]:
    import anthropic

    async def run() -> tuple[str | None, str | None]:
        client: Final = anthropic.AsyncAnthropic(base_url=f"{gateway.client.base_url}".rstrip("/"), api_key=key, timeout=30)
        last_id: str | None = None
        async with client.messages.stream(model=model, max_tokens=16, messages=[{"role": "user", "content": nonce}]) as stream:
            async for event in stream:
                if isinstance(event, anthropic.types.RawMessageStartEvent):
                    last_id = event.message.id
        return last_id, None

    return asyncio.run(run())


def _responses_httpx(gateway: Gateway, key: str, model: str, nonce: str, *, stream: bool) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/responses",
        {"model": model, "input": nonce, **({"stream": True} if stream else {})},
        key=key,
    )


def _responses_stream_id(response: httpx.Response) -> str | None:
    for line in response.text.splitlines():
        if line.startswith("data:"):
            try:
                event: Final = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            response_obj: Final = event.get("response")
            if isinstance(response_obj, dict) and isinstance(response_obj.get("id"), str):
                return response_obj["id"]
    return None


def _send_and_assert_no_internal(
    gateway: Gateway,
    upstream_url: str,
    audit_sinks: SpanSinks,
    langfuse_vars: Mapping[str, JsonValue],
    send: Callable[[Gateway, str, str, str], httpx.Response],
) -> None:
    traffic: Final[httpx.Response]
    nonce: Final[str]
    traffic, nonce = _no_internal_flow(gateway, upstream_url, langfuse_vars, send)
    _assert_trace_split(audit_sinks, traffic)
    _assert_upstream_saw(upstream_url, nonce)


@contextmanager
def _candidate(
    gateway: Gateway,
    tmp_path: Path,
    audit_sinks: SpanSinks,
    otel_audit_config: AuditConfigWriter,
    env: Mapping[str, str] = {},
    settings: Mapping[str, JsonValue] = {},
) -> Iterator[Gateway]:
    overrides: Final = {"LITELLM_OTEL_V2": "1", "ARIZE_HTTP_ENDPOINT": audit_sinks.arize, **dict(env)}
    with owned_proxy(
        gateway, tmp_path, overrides, config=otel_audit_config(tmp_path, settings), workers=2
    ) as candidate:
        yield candidate


def test_team_no_internal_chat_completions_openai_sync(gateway: Gateway, audit_sinks: SpanSinks, langfuse_vars: dict[str, JsonValue]) -> None:
    with gateway.scenario() as scenario:
        model: Final = _audit_model(scenario, gateway.upstream_url)
        team_id: Final = scenario.team()
        callback: Final = _add_callback(gateway, team_id, {**langfuse_vars, SPAN_SCOPE_VAR: "no_internal"})
        assert callback.status_code == 200, f"callback setup failed: {callback.status_code} {callback.text}"
        key: Final = _key_on_team(scenario, team_id)
        nonce = _nonce()
        response_id, call_id = _openai_sdk(gateway, key, model, nonce)
        tenant_trace: Final = _trace_id(audit_sinks.tenant, call_id=call_id, response_id=response_id)
        _assert_no_internal(_trace_spans(audit_sinks.tenant, tenant_trace))
        operator_trace: Final = _trace_id(audit_sinks.operator, call_id=call_id, response_id=response_id)
        _assert_full(_trace_spans(audit_sinks.operator, operator_trace))
        assert tenant_trace == operator_trace
        _assert_upstream_saw(gateway.upstream_url, nonce)


def test_team_no_internal_chat_completions_stream_openai_async(gateway: Gateway, audit_sinks: SpanSinks, langfuse_vars: dict[str, JsonValue]) -> None:
    with gateway.scenario() as scenario:
        model: Final = _audit_model(scenario, gateway.upstream_url)
        team_id: Final = scenario.team()
        callback: Final = _add_callback(gateway, team_id, {**langfuse_vars, SPAN_SCOPE_VAR: "no_internal"})
        assert callback.status_code == 200, callback.text
        key: Final = _key_on_team(scenario, team_id)
        nonce: Final = _nonce()
        response_id, call_id = _openai_sdk_async_stream(gateway, key, model, nonce)
        assert response_id is not None, "stream produced no response id"
        tenant_trace: Final = _trace_id(audit_sinks.tenant, call_id=call_id, response_id=response_id)
        _assert_no_internal(_trace_spans(audit_sinks.tenant, tenant_trace))
        operator_trace: Final = _trace_id(audit_sinks.operator, call_id=call_id, response_id=response_id)
        _assert_full(_trace_spans(audit_sinks.operator, operator_trace))
        _assert_upstream_saw(gateway.upstream_url, nonce)


def test_team_no_internal_messages_anthropic_sync(gateway: Gateway, audit_sinks: SpanSinks, langfuse_vars: dict[str, JsonValue]) -> None:
    with gateway.scenario() as scenario:
        model: Final = _audit_model(scenario, gateway.upstream_url)
        team_id: Final = scenario.team()
        callback: Final = _add_callback(gateway, team_id, {**langfuse_vars, SPAN_SCOPE_VAR: "no_internal"})
        assert callback.status_code == 200, callback.text
        key: Final = _key_on_team(scenario, team_id)
        nonce: Final = _nonce()
        response_id, call_id = _anthropic_sdk(gateway, key, model, nonce)
        assert call_id is not None, "no x-litellm-call-id header on /v1/messages"
        tenant_trace: Final = _trace_id(audit_sinks.tenant, call_id=call_id)
        _assert_no_internal(_trace_spans(audit_sinks.tenant, tenant_trace))
        operator_trace: Final = _trace_id(audit_sinks.operator, call_id=call_id)
        _assert_full(_trace_spans(audit_sinks.operator, operator_trace))
        _assert_upstream_saw(gateway.upstream_url, nonce)


def test_team_no_internal_messages_stream_anthropic_async(gateway: Gateway, audit_sinks: SpanSinks, langfuse_vars: dict[str, JsonValue]) -> None:
    with gateway.scenario() as scenario:
        model: Final = _audit_model(scenario, gateway.upstream_url)
        team_id: Final = scenario.team()
        callback: Final = _add_callback(gateway, team_id, {**langfuse_vars, SPAN_SCOPE_VAR: "no_internal"})
        assert callback.status_code == 200, callback.text
        key: Final = _key_on_team(scenario, team_id)
        nonce: Final = _nonce()
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 16,
                "stream": True,
                "messages": [{"role": "user", "content": nonce}],
            },
            key=key,
        )
        assert response.status_code == 200, response.text
        assert "message_stop" in response.text, response.text
        call_id: Final = _call_id(response)
        assert call_id is not None
        tenant_trace: Final = _trace_id(audit_sinks.tenant, call_id=call_id)
        _assert_no_internal(_trace_spans(audit_sinks.tenant, tenant_trace))
        operator_trace: Final = _trace_id(audit_sinks.operator, call_id=call_id)
        _assert_full(_trace_spans(audit_sinks.operator, operator_trace))
        _assert_upstream_saw(gateway.upstream_url, nonce)


def test_team_no_internal_responses_api(gateway: Gateway, audit_sinks: SpanSinks, langfuse_vars: dict[str, JsonValue]) -> None:
    with gateway.scenario() as scenario:
        model: Final = _audit_model(scenario, gateway.upstream_url)
        team_id: Final = scenario.team()
        callback: Final = _add_callback(gateway, team_id, {**langfuse_vars, SPAN_SCOPE_VAR: "no_internal"})
        assert callback.status_code == 200, callback.text
        key: Final = _key_on_team(scenario, team_id)
        nonce: Final = _nonce()
        response: Final = _responses_httpx(gateway, key, model, nonce, stream=False)
        assert response.status_code == 200, response.text
        call_id: Final = _call_id(response)
        response_id: Final = _response_id(response.json())
        tenant_trace: Final = _trace_id(audit_sinks.tenant, call_id=call_id, response_id=response_id)
        _assert_no_internal(_trace_spans(audit_sinks.tenant, tenant_trace))
        operator_trace: Final = _trace_id(audit_sinks.operator, call_id=call_id, response_id=response_id)
        _assert_full(_trace_spans(audit_sinks.operator, operator_trace))
        _assert_upstream_saw(gateway.upstream_url, nonce)


def test_team_no_internal_responses_stream(gateway: Gateway, audit_sinks: SpanSinks, langfuse_vars: dict[str, JsonValue]) -> None:
    with gateway.scenario() as scenario:
        model: Final = _audit_model(scenario, gateway.upstream_url)
        team_id: Final = scenario.team()
        callback: Final = _add_callback(gateway, team_id, {**langfuse_vars, SPAN_SCOPE_VAR: "no_internal"})
        assert callback.status_code == 200, callback.text
        key: Final = _key_on_team(scenario, team_id)
        nonce: Final = _nonce()
        response: Final = _responses_httpx(gateway, key, model, nonce, stream=True)
        assert response.status_code == 200, response.text
        response_id: Final = _responses_stream_id(response)
        call_id: Final = _call_id(response)
        assert response_id is not None or call_id is not None, response.text[:400]
        tenant_trace: Final = _trace_id(audit_sinks.tenant, call_id=call_id, response_id=response_id)
        _assert_no_internal(_trace_spans(audit_sinks.tenant, tenant_trace))
        operator_trace: Final = _trace_id(audit_sinks.operator, call_id=call_id, response_id=response_id)
        _assert_full(_trace_spans(audit_sinks.operator, operator_trace))
        _assert_upstream_saw(gateway.upstream_url, nonce)


def test_team_full_explicit_delivers_full_trace(gateway: Gateway, audit_sinks: SpanSinks, langfuse_vars: dict[str, JsonValue]) -> None:
    with gateway.scenario() as scenario:
        model: Final = _audit_model(scenario, gateway.upstream_url)
        team_id: Final = scenario.team()
        callback: Final = _add_callback(gateway, team_id, {**langfuse_vars, SPAN_SCOPE_VAR: "full"})
        assert callback.status_code == 200, callback.text
        key: Final = _key_on_team(scenario, team_id)
        response: Final = _chat(gateway, key, model, _nonce())
        assert response.status_code == 200, response.text
        call_id: Final = _call_id(response)
        response_id: Final = _response_id(response.json())
        tenant_trace: Final = _trace_id(audit_sinks.tenant, call_id=call_id, response_id=response_id)
        _assert_full(_trace_spans(audit_sinks.tenant, tenant_trace))


def test_var_absent_defaults_to_full(gateway: Gateway, audit_sinks: SpanSinks, langfuse_vars: dict[str, JsonValue]) -> None:
    with gateway.scenario() as scenario:
        model: Final = _audit_model(scenario, gateway.upstream_url)
        team_id: Final = scenario.team()
        callback: Final = _add_callback(gateway, team_id, langfuse_vars)
        assert callback.status_code == 200, callback.text
        key: Final = _key_on_team(scenario, team_id)
        response: Final = _chat(gateway, key, model, _nonce())
        assert response.status_code == 200, response.text
        tenant_trace: Final = _trace_id(audit_sinks.tenant, call_id=_call_id(response))
        _assert_full(_trace_spans(audit_sinks.tenant, tenant_trace))


def _key_logging_entry(callback_vars: Mapping[str, JsonValue], callback_name: str = "langfuse_otel") -> list[JsonValue]:
    return [{"callback_name": callback_name, "callback_vars": dict(callback_vars)}]


def test_key_level_no_internal_without_team_callbacks(gateway: Gateway, audit_sinks: SpanSinks, langfuse_vars: dict[str, JsonValue]) -> None:
    with gateway.scenario() as scenario:
        model: Final = _audit_model(scenario, gateway.upstream_url)
        key: Final = scenario.key(
            metadata={"logging": _key_logging_entry({**langfuse_vars, SPAN_SCOPE_VAR: "no_internal"})}
        )
        response: Final = _chat(gateway, key, model, _nonce())
        assert response.status_code == 200, response.text
        call_id: Final = _call_id(response)
        tenant_trace: Final = _trace_id(audit_sinks.tenant, call_id=call_id)
        _assert_no_internal(_trace_spans(audit_sinks.tenant, tenant_trace))
        operator_trace: Final = _trace_id(audit_sinks.operator, call_id=call_id)
        _assert_full(_trace_spans(audit_sinks.operator, operator_trace))


def test_key_full_wins_over_team_no_internal(gateway: Gateway, audit_sinks: SpanSinks, langfuse_vars: dict[str, JsonValue]) -> None:
    with gateway.scenario() as scenario:
        model: Final = _audit_model(scenario, gateway.upstream_url)
        team_id: Final = scenario.team()
        callback: Final = _add_callback(gateway, team_id, {**langfuse_vars, SPAN_SCOPE_VAR: "no_internal"})
        assert callback.status_code == 200, callback.text
        key: Final = scenario.key(
            team_id=team_id,
            metadata={"logging": _key_logging_entry({**langfuse_vars, SPAN_SCOPE_VAR: "full"})},
        )
        response: Final = _chat(gateway, key, model, _nonce())
        assert response.status_code == 200, response.text
        tenant_trace: Final = _trace_id(audit_sinks.tenant, call_id=_call_id(response))
        _assert_full(_trace_spans(audit_sinks.tenant, tenant_trace))


def test_key_no_internal_wins_over_team_full(gateway: Gateway, audit_sinks: SpanSinks, langfuse_vars: dict[str, JsonValue]) -> None:
    with gateway.scenario() as scenario:
        model: Final = _audit_model(scenario, gateway.upstream_url)
        team_id: Final = scenario.team()
        callback: Final = _add_callback(gateway, team_id, {**langfuse_vars, SPAN_SCOPE_VAR: "full"})
        assert callback.status_code == 200, callback.text
        key: Final = scenario.key(
            team_id=team_id,
            metadata={"logging": _key_logging_entry({**langfuse_vars, SPAN_SCOPE_VAR: "no_internal"})},
        )
        response: Final = _chat(gateway, key, model, _nonce())
        assert response.status_code == 200, response.text
        tenant_trace: Final = _trace_id(audit_sinks.tenant, call_id=_call_id(response))
        _assert_no_internal(_trace_spans(audit_sinks.tenant, tenant_trace))


def test_langfuse_no_internal_arize_full_split(gateway: Gateway, audit_sinks: SpanSinks, langfuse_vars: dict[str, JsonValue]) -> None:
    with gateway.scenario() as scenario:
        model: Final = _audit_model(scenario, gateway.upstream_url)
        team_id: Final = scenario.team()
        langfuse: Final = _add_callback(gateway, team_id, {**langfuse_vars, SPAN_SCOPE_VAR: "no_internal"})
        assert langfuse.status_code == 200, langfuse.text
        arize: Final = _add_callback(
            gateway, team_id, {**ARIZE_VARS, SPAN_SCOPE_VAR: "full"}, callback_name="arize"
        )
        assert arize.status_code == 200, arize.text
        key: Final = _key_on_team(scenario, team_id)
        response: Final = _chat(gateway, key, model, _nonce())
        assert response.status_code == 200, response.text
        call_id: Final = _call_id(response)
        langfuse_trace: Final = _trace_id(audit_sinks.tenant, call_id=call_id)
        _assert_no_internal(_trace_spans(audit_sinks.tenant, langfuse_trace))
        arize_trace: Final = _trace_id(audit_sinks.arize, call_id=call_id)
        arize_spans: Final = _trace_spans(audit_sinks.arize, arize_trace)
        _assert_full(arize_spans)
        assert arize_trace == langfuse_trace


def test_llm_only_scope_on_arize_grpc_keeps_model_span(
    gateway: Gateway, audit_sinks: SpanSinks, arize_grpc_sink: GrpcSink, otel_audit_config: AuditConfigWriter, tmp_path: Path
) -> None:
    overrides: Final = {
        "LITELLM_OTEL_V2": "1",
        "ARIZE_ENDPOINT": arize_grpc_sink.url,
        "ARIZE_HTTP_ENDPOINT": audit_sinks.arize,
    }
    with owned_proxy(gateway, tmp_path, overrides, config=otel_audit_config(tmp_path, {}), workers=2) as candidate:
        with candidate.scenario() as scenario:
            model: Final = _audit_model(scenario, candidate.upstream_url)
            team_id: Final = scenario.team()
            callback: Final = _add_callback(
                candidate,
                team_id,
                {**ARIZE_VARS, SPAN_SCOPE_VAR: "llm_only"},
                callback_name="arize",
            )
            assert callback.status_code == 200, callback.text
            key: Final = _key_on_team(scenario, team_id)
            response: Final = _chat(candidate, key, model, _nonce())
            assert response.status_code == 200, response.text
            arize_trace: Final = _trace_id(arize_grpc_sink.control_url, call_id=_call_id(response))

            def settled() -> tuple[Span, ...] | None:
                _, spans = recorded_spans(arize_grpc_sink.control_url)
                group: Final = spans_for_trace(spans, arize_trace)
                return group if group else None

            group: Final = eventually(settled, lambda value: value is not None, seconds=30)
            assert group is not None
            names: Final = sorted(str(span["name"]) for span in group)
            assert all("gen_ai.operation.name" in span["attributes"] for span in group), names


def test_langfuse_alias_and_otel_scope_disagree_rejected(gateway: Gateway, langfuse_vars: dict[str, JsonValue]) -> None:
    with gateway.scenario() as scenario:
        team_id: Final = scenario.team()
        response: Final = _add_callback(
            gateway,
            team_id,
            {**langfuse_vars, "langfuse_span_scope": "llm_only", SPAN_SCOPE_VAR: "no_internal"},
        )
        assert response.status_code == 400, f"expected 400, got {response.status_code}: {response.text}"
        assert "span_scope" in response.text, response.text


def test_split_scope_aliases_on_two_entries_rejected(gateway: Gateway, langfuse_vars: dict[str, JsonValue]) -> None:
    response: Final = gateway.request(
        "POST",
        "/team/new",
        {
            "metadata": {
                "logging": [
                    *_key_logging_entry({**langfuse_vars, "langfuse_span_scope": "full"}),
                    *_key_logging_entry({**langfuse_vars, SPAN_SCOPE_VAR: "llm_only"}),
                ]
            }
        },
    )
    assert response.status_code == 400, f"expected 400, got {response.status_code}: {response.text}"
    assert "span_scope" in response.text, response.text


def test_additive_mode_operator_full_tenant_no_internal(gateway: Gateway, audit_sinks: SpanSinks, langfuse_vars: dict[str, JsonValue], otel_audit_config: AuditConfigWriter, tmp_path: Path) -> None:
    with _candidate(gateway, tmp_path, audit_sinks, otel_audit_config, settings={"otel_tenant_destination_mode": "additive"}) as candidate:
        with candidate.scenario() as scenario:
            model: Final = _audit_model(scenario, candidate.upstream_url)
            team_id: Final = scenario.team()
            callback: Final = _add_callback(candidate, team_id, {**langfuse_vars, SPAN_SCOPE_VAR: "no_internal"})
            assert callback.status_code == 200, callback.text
            key: Final = _key_on_team(scenario, team_id)
            response: Final = _chat(candidate, key, model, _nonce())
            assert response.status_code == 200, response.text
            call_id: Final = _call_id(response)
            operator_trace: Final = _trace_id(audit_sinks.operator, call_id=call_id)
            _assert_full(_trace_spans(audit_sinks.operator, operator_trace))
            tenant_trace: Final = _trace_id(audit_sinks.tenant, call_id=call_id)
            _assert_no_internal(_trace_spans(audit_sinks.tenant, tenant_trace))


def test_operator_sink_keeps_span_scope_under_no_internal(gateway: Gateway, audit_sinks: SpanSinks, langfuse_vars: dict[str, JsonValue]) -> None:
    with gateway.scenario() as scenario:
        model: Final = _audit_model(scenario, gateway.upstream_url)
        team_id: Final = scenario.team()
        callback: Final = _add_callback(gateway, team_id, {**langfuse_vars, SPAN_SCOPE_VAR: "no_internal"})
        assert callback.status_code == 200, callback.text
        key: Final = _key_on_team(scenario, team_id)
        response: Final = _chat(gateway, key, model, _nonce())
        assert response.status_code == 200, response.text
        operator_trace: Final = _trace_id(audit_sinks.operator, call_id=_call_id(response))
        operator_spans: Final = _trace_spans(audit_sinks.operator, operator_trace)
        _assert_full(operator_spans)
        internal_names: Final = sorted(
            str(span["name"]) for span in operator_spans if span_class(span) == "internal"
        )
        assert any("auth" in name or "redis" in name or "postgres" in name for name in internal_names), internal_names


def test_guardrail_span_survives_no_internal(gateway: Gateway, audit_sinks: SpanSinks, langfuse_vars: dict[str, JsonValue], otel_audit_config: AuditConfigWriter, tmp_path: Path) -> None:
    guardrail_name: Final = f"audit-filter-{uuid.uuid4().hex[:8]}"
    config: Final = yaml.safe_load(otel_audit_config(tmp_path, {}).read_text())
    config["guardrails"] = [
        {
            "guardrail_name": guardrail_name,
            "litellm_params": {
                "guardrail": "litellm_content_filter",
                "mode": "pre_call",
                "default_on": True,
                "patterns": [
                    {
                        "pattern_type": "regex",
                        "pattern_name": "audit_secret",
                        "pattern": "TOPSECRET\\d{9}",
                        "action": "BLOCK",
                    }
                ],
            },
        }
    ]
    path: Final = tmp_path / "audit-guardrail.yaml"
    path.write_text(yaml.safe_dump(config))
    with owned_proxy(
        gateway, tmp_path, {"LITELLM_OTEL_V2": "1"}, config=path, workers=2
    ) as candidate:
        with candidate.scenario() as scenario:
            model: Final = _audit_model(scenario, candidate.upstream_url)
            team_id: Final = scenario.team()
            callback: Final = _add_callback(candidate, team_id, {**langfuse_vars, SPAN_SCOPE_VAR: "no_internal"})
            assert callback.status_code == 200, callback.text
            key: Final = _key_on_team(scenario, team_id)
            response: Final = _chat(candidate, key, model, _nonce())
            assert response.status_code == 200, response.text
            call_id: Final = _call_id(response)
            tenant_trace: Final = _trace_id(audit_sinks.tenant, call_id=call_id)
            tenant_spans: Final = _trace_spans(audit_sinks.tenant, tenant_trace)

            def guardrail_seen() -> tuple[Span, ...] | None:
                _, spans = recorded_spans(audit_sinks.tenant)
                group: Final = spans_for_trace(spans, tenant_trace)
                kept: Final = tuple(
                    span
                    for span in group
                    if "litellm.guardrail.name" in span["attributes"]
                )
                return kept or None

            kept: Final = eventually(guardrail_seen, lambda value: value is not None, seconds=30)
            assert kept is not None, f"guardrail span missing at tenant sink: {tenant_spans}"


def test_env_no_internal_applies_when_var_absent(gateway: Gateway, audit_sinks: SpanSinks, langfuse_vars: dict[str, JsonValue], otel_audit_config: AuditConfigWriter, tmp_path: Path) -> None:
    with _candidate(gateway, tmp_path, audit_sinks, otel_audit_config, env={"LITELLM_OTEL_TENANT_SPAN_SCOPE": "no_internal"}) as candidate:
        with candidate.scenario() as scenario:
            model: Final = _audit_model(scenario, candidate.upstream_url)
            team_id: Final = scenario.team()
            callback: Final = _add_callback(candidate, team_id, langfuse_vars)
            assert callback.status_code == 200, callback.text
            key: Final = _key_on_team(scenario, team_id)
            response: Final = _chat(candidate, key, model, _nonce())
            assert response.status_code == 200, response.text
            tenant_trace: Final = _trace_id(audit_sinks.tenant, call_id=_call_id(response))
            _assert_no_internal(_trace_spans(audit_sinks.tenant, tenant_trace))


def test_var_full_beats_env_no_internal(gateway: Gateway, audit_sinks: SpanSinks, langfuse_vars: dict[str, JsonValue], otel_audit_config: AuditConfigWriter, tmp_path: Path) -> None:
    with _candidate(gateway, tmp_path, audit_sinks, otel_audit_config, env={"LITELLM_OTEL_TENANT_SPAN_SCOPE": "no_internal"}) as candidate:
        with candidate.scenario() as scenario:
            model: Final = _audit_model(scenario, candidate.upstream_url)
            team_id: Final = scenario.team()
            callback: Final = _add_callback(candidate, team_id, {**langfuse_vars, SPAN_SCOPE_VAR: "full"})
            assert callback.status_code == 200, callback.text
            key: Final = _key_on_team(scenario, team_id)
            response: Final = _chat(candidate, key, model, _nonce())
            assert response.status_code == 200, response.text
            tenant_trace: Final = _trace_id(audit_sinks.tenant, call_id=_call_id(response))
            _assert_full(_trace_spans(audit_sinks.tenant, tenant_trace))


def test_litellm_setting_no_internal_applies_when_var_absent(gateway: Gateway, audit_sinks: SpanSinks, langfuse_vars: dict[str, JsonValue], otel_audit_config: AuditConfigWriter, tmp_path: Path) -> None:
    with _candidate(gateway, tmp_path, audit_sinks, otel_audit_config, settings={"otel_tenant_span_scope": "no_internal"}) as candidate:
        with candidate.scenario() as scenario:
            model: Final = _audit_model(scenario, candidate.upstream_url)
            team_id: Final = scenario.team()
            callback: Final = _add_callback(candidate, team_id, langfuse_vars)
            assert callback.status_code == 200, callback.text
            key: Final = _key_on_team(scenario, team_id)
            response: Final = _chat(candidate, key, model, _nonce())
            assert response.status_code == 200, response.text
            tenant_trace: Final = _trace_id(audit_sinks.tenant, call_id=_call_id(response))
            _assert_no_internal(_trace_spans(audit_sinks.tenant, tenant_trace))


def test_setting_full_beats_env_no_internal(gateway: Gateway, audit_sinks: SpanSinks, langfuse_vars: dict[str, JsonValue], otel_audit_config: AuditConfigWriter, tmp_path: Path) -> None:
    with _candidate(gateway, tmp_path, audit_sinks, otel_audit_config, env={"LITELLM_OTEL_TENANT_SPAN_SCOPE": "no_internal"}, settings={"otel_tenant_span_scope": "full"}
    ) as candidate:
        with candidate.scenario() as scenario:
            model: Final = _audit_model(scenario, candidate.upstream_url)
            team_id: Final = scenario.team()
            callback: Final = _add_callback(candidate, team_id, langfuse_vars)
            assert callback.status_code == 200, callback.text
            key: Final = _key_on_team(scenario, team_id)
            response: Final = _chat(candidate, key, model, _nonce())
            assert response.status_code == 200, response.text
            tenant_trace: Final = _trace_id(audit_sinks.tenant, call_id=_call_id(response))
            _assert_full(_trace_spans(audit_sinks.tenant, tenant_trace))


def test_env_no_internal_with_whitespace_and_case(gateway: Gateway, audit_sinks: SpanSinks, langfuse_vars: dict[str, JsonValue], otel_audit_config: AuditConfigWriter, tmp_path: Path) -> None:
    with _candidate(gateway, tmp_path, audit_sinks, otel_audit_config, env={"LITELLM_OTEL_TENANT_SPAN_SCOPE": " NO_INTERNAL "}) as candidate:
        with candidate.scenario() as scenario:
            model: Final = _audit_model(scenario, candidate.upstream_url)
            team_id: Final = scenario.team()
            callback: Final = _add_callback(candidate, team_id, langfuse_vars)
            assert callback.status_code == 200, callback.text
            key: Final = _key_on_team(scenario, team_id)
            response: Final = _chat(candidate, key, model, _nonce())
            assert response.status_code == 200, response.text
            tenant_trace: Final = _trace_id(audit_sinks.tenant, call_id=_call_id(response))
            _assert_no_internal(_trace_spans(audit_sinks.tenant, tenant_trace))


def test_env_bogus_value_falls_back_to_full(gateway: Gateway, audit_sinks: SpanSinks, langfuse_vars: dict[str, JsonValue], otel_audit_config: AuditConfigWriter, tmp_path: Path) -> None:
    with _candidate(gateway, tmp_path, audit_sinks, otel_audit_config, env={"LITELLM_OTEL_TENANT_SPAN_SCOPE": "bogus"}) as candidate:
        with candidate.scenario() as scenario:
            model: Final = _audit_model(scenario, candidate.upstream_url)
            team_id: Final = scenario.team()
            callback: Final = _add_callback(candidate, team_id, langfuse_vars)
            assert callback.status_code == 200, callback.text
            key: Final = _key_on_team(scenario, team_id)
            response: Final = _chat(candidate, key, model, _nonce())
            assert response.status_code == 200, response.text
            tenant_trace: Final = _trace_id(audit_sinks.tenant, call_id=_call_id(response))
            _assert_full(_trace_spans(audit_sinks.tenant, tenant_trace))


def test_empty_setting_falls_through_to_env(gateway: Gateway, audit_sinks: SpanSinks, langfuse_vars: dict[str, JsonValue], otel_audit_config: AuditConfigWriter, tmp_path: Path) -> None:
    with _candidate(gateway, tmp_path, audit_sinks, otel_audit_config, env={"LITELLM_OTEL_TENANT_SPAN_SCOPE": "no_internal"}, settings={"otel_tenant_span_scope": ""}
    ) as candidate:
        with candidate.scenario() as scenario:
            model: Final = _audit_model(scenario, candidate.upstream_url)
            team_id: Final = scenario.team()
            callback: Final = _add_callback(candidate, team_id, langfuse_vars)
            assert callback.status_code == 200, callback.text
            key: Final = _key_on_team(scenario, team_id)
            response: Final = _chat(candidate, key, model, _nonce())
            assert response.status_code == 200, response.text
            tenant_trace: Final = _trace_id(audit_sinks.tenant, call_id=_call_id(response))
            _assert_no_internal(_trace_spans(audit_sinks.tenant, tenant_trace))


def _bad_var_rejected(response: httpx.Response) -> None:
    assert response.status_code in (400, 422), f"expected rejection, got {response.status_code}: {response.text}"
    assert SPAN_SCOPE_VAR in response.text or "callback" in response.text, response.text


def test_callback_var_int_rejected(gateway: Gateway, langfuse_vars: dict[str, JsonValue]) -> None:
    with gateway.scenario() as scenario:
        team_id: Final = scenario.team()
        response: Final = _add_callback(gateway, team_id, {**langfuse_vars, SPAN_SCOPE_VAR: 1})
        _bad_var_rejected(response)


def test_callback_var_list_rejected(gateway: Gateway, langfuse_vars: dict[str, JsonValue]) -> None:
    with gateway.scenario() as scenario:
        team_id: Final = scenario.team()
        response: Final = _add_callback(
            gateway, team_id, {**langfuse_vars, SPAN_SCOPE_VAR: ["no_internal"]}
        )
        _bad_var_rejected(response)


def test_callback_var_empty_string_rejected(gateway: Gateway, langfuse_vars: dict[str, JsonValue]) -> None:
    with gateway.scenario() as scenario:
        team_id: Final = scenario.team()
        response: Final = _add_callback(gateway, team_id, {**langfuse_vars, SPAN_SCOPE_VAR: ""})
        _bad_var_rejected(response)


def test_callback_var_oversized_string_rejected(gateway: Gateway, langfuse_vars: dict[str, JsonValue]) -> None:
    with gateway.scenario() as scenario:
        team_id: Final = scenario.team()
        response: Final = _add_callback(gateway, team_id, {**langfuse_vars, SPAN_SCOPE_VAR: "x" * 5120})
        _bad_var_rejected(response)


def test_callback_var_case_sensitive_rejected(gateway: Gateway, langfuse_vars: dict[str, JsonValue]) -> None:
    with gateway.scenario() as scenario:
        team_id: Final = scenario.team()
        response: Final = _add_callback(gateway, team_id, {**langfuse_vars, SPAN_SCOPE_VAR: "No_Internal"})
        _bad_var_rejected(response)


def test_same_span_scope_value_on_second_entry_accepted(gateway: Gateway, audit_sinks: SpanSinks, langfuse_vars: dict[str, JsonValue]) -> None:
    with gateway.scenario() as scenario:
        model: Final = _audit_model(scenario, gateway.upstream_url)
        team_id: Final = scenario.team()
        first: Final = _add_callback(
            gateway, team_id, {**langfuse_vars, SPAN_SCOPE_VAR: "no_internal"}, callback_type="success_and_failure"
        )
        assert first.status_code == 200, first.text
        second: Final = _add_callback(
            gateway, team_id, {**langfuse_vars, SPAN_SCOPE_VAR: "no_internal"}, callback_type="success"
        )
        assert second.status_code == 200, f"identical value rejected: {second.status_code} {second.text}"
        listed: Final = gateway.get(f"/team/{team_id}/callback")
        data: Final = object_value(listed["data"])
        assert "langfuse_otel" in data.get("success_callbacks", []), data
        assert object_value(data["callback_vars"]).get(SPAN_SCOPE_VAR) == "no_internal", data
        key: Final = _key_on_team(scenario, team_id)
        response: Final = _chat(gateway, key, model, _nonce())
        assert response.status_code == 200, response.text
        tenant_trace: Final = _trace_id(audit_sinks.tenant, call_id=_call_id(response))
        _assert_no_internal(_trace_spans(audit_sinks.tenant, tenant_trace))


def test_conflicting_span_scope_value_rejected(gateway: Gateway, langfuse_vars: dict[str, JsonValue]) -> None:
    with gateway.scenario() as scenario:
        team_id: Final = scenario.team()
        first: Final = _add_callback(
            gateway, team_id, {**langfuse_vars, SPAN_SCOPE_VAR: "no_internal"}, callback_type="success_and_failure"
        )
        assert first.status_code == 200, first.text
        second: Final = _add_callback(
            gateway, team_id, {**langfuse_vars, SPAN_SCOPE_VAR: "full"}, callback_type="success"
        )
        assert second.status_code == 400, f"expected 400 conflict, got {second.status_code}: {second.text}"
        assert SPAN_SCOPE_VAR in second.text, second.text


def test_span_scope_on_non_otel_callback_rejected(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        team_id: Final = scenario.team()
        response: Final = _add_callback(
            gateway,
            team_id,
            {"langsmith_api_key": "sk-ls-audit", SPAN_SCOPE_VAR: "no_internal"},
            callback_name="langsmith",
        )
        assert response.status_code == 400, f"expected 400, got {response.status_code}: {response.text}"
        assert "callback" in response.text, response.text


def test_span_scope_on_newrelic_accepted(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        team_id: Final = scenario.team()
        response: Final = _add_callback(
            gateway,
            team_id,
            {"newrelic_api_key": "nr-audit-key", SPAN_SCOPE_VAR: "no_internal"},
            callback_name="newrelic",
        )
        assert response.status_code == 200, f"expected 200, got {response.status_code}: {response.text}"
        listed: Final = gateway.get(f"/team/{team_id}/callback")
        data: Final = object_value(listed["data"])
        assert object_value(data["callback_vars"]).get(SPAN_SCOPE_VAR) == "no_internal", data
        assert "newrelic" in data.get("success_callbacks", []) or "newrelic" in data.get("failure_callbacks", []), data


def test_newrelic_no_internal_over_connect(
    gateway: Gateway, audit_sinks: SpanSinks, newrelic_sink: ConnectSink, otel_audit_config: AuditConfigWriter, tmp_path: Path
) -> None:
    _newrelic_flow(gateway, audit_sinks, newrelic_sink, otel_audit_config, tmp_path, span_scope="no_internal")


def test_newrelic_full_over_connect(
    gateway: Gateway, audit_sinks: SpanSinks, newrelic_sink: ConnectSink, otel_audit_config: AuditConfigWriter, tmp_path: Path
) -> None:
    _newrelic_flow(gateway, audit_sinks, newrelic_sink, otel_audit_config, tmp_path, span_scope="full")


def _newrelic_flow(
    gateway: Gateway,
    audit_sinks: SpanSinks,
    newrelic_sink: ConnectSink,
    otel_audit_config: AuditConfigWriter,
    tmp_path: Path,
    *,
    span_scope: str,
) -> None:
    overrides: Final = {
        "LITELLM_OTEL_V2": "1",
        "ARIZE_HTTP_ENDPOINT": audit_sinks.arize,
        "HTTPS_PROXY": newrelic_sink.proxy_url,
        "NO_PROXY": "127.0.0.1,localhost",
        "OTEL_EXPORTER_OTLP_TRACES_CERTIFICATE": newrelic_sink.ca_pem,
    }
    with owned_proxy(gateway, tmp_path, overrides, config=otel_audit_config(tmp_path, {}), workers=2) as candidate:
        with candidate.scenario() as scenario:
            model: Final = _audit_model(scenario, candidate.upstream_url)
            team_id: Final = scenario.team()
            callback: Final = _add_callback(
                candidate,
                team_id,
                {"newrelic_api_key": "nr-synthetic-audit", SPAN_SCOPE_VAR: span_scope},
                callback_name="newrelic",
            )
            assert callback.status_code == 200, callback.text
            key: Final = _key_on_team(scenario, team_id)
            response: Final = _chat(candidate, key, model, _nonce())
            assert response.status_code == 200, response.text
            call_id: Final = _call_id(response)
            trace: Final = _trace_id(newrelic_sink.control_url, call_id=call_id)
            group: Final = _trace_spans(newrelic_sink.control_url, trace)
            if span_scope == "no_internal":
                _assert_no_internal(group)
            else:
                _assert_full(group)
            requests: Final = recorded_requests(newrelic_sink.control_url)
            assert any(entry.get("connect") == "otlp.nr-data.net:443" for entry in requests), requests
            posted: Final = tuple(entry for entry in requests if entry.get("path") == "/v1/traces")
            assert posted, f"tunnel saw no /v1/traces posts: {requests}"
            for entry in posted:
                assert entry.get("host") == "otlp.nr-data.net", entry
                headers: Final = entry.get("headers")
                assert isinstance(headers, Mapping) and headers.get("api-key") == "nr-synthetic-audit", entry


def test_arize_grpc_no_internal(
    gateway: Gateway, audit_sinks: SpanSinks, arize_grpc_sink: GrpcSink, otel_audit_config: AuditConfigWriter, tmp_path: Path
) -> None:
    _arize_grpc_flow(gateway, audit_sinks, arize_grpc_sink, otel_audit_config, tmp_path, span_scope="no_internal")


def test_arize_grpc_full(
    gateway: Gateway, audit_sinks: SpanSinks, arize_grpc_sink: GrpcSink, otel_audit_config: AuditConfigWriter, tmp_path: Path
) -> None:
    _arize_grpc_flow(gateway, audit_sinks, arize_grpc_sink, otel_audit_config, tmp_path, span_scope="full")


def _arize_grpc_flow(
    gateway: Gateway,
    audit_sinks: SpanSinks,
    arize_grpc_sink: GrpcSink,
    otel_audit_config: AuditConfigWriter,
    tmp_path: Path,
    *,
    span_scope: str,
) -> None:
    overrides: Final = {
        "LITELLM_OTEL_V2": "1",
        "ARIZE_ENDPOINT": arize_grpc_sink.url,
        "ARIZE_HTTP_ENDPOINT": audit_sinks.arize,
    }
    with owned_proxy(gateway, tmp_path, overrides, config=otel_audit_config(tmp_path, {}), workers=2) as candidate:
        with candidate.scenario() as scenario:
            model: Final = _audit_model(scenario, candidate.upstream_url)
            team_id: Final = scenario.team()
            callback: Final = _add_callback(
                candidate,
                team_id,
                {**ARIZE_VARS, SPAN_SCOPE_VAR: span_scope},
                callback_name="arize",
            )
            assert callback.status_code == 200, callback.text
            key: Final = _key_on_team(scenario, team_id)
            response: Final = _chat(candidate, key, model, _nonce())
            assert response.status_code == 200, response.text
            call_id: Final = _call_id(response)
            trace: Final = _trace_id(arize_grpc_sink.control_url, call_id=call_id)
            group: Final = _trace_spans(arize_grpc_sink.control_url, trace)
            if span_scope == "no_internal":
                _assert_no_internal(group)
            else:
                _assert_full(group)
            requests: Final = recorded_requests(arize_grpc_sink.control_url)
            exports: Final = tuple(entry for entry in requests if entry.get("grpc") == "Export")
            assert exports, f"grpc sink saw no Export calls: {requests}"
            for entry in exports:
                metadata: Final = entry.get("metadata")
                assert isinstance(metadata, Mapping), entry
                assert metadata.get("arize-space-id") == ARIZE_VARS["arize_space_id"], entry
                assert metadata.get("api_key") == ARIZE_VARS["arize_api_key"], entry


def test_unauthenticated_callback_post_rejected(gateway: Gateway, langfuse_vars: dict[str, JsonValue]) -> None:
    with httpx.Client(base_url=str(gateway.client.base_url), timeout=15, trust_env=False) as client:
        response: Final = client.post(
            "/team/some-team/callback", json={"callback_name": "langfuse_otel", "callback_vars": dict(langfuse_vars)}
        )
    assert response.status_code == 401, f"expected 401, got {response.status_code}: {response.text}"


def test_key_generate_bogus_span_scope_rejected(gateway: Gateway, langfuse_vars: dict[str, JsonValue]) -> None:
    response: Final = gateway.request(
        "POST", "/key/generate", {"metadata": {"logging": _key_logging_entry({**langfuse_vars, SPAN_SCOPE_VAR: "bogus"})}}
    )
    assert response.status_code == 400, f"expected 400, got {response.status_code}: {response.text}"


def test_key_update_bogus_span_scope_rejected(gateway: Gateway, langfuse_vars: dict[str, JsonValue]) -> None:
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        response: Final = gateway.request(
            "POST",
            "/key/update",
            {"key": key, "metadata": {"logging": _key_logging_entry({**langfuse_vars, SPAN_SCOPE_VAR: "bogus"})}},
        )
        assert response.status_code == 400, f"expected 400, got {response.status_code}: {response.text}"


def test_team_logging_metadata_rejects_bogus_span_scope(gateway: Gateway, audit_sinks: SpanSinks, langfuse_vars: dict[str, JsonValue]) -> None:
    with gateway.scenario() as scenario:
        model: Final = _audit_model(scenario, gateway.upstream_url)
        team_id: Final = scenario.team()
        registered: Final = _add_callback(gateway, team_id, langfuse_vars)
        assert registered.status_code == 200, registered.text
        bogus_metadata: Final[Mapping[str, JsonValue]] = {
            "logging": _key_logging_entry({**langfuse_vars, SPAN_SCOPE_VAR: "bogus"})
        }
        update: Final = gateway.request(
            "POST", "/team/update", {"team_id": team_id, "metadata": bogus_metadata}
        )
        assert update.status_code == 400, update.text
        assert "otel_span_scope" in update.text, update.text
        created: Final = gateway.request("POST", "/team/new", {"metadata": bogus_metadata})
        assert created.status_code == 400, created.text
        assert "otel_span_scope" in created.text, created.text
        key: Final = _key_on_team(scenario, team_id)
        response: Final = _chat(gateway, key, model, _nonce())
        assert response.status_code == 200, response.text
        call_id: Final = _call_id(response)
        tenant_trace: Final = _trace_id(audit_sinks.tenant, call_id=call_id)
        _assert_full(_trace_spans(audit_sinks.tenant, tenant_trace))
        unrelated_key: Final = scenario.key()
        unrelated: Final = _chat(gateway, unrelated_key, model, _nonce())
        assert unrelated.status_code == 200, unrelated.text


def _sink_status_flow(
    gateway: Gateway, audit_sinks: SpanSinks, langfuse_vars: Mapping[str, JsonValue], status: int
) -> None:
    configure_sink(audit_sinks.tenant, status=status)
    try:
        with gateway.scenario() as scenario:
            model: Final = _audit_model(scenario, gateway.upstream_url)
            team_id: Final = scenario.team()
            callback: Final = _add_callback(gateway, team_id, langfuse_vars)
            assert callback.status_code == 200, callback.text
            key: Final = _key_on_team(scenario, team_id)
            response: Final = _chat(gateway, key, model, _nonce())
            assert response.status_code == 200, f"caller broke on sink {status}: {response.status_code} {response.text}"
            call_id: Final = _call_id(response)
            operator_trace: Final = _trace_id(audit_sinks.operator, call_id=call_id)
            _assert_full(_trace_spans(audit_sinks.operator, operator_trace))
            response_id: Final = _response_id(response.json())
            rows: Final = eventually(
                lambda: read_rows(
                    'SELECT spend FROM "LiteLLM_SpendLogs" WHERE request_id IN (%s, %s)',
                    (str(call_id), str(response_id)),
                ),
                lambda values: len(values) >= 1,
                seconds=70,
            )
            assert rows, "no spend row"
    finally:
        configure_sink(audit_sinks.tenant, status=200)


def test_tenant_sink_403_does_not_break_caller(
    gateway: Gateway, audit_sinks: SpanSinks, langfuse_vars: dict[str, JsonValue]
) -> None:
    _sink_status_flow(gateway, audit_sinks, langfuse_vars, 403)


def test_tenant_sink_404_does_not_break_caller(
    gateway: Gateway, audit_sinks: SpanSinks, langfuse_vars: dict[str, JsonValue]
) -> None:
    _sink_status_flow(gateway, audit_sinks, langfuse_vars, 404)


def test_upstream_500_under_no_internal_keeps_span_scope_back(gateway: Gateway, audit_sinks: SpanSinks, langfuse_vars: dict[str, JsonValue]) -> None:
    upstream_model: Final = "audit-chat"
    httpx.post(
        f"{gateway.upstream_url}/__scripts/{upstream_model}", json={"statuses": [500]}, trust_env=False, timeout=15
    ).raise_for_status()
    try:
        with gateway.scenario() as scenario:
            model: Final = _audit_model(scenario, gateway.upstream_url)
            team_id: Final = scenario.team()
            callback: Final = _add_callback(gateway, team_id, {**langfuse_vars, SPAN_SCOPE_VAR: "no_internal"})
            assert callback.status_code == 200, callback.text
            key: Final = _key_on_team(scenario, team_id)
            response: Final = _chat(gateway, key, model, _nonce())
            assert response.status_code >= 500, f"expected caller 5xx, got {response.status_code}: {response.text}"
            call_id: Final = _call_id(response)

            def tenant_group() -> tuple[Span, ...] | None:
                _, spans = recorded_spans(audit_sinks.tenant)
                group: Final = tuple(
                    span
                    for span in spans
                    if span["attributes"].get("litellm.call_id") == call_id
                    or span["trace_id"] in {s["trace_id"] for s in spans if s["attributes"].get("litellm.call_id") == call_id}
                )
                return group if group else None

            group: Final = eventually(tenant_group, lambda value: value is not None, seconds=40)
            assert group is not None, "tenant sink never received the failed-request trace"
            counts: Final = _classes(group)
            assert counts["internal"] == 0, f"internal spans leaked on error trace: {group}"
    finally:
        httpx.delete(f"{gateway.upstream_url}/__scripts/{upstream_model}", trust_env=False, timeout=15)


def test_unrelated_key_unaffected_by_tenant_sink_failure(gateway: Gateway, audit_sinks: SpanSinks) -> None:
    configure_sink(audit_sinks.tenant, status=403)
    try:
        with gateway.scenario() as scenario:
            model: Final = _audit_model(scenario, gateway.upstream_url)
            key: Final = scenario.key()
            response: Final = _chat(gateway, key, model, _nonce())
            assert response.status_code == 200, f"unrelated key broke: {response.status_code} {response.text}"
    finally:
        configure_sink(audit_sinks.tenant, status=200)


def test_key_health_with_no_internal_team_callback(gateway: Gateway, langfuse_vars: dict[str, JsonValue]) -> None:
    with gateway.scenario() as scenario:
        team_id: Final = scenario.team()
        callback: Final = _add_callback(gateway, team_id, {**langfuse_vars, SPAN_SCOPE_VAR: "no_internal"})
        assert callback.status_code == 200, callback.text
        key: Final = _key_on_team(scenario, team_id)
        response: Final = gateway.request("POST", "/key/health", key=key)
        assert response.status_code == 200, f"/key/health failed: {response.status_code} {response.text}"


def test_callback_var_update_full_to_no_internal_takes_effect(gateway: Gateway, audit_sinks: SpanSinks, langfuse_vars: dict[str, JsonValue]) -> None:
    with gateway.scenario() as scenario:
        model: Final = _audit_model(scenario, gateway.upstream_url)
        team_id: Final = scenario.team()
        first: Final = _add_callback(gateway, team_id, {**langfuse_vars, SPAN_SCOPE_VAR: "full"})
        assert first.status_code == 200, first.text
        key: Final = _key_on_team(scenario, team_id)
        warm: Final = _chat(gateway, key, model, _nonce())
        assert warm.status_code == 200, warm.text
        warm_trace: Final = _trace_id(audit_sinks.tenant, call_id=_call_id(warm))
        _assert_full(_trace_spans(audit_sinks.tenant, warm_trace))
        deleted: Final = gateway.request("DELETE", f"/team/{team_id}/callback/langfuse_otel")
        assert deleted.status_code == 200, deleted.text
        updated: Final = _add_callback(gateway, team_id, {**langfuse_vars, SPAN_SCOPE_VAR: "no_internal"})
        assert updated.status_code == 200, updated.text

        issued: Final[list[str | None]] = []

        def flipped() -> str | None:
            response: Final = _chat(gateway, key, model, _nonce())
            assert response.status_code == 200, response.text
            issued.append(_call_id(response))
            _, spans = recorded_spans(audit_sinks.tenant)
            group: Final = tuple(
                span
                for span in spans
                if span["attributes"].get("litellm.call_id") in issued
            )
            if not group:
                return None
            newest: Final = next(
                (span for span in reversed(group) if span["attributes"].get("litellm.call_id") == issued[-1]),
                group[-1],
            )
            full_group: Final = spans_for_trace(spans, str(newest["trace_id"]))
            if _classes(full_group)["internal"] == 0:
                return str(newest["trace_id"])
            return None

        trace: Final = eventually(flipped, lambda value: value is not None, seconds=70)
        assert trace is not None, "no_internal never took effect within the cache TTL"


def test_callback_delete_stops_tenant_export(gateway: Gateway, audit_sinks: SpanSinks, langfuse_vars: dict[str, JsonValue]) -> None:
    with gateway.scenario() as scenario:
        model: Final = _audit_model(scenario, gateway.upstream_url)
        team_id: Final = scenario.team()
        callback: Final = _add_callback(gateway, team_id, langfuse_vars)
        assert callback.status_code == 200, callback.text
        key: Final = _key_on_team(scenario, team_id)
        warm: Final = _chat(gateway, key, model, _nonce())
        assert warm.status_code == 200, warm.text
        _trace_id(audit_sinks.tenant, call_id=_call_id(warm))
        deleted: Final = gateway.request("DELETE", f"/team/{team_id}/callback/langfuse_otel")
        assert deleted.status_code == 200, deleted.text

        def drained() -> str | None:
            response: Final = _chat(gateway, key, model, _nonce())
            if response.status_code != 200:
                return None
            call_id: Final = _call_id(response)
            operator_trace: Final = _trace_id(audit_sinks.operator, call_id=call_id)
            return call_id if operator_trace else None

        call_id: Final = eventually(drained, lambda value: value is not None, seconds=70)
        assert call_id is not None
        _, spans = recorded_spans(audit_sinks.tenant)
        leaked: Final = tuple(
            span for span in spans if span["attributes"].get("litellm.call_id") == call_id
        )
        assert not leaked, f"tenant sink still received spans after callback delete: {leaked}"


def test_identical_requests_export_exactly_once(gateway: Gateway, audit_sinks: SpanSinks, langfuse_vars: dict[str, JsonValue]) -> None:
    with gateway.scenario() as scenario:
        model: Final = _audit_model(scenario, gateway.upstream_url)
        team_id: Final = scenario.team()
        callback: Final = _add_callback(gateway, team_id, {**langfuse_vars, SPAN_SCOPE_VAR: "no_internal"})
        assert callback.status_code == 200, callback.text
        key: Final = _key_on_team(scenario, team_id)
        call_ids: Final = []
        response_ids: Final = []
        for _ in range(5):
            response: Final = _chat(gateway, key, model, _nonce())
            assert response.status_code == 200, response.text
            call_ids.append(_call_id(response))
            response_ids.append(_response_id(response.json()))
        for call_id, response_id in zip(call_ids, response_ids):
            tenant_trace: Final = _trace_id(audit_sinks.tenant, call_id=call_id, response_id=response_id)
            _, spans = recorded_spans(audit_sinks.tenant)
            matching: Final = tuple(
                span
                for span in spans_for_trace(spans, tenant_trace)
                if "gen_ai.operation.name" in span["attributes"]
            )
            assert len(matching) == 1, f"model span for {response_id} exported {len(matching)} times"
            operator_trace: Final = _trace_id(audit_sinks.operator, call_id=call_id, response_id=response_id)
            _, operator_spans = recorded_spans(audit_sinks.operator)
            operator_matching: Final = tuple(
                span
                for span in spans_for_trace(operator_spans, operator_trace)
                if span["attributes"].get("gen_ai.response.id") == response_id
            )
            assert len(operator_matching) == 1, f"operator exported {response_id} {len(operator_matching)} times"


def test_concurrent_requests_all_no_internal_once(gateway: Gateway, audit_sinks: SpanSinks, langfuse_vars: dict[str, JsonValue]) -> None:

    with gateway.scenario() as scenario:
        model: Final = _audit_model(scenario, gateway.upstream_url)
        team_id: Final = scenario.team()
        callback: Final = _add_callback(gateway, team_id, {**langfuse_vars, SPAN_SCOPE_VAR: "no_internal"})
        assert callback.status_code == 200, callback.text
        key: Final = _key_on_team(scenario, team_id)
        responses: Final[list[httpx.Response]] = []

        def hit() -> None:
            responses.append(_chat(gateway, key, model, _nonce()))

        threads: Final = [threading.Thread(target=hit) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        assert len(responses) == 10
        for response in responses:
            assert response.status_code == 200, response.text
            call_id: Final = _call_id(response)
            tenant_trace: Final = _trace_id(audit_sinks.tenant, call_id=call_id)
            group: Final = _trace_spans(audit_sinks.tenant, tenant_trace)
            _assert_no_internal(group)
            workers: Final = {
                str(span["resource"].get("process.pid")) for span in group
            }
            assert workers, "no process attribution on tenant spans"


def test_failure_only_callback_entry_anchors_no_destination(gateway: Gateway, audit_sinks: SpanSinks, langfuse_vars: dict[str, JsonValue]) -> None:
    with gateway.scenario() as scenario:
        model: Final = _audit_model(scenario, gateway.upstream_url)
        team_id: Final = scenario.team()
        callback: Final = _add_callback(
            gateway,
            team_id,
            {**langfuse_vars, SPAN_SCOPE_VAR: "no_internal"},
            callback_type="failure",
        )
        assert callback.status_code == 200, callback.text
        key: Final = _key_on_team(scenario, team_id)
        response: Final = _chat(gateway, key, model, _nonce())
        assert response.status_code == 200, response.text
        call_id: Final = _call_id(response)
        operator_trace: Final = _trace_id(audit_sinks.operator, call_id=call_id)
        _assert_full(_trace_spans(audit_sinks.operator, operator_trace))
        _, spans = recorded_spans(audit_sinks.tenant)
        leaked: Final = tuple(
            span for span in spans if span["attributes"].get("litellm.call_id") == call_id
        )
        assert not leaked, f"failure-only entry anchored a tenant destination: {leaked}"
