from __future__ import annotations

import os
import signal
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.otlp_sink import Span, SpanSinks, configure_sink, recorded_spans, sink_pid, span_class
from integration._support.process import owned_proxy, owned_proxy_process
from pydantic import JsonValue

AuditConfigWriter = Callable[[Path, Mapping[str, JsonValue]], Path]
SPAN_SCOPE_VAR: Final = "otel_span_scope"


@pytest.fixture(scope="module")
def gateway(
    audit_sinks: SpanSinks,
    otel_audit_config: AuditConfigWriter,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("otel-audit-chaos-proxy")
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
    return f"otelchaos-{uuid.uuid4().hex}"


def _classes(spans: tuple[Span, ...]) -> dict[str, int]:
    return {name: sum(1 for span in spans if span_class(span) == name) for name in ("root", "tenant", "internal")}


def _send(gateway: Gateway, key: str, model: str, nonce: str, index: int) -> httpx.Response:
    if index % 3 == 0:
        return gateway.request("POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": nonce}]}, key=key)
    if index % 3 == 1:
        return gateway.request(
            "POST", "/v1/messages", {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": nonce}]}, key=key
        )
    return gateway.request("POST", "/v1/responses", {"model": model, "input": nonce}, key=key)


def _send_burst(gateway: Gateway, key: str, model: str, count: int) -> list[httpx.Response]:
    responses: Final[list[httpx.Response]] = []
    lock: Final = threading.Lock()

    def hit(index: int) -> None:
        nonce: Final = _nonce()
        if index % 2 == 0:
            response: Final = _send(gateway, key, model, nonce, index)
        else:
            response = gateway.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": nonce}], "stream": True},
                key=key,
            )
        with lock:
            responses.append(response)

    threads: Final = [threading.Thread(target=hit, args=(i,)) for i in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=90)
    return responses


def _wait_trace_count(sink_url: str, call_ids: list[str | None], seconds: float) -> tuple[Span, ...]:
    wanted: Final = {call_id for call_id in call_ids if call_id}

    def gathered() -> tuple[Span, ...] | None:
        _, spans = recorded_spans(sink_url)
        covered: Final = {
            span["attributes"].get("litellm.call_id") for span in spans
        }
        return spans if wanted <= covered else None

    result: Final = eventually(gathered, lambda value: value is not None, seconds=seconds)
    assert result is not None
    return result


def test_frozen_tenant_sink_receives_every_span_after_resume(gateway: Gateway, audit_sinks: SpanSinks, langfuse_vars: dict[str, JsonValue]) -> None:
    pid: Final = sink_pid(audit_sinks.tenant)
    with gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/audit-chat", api_base=f"{gateway.upstream_url}/v1")
        team_id: Final = scenario.team()
        callback: Final = gateway.request(
            "POST", f"/team/{team_id}/callback", {"callback_name": "langfuse_otel", "callback_vars": {**langfuse_vars, SPAN_SCOPE_VAR: "no_internal"}}
        )
        assert callback.status_code == 200, callback.text
        key: Final = scenario.key(team_id=team_id)
        os.kill(pid, signal.SIGSTOP)
        try:
            responses: Final = _send_burst(gateway, key, model, 30)
        finally:
            os.kill(pid, signal.SIGCONT)
        assert all(response.status_code == 200 for response in responses), [r.status_code for r in responses]
        call_ids: Final = [response.headers.get("x-litellm-call-id") for response in responses]
        spans: Final = _wait_trace_count(audit_sinks.tenant, call_ids, seconds=120)
        for call_id in call_ids:
            group: Final = tuple(
                span for span in spans if span["attributes"].get("litellm.call_id") == call_id
            )
            assert group, f"call {call_id} never reached the tenant sink"
            model_spans: Final = tuple(span for span in group if "gen_ai.operation.name" in span["attributes"])
            assert len(model_spans) == 1, f"call {call_id} exported {len(model_spans)} times"
            assert _classes(group)["internal"] == 0, f"internal spans leaked for {call_id}"


def test_sink_outage_keeps_diagnostics_green_and_recovers(gateway: Gateway, audit_sinks: SpanSinks, langfuse_vars: dict[str, JsonValue]) -> None:
    pid: Final = sink_pid(audit_sinks.tenant)
    with gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/audit-chat", api_base=f"{gateway.upstream_url}/v1")
        team_id: Final = scenario.team()
        callback: Final = gateway.request(
            "POST", f"/team/{team_id}/callback", {"callback_name": "langfuse_otel", "callback_vars": {**langfuse_vars, SPAN_SCOPE_VAR: "no_internal"}}
        )
        assert callback.status_code == 200, callback.text
        key: Final = scenario.key(team_id=team_id)
        os.kill(pid, signal.SIGSTOP)
        try:
            responses: Final = _send_burst(gateway, key, model, 20)
            readiness: Final = gateway.request("GET", "/health/readiness")
            assert readiness.status_code == 200, readiness.text
            details: Final = gateway.request("GET", "/health/readiness/details")
            assert details.status_code == 200, details.text
            success_callbacks: Final = details.json().get("success_callbacks", [])
            assert "OpenTelemetryV2" in success_callbacks, f"otel callback missing during outage: {details.json()}"
            liveliness: Final = gateway.request("GET", "/health/liveliness")
            assert liveliness.status_code == 200, liveliness.text
        finally:
            os.kill(pid, signal.SIGCONT)
        assert all(response.status_code == 200 for response in responses), [r.status_code for r in responses]
        call_ids: Final = [response.headers.get("x-litellm-call-id") for response in responses]
        spans: Final = _wait_trace_count(audit_sinks.tenant, call_ids, seconds=120)
        for call_id in call_ids:
            group: Final = tuple(
                span for span in spans if span["attributes"].get("litellm.call_id") == call_id
            )
            assert group, f"call {call_id} never reached the tenant sink"
            model_spans: Final = tuple(span for span in group if "gen_ai.operation.name" in span["attributes"])
            assert len(model_spans) == 1, f"call {call_id} exported {len(model_spans)} times"


def test_slow_tenant_sink_exports_each_span_once(gateway: Gateway, audit_sinks: SpanSinks, langfuse_vars: dict[str, JsonValue]) -> None:
    configure_sink(audit_sinks.tenant, delay_seconds=2.0)
    try:
        with gateway.scenario() as scenario:
            model: Final = scenario.model(model="openai/audit-chat", api_base=f"{gateway.upstream_url}/v1")
            team_id: Final = scenario.team()
            callback: Final = gateway.request(
                "POST", f"/team/{team_id}/callback", {"callback_name": "langfuse_otel", "callback_vars": {**langfuse_vars, SPAN_SCOPE_VAR: "no_internal"}}
            )
            assert callback.status_code == 200, callback.text
            key: Final = scenario.key(team_id=team_id)
            responses: Final = _send_burst(gateway, key, model, 20)
            assert all(response.status_code == 200 for response in responses), [r.status_code for r in responses]
            call_ids: Final = [response.headers.get("x-litellm-call-id") for response in responses]
            spans: Final = _wait_trace_count(audit_sinks.tenant, call_ids, seconds=120)
            for call_id in call_ids:
                group: Final = tuple(
                    span for span in spans if span["attributes"].get("litellm.call_id") == call_id
                )
                assert group, f"call {call_id} never reached the slow sink"
                span_ids: Final = [span["span_id"] for span in group]
                assert len(span_ids) == len(set(span_ids)), f"duplicate spans for {call_id}"
    finally:
        configure_sink(audit_sinks.tenant, delay_seconds=0.0)


def test_proxy_restart_mid_burst_keeps_serving(gateway: Gateway, audit_sinks: SpanSinks, langfuse_vars: dict[str, JsonValue], otel_audit_config: AuditConfigWriter, tmp_path: Path) -> None:
    path: Final = otel_audit_config(tmp_path, {})
    overrides: Final = {"LITELLM_OTEL_V2": "1", "ARIZE_HTTP_ENDPOINT": audit_sinks.arize}
    with owned_proxy(gateway, tmp_path, overrides, config=path, workers=2) as candidate:
        with candidate.scenario() as scenario:
            model: Final = scenario.model(model="openai/audit-chat", api_base=f"{candidate.upstream_url}/v1")
            team_id: Final = scenario.team()
            callback: Final = candidate.request(
                "POST", f"/team/{team_id}/callback", {"callback_name": "langfuse_otel", "callback_vars": langfuse_vars}
            )
            assert callback.status_code == 200, callback.text
            key: Final = scenario.key(team_id=team_id)
            first: Final = candidate.request(
                "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": _nonce()}]}, key=key
            )
            assert first.status_code == 200, first.text
            first_call_id: Final = first.headers.get("x-litellm-call-id")

            def first_landed() -> bool:
                _, spans = recorded_spans(audit_sinks.tenant)
                return any(span["attributes"].get("litellm.call_id") == first_call_id for span in spans)

            assert eventually(first_landed, bool, seconds=40), "pre-restart trace never reached the tenant sink"
    with owned_proxy(gateway, tmp_path, overrides, config=path, workers=2) as candidate:
        with candidate.scenario() as scenario:
            model: Final = scenario.model(model="openai/audit-chat", api_base=f"{candidate.upstream_url}/v1")
            team_id: Final = scenario.team()
            callback: Final = candidate.request(
                "POST", f"/team/{team_id}/callback", {"callback_name": "langfuse_otel", "callback_vars": langfuse_vars}
            )
            assert callback.status_code == 200, callback.text
            key: Final = scenario.key(team_id=team_id)
            response: Final = candidate.request(
                "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": _nonce()}]}, key=key
            )
            assert response.status_code == 200, f"post-restart request failed: {response.status_code} {response.text}"
            call_id: Final = response.headers.get("x-litellm-call-id")

        def landed() -> bool:
            _, spans = recorded_spans(audit_sinks.operator)
            return any(span["attributes"].get("litellm.call_id") == call_id for span in spans)

        assert eventually(landed, bool, seconds=40), "post-restart trace never reached the operator sink"


def test_killing_one_worker_leaves_serving(
    gateway: Gateway,
    audit_sinks: SpanSinks,
    langfuse_vars: dict[str, JsonValue],
    otel_audit_config: AuditConfigWriter,
    tmp_path: Path,
) -> None:
    import psutil

    overrides: Final = {"LITELLM_OTEL_V2": "1", "ARIZE_HTTP_ENDPOINT": audit_sinks.arize}
    with owned_proxy_process(gateway, tmp_path, overrides, config=otel_audit_config(tmp_path, {}), workers=2) as owned:
        candidate: Final = owned.gateway
        children: Final = psutil.Process(owned.process.pid).children(recursive=True)
        assert children, "audit proxy has no worker children to kill"
        with candidate.scenario() as scenario:
            model: Final = scenario.model(model="openai/audit-chat", api_base=f"{candidate.upstream_url}/v1")
            team_id: Final = scenario.team()
            callback: Final = candidate.request(
                "POST", f"/team/{team_id}/callback", {"callback_name": "langfuse_otel", "callback_vars": langfuse_vars}
            )
            assert callback.status_code == 200, callback.text
            key: Final = scenario.key(team_id=team_id)
            victim: Final = children[-1]
            victim.terminate()
            responses: Final = _send_burst(candidate, key, model, 10)
            assert all(response.status_code == 200 for response in responses), [r.status_code for r in responses]
