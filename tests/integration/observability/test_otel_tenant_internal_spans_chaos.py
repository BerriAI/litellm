from __future__ import annotations

import json
import os
import signal
import threading
import uuid
from pathlib import Path
from typing import Final
from urllib.parse import urlparse

import httpx
import pytest
from pydantic import JsonValue

from integration._support.client import Gateway, eventually
from integration._support.otlp_sink import configure_sink, recorded_spans, sink_pid, span_class, spans_for_trace

SINK_OPERATOR: Final = os.environ.get("OTEL_AUDIT_OPERATOR_SINK", "http://127.0.0.1:8191")
SINK_TENANT: Final = os.environ.get("OTEL_AUDIT_TENANT_SINK", "http://127.0.0.1:8192")
INTERNAL_SPANS_VAR: Final = "otel_internal_spans"
LANGFUSE_VARS: Final[dict[str, str]] = {
    "langfuse_public_key": "pk-lf-audit",
    "langfuse_secret_key": "sk-lf-audit",
    "langfuse_host": SINK_TENANT,
}


def _nonce() -> str:
    return f"otelchaos-{uuid.uuid4().hex}"


def _classes(spans: tuple[dict[str, JsonValue], ...]) -> dict[str, int]:
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


def _wait_trace_count(sink_url: str, call_ids: list[str | None], seconds: float) -> tuple[dict[str, JsonValue], ...]:
    wanted: Final = {call_id for call_id in call_ids if call_id}

    def gathered() -> tuple[dict[str, JsonValue], ...] | None:
        _, spans = recorded_spans(sink_url)
        covered: Final = {
            (span["attributes"] or {}).get("litellm.call_id") for span in spans  # type: ignore[union-attr]
        }
        return spans if wanted <= covered else None

    result: Final = eventually(gathered, lambda value: value is not None, seconds=seconds)
    assert result is not None
    return result


@pytest.mark.covers("other.observability.otel.tenant_internal_spans.c1_frozen_sink_delivers_after_resume")
def test_frozen_tenant_sink_receives_every_span_after_resume(gateway: Gateway) -> None:
    pid: Final = sink_pid(SINK_TENANT)
    with gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/audit-chat", api_base=f"{gateway.upstream_url}/v1")
        team_id: Final = scenario.team()
        callback: Final = gateway.request(
            "POST", f"/team/{team_id}/callback", {"callback_name": "langfuse_otel", "callback_vars": {**LANGFUSE_VARS, INTERNAL_SPANS_VAR: "exclude"}}
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
        spans: Final = _wait_trace_count(SINK_TENANT, call_ids, seconds=120)
        for call_id in call_ids:
            group: Final = tuple(
                span for span in spans if (span["attributes"] or {}).get("litellm.call_id") == call_id  # type: ignore[union-attr]
            )
            assert group, f"call {call_id} never reached the tenant sink"
            model_spans: Final = tuple(span for span in group if "gen_ai.operation.name" in (span["attributes"] or {}))  # type: ignore[union-attr]
            assert len(model_spans) == 1, f"call {call_id} exported {len(model_spans)} times"
            assert _classes(group)["internal"] == 0, f"internal spans leaked for {call_id}"


@pytest.mark.covers("other.observability.otel.tenant_internal_spans.c3_slow_sink_no_duplicates")
def test_slow_tenant_sink_exports_each_span_once(gateway: Gateway) -> None:
    configure_sink(SINK_TENANT, delay_seconds=2.0)
    try:
        with gateway.scenario() as scenario:
            model: Final = scenario.model(model="openai/audit-chat", api_base=f"{gateway.upstream_url}/v1")
            team_id: Final = scenario.team()
            callback: Final = gateway.request(
                "POST", f"/team/{team_id}/callback", {"callback_name": "langfuse_otel", "callback_vars": {**LANGFUSE_VARS, INTERNAL_SPANS_VAR: "exclude"}}
            )
            assert callback.status_code == 200, callback.text
            key: Final = scenario.key(team_id=team_id)
            responses: Final = _send_burst(gateway, key, model, 20)
            assert all(response.status_code == 200 for response in responses), [r.status_code for r in responses]
            call_ids: Final = [response.headers.get("x-litellm-call-id") for response in responses]
            spans: Final = _wait_trace_count(SINK_TENANT, call_ids, seconds=120)
            for call_id in call_ids:
                group: Final = tuple(
                    span for span in spans if (span["attributes"] or {}).get("litellm.call_id") == call_id  # type: ignore[union-attr]
                )
                assert group, f"call {call_id} never reached the slow sink"
                span_ids: Final = [span["span_id"] for span in group]
                assert len(span_ids) == len(set(span_ids)), f"duplicate spans for {call_id}"
    finally:
        configure_sink(SINK_TENANT, delay_seconds=0.0)


@pytest.mark.covers("other.observability.otel.tenant_internal_spans.c4_proxy_restart_keeps_serving")
def test_proxy_restart_mid_burst_keeps_serving(gateway: Gateway, tmp_path: Path) -> None:
    import yaml

    from integration._support.otlp_sink import spans_for_trace as _trace
    from integration._support.process import owned_proxy

    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["litellm_settings"] = {
        **config.get("litellm_settings", {}),
        "callbacks": ["otel"],
        "provider_url_destination_allowed_hosts": [urlparse(SINK_TENANT).netloc],
    }
    config["callback_settings"] = {"otel": {"exporter": "http/json", "endpoint": SINK_OPERATOR, "use_simple_processor": True}}
    path: Final = tmp_path / "audit-restart.yaml"
    path.write_text(yaml.safe_dump(config))
    with owned_proxy(gateway, tmp_path, {"LITELLM_OTEL_V2": "1"}, config=path, num_workers=2) as candidate:
        with candidate.scenario() as scenario:
            model: Final = scenario.model(model="openai/audit-chat", api_base=f"{candidate.upstream_url}/v1")
            team_id: Final = scenario.team()
            callback: Final = candidate.request(
                "POST", f"/team/{team_id}/callback", {"callback_name": "langfuse_otel", "callback_vars": LANGFUSE_VARS}
            )
            assert callback.status_code == 200, callback.text
            key: Final = scenario.key(team_id=team_id)
            first: Final = candidate.request(
                "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": _nonce()}]}, key=key
            )
            assert first.status_code == 200, first.text
            first_call_id: Final = first.headers.get("x-litellm-call-id")

            def first_landed() -> bool:
                _, spans = recorded_spans(SINK_TENANT)
                return any((span["attributes"] or {}).get("litellm.call_id") == first_call_id for span in spans)  # type: ignore[union-attr]

            assert eventually(first_landed, bool, seconds=40), "pre-restart trace never reached the tenant sink"
    with owned_proxy(gateway, tmp_path, {"LITELLM_OTEL_V2": "1"}, config=path, num_workers=2) as candidate:
        with candidate.scenario() as scenario:
            model = scenario.model(model="openai/audit-chat", api_base=f"{candidate.upstream_url}/v1")
            team_id = scenario.team()
            callback = candidate.request(
                "POST", f"/team/{team_id}/callback", {"callback_name": "langfuse_otel", "callback_vars": LANGFUSE_VARS}
            )
            assert callback.status_code == 200, callback.text
            key = scenario.key(team_id=team_id)
            response: Final = candidate.request(
                "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": _nonce()}]}, key=key
            )
            assert response.status_code == 200, f"post-restart request failed: {response.status_code} {response.text}"
            call_id: Final = response.headers.get("x-litellm-call-id")

        def landed() -> bool:
            _, spans = recorded_spans(SINK_OPERATOR)
            return any((span["attributes"] or {}).get("litellm.call_id") == call_id for span in spans)  # type: ignore[union-attr]

        assert eventually(landed, bool, seconds=40), "post-restart trace never reached the operator sink"


@pytest.mark.covers("other.observability.otel.tenant_internal_spans.c5_worker_kill_survivor_serves")
def test_killing_one_worker_leaves_serving(gateway: Gateway) -> None:
    import psutil

    port: Final = int(urlparse(str(gateway.client.base_url)).port or 0)
    listeners: Final = {
        connection.laddr.port: connection.pid
        for connection in psutil.net_connections(kind="tcp")
        if connection.status == "LISTEN" and connection.pid
    }
    owner: Final = listeners.get(port)
    assert owner is not None, f"no process listening on {port}"
    process: Final = psutil.Process(owner)
    children: Final = process.children(recursive=True)
    assert children, "leg proxy has no worker children to kill"
    with gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/audit-chat", api_base=f"{gateway.upstream_url}/v1")
        team_id: Final = scenario.team()
        callback: Final = gateway.request(
            "POST", f"/team/{team_id}/callback", {"callback_name": "langfuse_otel", "callback_vars": LANGFUSE_VARS}
        )
        assert callback.status_code == 200, callback.text
        key: Final = scenario.key(team_id=team_id)
        victim: Final = children[-1]
        victim.terminate()
        responses: Final = _send_burst(gateway, key, model, 10)
        assert all(response.status_code == 200 for response in responses), [r.status_code for r in responses]
