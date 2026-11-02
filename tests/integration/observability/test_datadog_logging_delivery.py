"""Datadog success callback delivers one log payload per completion.

Deterministic rewrite of the legacy load test: fixed request count, scripted
Datadog intake upstream, asserts the outbound posts arrive rather than a wall
clock overhead ratio.
"""

import asyncio
import json
import zlib
from typing import Final

import pytest

import litellm
from tests._wait_helpers import await_until
from tests.integration._support.wire import Reply, Request, wire_server

REQUEST_COUNT: Final = 20


def _datadog_upstream(request: Request, captured: list) -> Reply:
    assert request.method == "POST"
    body = request.body
    try:
        payload = zlib.decompress(body, zlib.MAX_WBITS | 16)
    except zlib.error:
        payload = body
    captured.append(payload)
    return Reply(status=202, body=b"{}")


def _delivered_logs(captured: list) -> list:
    return [entry for payload in captured for entry in json.loads(payload)]


@pytest.mark.asyncio
async def test_datadog_logging_async_delivers_logs(monkeypatch: pytest.MonkeyPatch):
    captured: list = []

    def upstream(request: Request) -> Reply:
        return _datadog_upstream(request, captured)

    monkeypatch.setenv("DD_API_KEY", "unit-dd-key")
    monkeypatch.setenv("DD_SITE", "us5.datadoghq.com")

    litellm.logging_callback_manager._reset_all_callbacks()

    with wire_server(upstream) as wire:
        monkeypatch.setenv("DD_BASE_URL", wire.url)
        litellm.success_callback = ["datadog"]
        try:
            responses = await asyncio.gather(
                *(
                    litellm.acompletion(
                        model="openai/chatgpt-v-3",
                        messages=[{"role": "user", "content": "This is a test"}],
                        max_tokens=5,
                        user="datadog_latency_test_user",
                        mock_response="hello from my load test",
                    )
                    for _ in range(REQUEST_COUNT)
                )
            )
            assert len(responses) == REQUEST_COUNT

            from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

            await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=30)

            await await_until(
                lambda: len(_delivered_logs(captured)) >= REQUEST_COUNT,
                message=f"{REQUEST_COUNT} Datadog log entries delivered",
                timeout_s=30,
            )
        finally:
            litellm.success_callback = []
            litellm.logging_callback_manager._reset_all_callbacks()

    logs = _delivered_logs(captured)
    assert len(logs) == REQUEST_COUNT
    assert {log["ddsource"] for log in logs} == {"litellm"}
    payloads = [json.loads(log["message"]) for log in logs]
    assert len({payload["id"] for payload in payloads}) == REQUEST_COUNT
    assert {(payload["status"], payload["model"]) for payload in payloads} == {("success", "chatgpt-v-3")}
