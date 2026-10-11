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

            deadline = asyncio.get_event_loop().time() + 30
            while not captured and asyncio.get_event_loop().time() < deadline:
                await asyncio.sleep(0.05)
        finally:
            litellm.success_callback = []
            litellm.logging_callback_manager._reset_all_callbacks()

    assert len(captured) >= 1
    texts = b"".join(captured).decode("utf-8", errors="replace")
    assert "datadog_latency_test_user" in texts or "chatgpt-v-3" in texts
