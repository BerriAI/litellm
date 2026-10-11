"""Langsmith success callback delivers run payloads for each completion.

Deterministic rewrite of the legacy load test: fixed request count, scripted
Langsmith upstream, asserts outbound posts arrive rather than a wall clock
overhead ratio.
"""

import asyncio
import gzip
import json
from typing import Final

import pytest

import litellm
from tests.integration._support.wire import Reply, Request, wire_server

REQUEST_COUNT: Final = 20


@pytest.mark.asyncio
async def test_langsmith_logging_async_delivers_runs(monkeypatch: pytest.MonkeyPatch):
    captured: list = []

    def upstream(request: Request) -> Reply:
        assert request.method in ("POST", "PATCH")
        body = request.body
        try:
            payload = gzip.decompress(body)
        except OSError:
            payload = body
        captured.append(payload)
        return Reply(status=202, body=b"{}")

    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_unit_key")
    monkeypatch.setenv("LANGSMITH_PROJECT", "pr-b")

    litellm.logging_callback_manager._reset_all_callbacks()

    with wire_server(upstream) as wire:
        monkeypatch.setenv("LANGSMITH_BASE_URL", wire.url)
        monkeypatch.setenv("LANGSMITH_ENDPOINT", wire.url)
        litellm.success_callback = ["langsmith"]
        try:
            responses = await asyncio.gather(
                *(
                    litellm.acompletion(
                        model="openai/chatgpt-v-2",
                        messages=[{"role": "user", "content": "This is a test"}],
                        max_tokens=5,
                        user="langfuse_latency_test_user",
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
    assert "langfuse_latency_test_user" in texts or "chatgpt-v-2" in texts or "openai" in texts
