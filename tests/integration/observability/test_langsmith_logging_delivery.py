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
from tests._wait_helpers import await_until
from tests.integration._support.wire import Reply, Request, wire_server

REQUEST_COUNT: Final = 20


def _posted_runs(captured: list) -> list:
    return [run for payload in captured for run in json.loads(payload).get("post", [])]


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

            await await_until(
                lambda: len(_posted_runs(captured)) >= REQUEST_COUNT,
                message=f"{REQUEST_COUNT} LangSmith runs posted",
                timeout_s=30,
            )
        finally:
            litellm.success_callback = []
            litellm.logging_callback_manager._reset_all_callbacks()

    runs = _posted_runs(captured)
    assert len(runs) == REQUEST_COUNT
    assert {(run["run_type"], run["session_name"]) for run in runs} == {("llm", "pr-b")}
    assert len({run["inputs"]["id"] for run in runs}) == REQUEST_COUNT
    assert {run["inputs"]["model"] for run in runs} == {"chatgpt-v-2"}
