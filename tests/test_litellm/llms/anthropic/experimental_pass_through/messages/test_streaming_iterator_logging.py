"""
Regression test for https://github.com/BerriAI/litellm/issues/32019.

End-of-stream logging for /v1/messages must be enqueued on the logging
worker instead of a bare ``asyncio.create_task``: an unreferenced task can
be garbage-collected before it runs, dropping the success log entirely
(same class of bug fixed for pass-through routes in #31485).
"""

from unittest.mock import MagicMock

import pytest

from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.llms.anthropic.experimental_pass_through.messages.streaming_iterator import (
    BaseAnthropicMessagesStreamingIterator,
)


@pytest.mark.asyncio
async def test_handle_streaming_logging_enqueues_on_logging_worker(monkeypatch):
    enqueued = []

    def fake_enqueue(async_coroutine):
        enqueued.append(async_coroutine)
        async_coroutine.close()

    monkeypatch.setattr(GLOBAL_LOGGING_WORKER, "ensure_initialized_and_enqueue", fake_enqueue)

    iterator = BaseAnthropicMessagesStreamingIterator(
        litellm_logging_obj=MagicMock(),
        request_body={"model": "claude-3-5-sonnet-20240620", "stream": True},
    )

    await iterator._handle_streaming_logging(collected_chunks=[b"data: {}\n\n"])

    assert len(enqueued) == 1
