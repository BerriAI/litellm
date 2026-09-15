"""
Regression tests for issue #34692.

ollama_chat streams tool_calls in a mid-stream chunk while its final
(``done: true``) chunk carries only ``done_reason: "stop"``. The provider
iterator must remember the earlier tool_calls and stamp
``finish_reason="tool_calls"`` on the final chunk, so the Anthropic
``/v1/messages`` bridge emits ``stop_reason: "tool_use"``. Before the fix the
bridge emitted ``stop_reason: "end_turn"`` and Anthropic tool-runners
(Claude Code, ``messages.stream``) silently dropped the tool call.
"""

import pytest

from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
    AnthropicStreamWrapper,
)
from litellm.llms.ollama.chat.transformation import (
    OllamaChatCompletionResponseIterator,
)
from litellm.types.utils import ModelResponseStream

_OLLAMA_TOOL_CHUNK = {
    "model": "qwen3:8b",
    "message": {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "San Francisco"}}}],
    },
    "done": False,
}
_OLLAMA_DONE_CHUNK = {
    "model": "qwen3:8b",
    "message": {"role": "assistant", "content": ""},
    "done": True,
    "done_reason": "stop",
    "prompt_eval_count": 100,
    "eval_count": 20,
}


def _ollama_streamed_chunks() -> list[ModelResponseStream]:
    iterator = OllamaChatCompletionResponseIterator(streaming_response=iter([]), sync_stream=True)
    return [iterator.chunk_parser(_OLLAMA_TOOL_CHUNK), iterator.chunk_parser(_OLLAMA_DONE_CHUNK)]


class _AsyncStream:
    def __init__(self, items: list[ModelResponseStream]):
        self._it = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration


def _assert_tool_use_stop_reason(events: list[dict]) -> None:
    block_types = [e["content_block"]["type"] for e in events if e.get("type") == "content_block_start"]
    assert "tool_use" in block_types, f"no tool_use content block opened: {events}"
    message_deltas = [e for e in events if e.get("type") == "message_delta"]
    assert message_deltas, f"no message_delta emitted: {events}"
    assert message_deltas[-1]["delta"]["stop_reason"] == "tool_use", (
        f"expected stop_reason 'tool_use', got: {message_deltas[-1]}"
    )


def test_ollama_mid_stream_tool_call_yields_tool_use_stop_reason_sync():
    wrapper = AnthropicStreamWrapper(completion_stream=iter(_ollama_streamed_chunks()), model="qwen3:8b")
    _assert_tool_use_stop_reason(list(wrapper))


@pytest.mark.asyncio
async def test_ollama_mid_stream_tool_call_yields_tool_use_stop_reason_async():
    wrapper = AnthropicStreamWrapper(completion_stream=_AsyncStream(_ollama_streamed_chunks()), model="qwen3:8b")
    _assert_tool_use_stop_reason([event async for event in wrapper])
