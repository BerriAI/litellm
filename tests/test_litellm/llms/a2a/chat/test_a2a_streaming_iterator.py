"""Tests for the A2A chat streaming iterator."""

import pytest

from litellm.llms.a2a.chat.streaming_iterator import A2AModelResponseIterator
from litellm.llms.a2a.common_utils import A2AError
from litellm.types.utils import Delta


@pytest.mark.asyncio
async def test_async_iterator_accepts_decoded_a2a_events():
    async def _events():
        yield {
            "jsonrpc": "2.0",
            "result": {
                "kind": "artifact-update",
                "artifact": {"parts": [{"kind": "text", "text": "Hello"}]},
            },
        }

    iterator = A2AModelResponseIterator(
        streaming_response=_events(),
        sync_stream=False,
    )

    chunk = await iterator.__aiter__().__anext__()

    assert chunk["text"] == "Hello"


@pytest.mark.asyncio
async def test_async_iterator_preserves_tool_calls():
    tool_calls = [
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": "lookup", "arguments": "{}"},
        }
    ]

    async def _events():
        yield {"jsonrpc": "2.0", "result": {"tool_calls": tool_calls}}

    iterator = A2AModelResponseIterator(streaming_response=_events(), sync_stream=False)

    chunk = await iterator.__aiter__().__anext__()

    assert chunk["tool_use"] == tool_calls[0]
    assert chunk["finish_reason"] == "tool_calls"


@pytest.mark.asyncio
async def test_async_iterator_serializes_delta_tool_calls_and_usage():
    delta = Delta(
        tool_calls=[
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{}"},
            }
        ]
    )

    async def _events():
        yield {
            "jsonrpc": "2.0",
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
            "result": {
                "tool_calls": [delta.tool_calls[0]],
                "finish_reason": "length",
            },
        }

    iterator = A2AModelResponseIterator(streaming_response=_events(), sync_stream=False)
    chunk = await iterator.__aiter__().__anext__()

    assert chunk["tool_use"]["id"] == "call-1"
    assert chunk["finish_reason"] == "length"
    assert chunk["usage"].total_tokens == 5


@pytest.mark.asyncio
async def test_async_iterator_closes_nested_stream():
    closed = False

    async def _events():
        nonlocal closed
        try:
            yield {"jsonrpc": "2.0", "result": {"kind": "artifact-update"}}
            raise AssertionError("stream should be closed before a second event")
        finally:
            closed = True

    iterator = A2AModelResponseIterator(streaming_response=_events(), sync_stream=False)
    await iterator.__aiter__().__anext__()
    await iterator.aclose()

    assert closed is True


@pytest.mark.asyncio
async def test_async_iterator_propagates_jsonrpc_errors():
    async def _events():
        yield {"jsonrpc": "2.0", "error": {"code": -32000, "message": "agent failed"}}

    iterator = A2AModelResponseIterator(streaming_response=_events(), sync_stream=False)

    with pytest.raises(A2AError, match="agent failed"):
        await iterator.__aiter__().__anext__()
