"""Tests for the A2A chat streaming iterator."""

import pytest

from litellm.llms.a2a.chat.streaming_iterator import A2AModelResponseIterator
from litellm.llms.a2a.common_utils import A2AError


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

    assert chunk["tool_use"] == tool_calls
    assert chunk["finish_reason"] == "tool_calls"


@pytest.mark.asyncio
async def test_async_iterator_propagates_jsonrpc_errors():
    async def _events():
        yield {"jsonrpc": "2.0", "error": {"code": -32000, "message": "agent failed"}}

    iterator = A2AModelResponseIterator(streaming_response=_events(), sync_stream=False)

    with pytest.raises(A2AError, match="agent failed"):
        await iterator.__aiter__().__anext__()
