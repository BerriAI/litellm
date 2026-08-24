"""Tests for the A2A chat streaming iterator."""

import pytest

from litellm.llms.a2a.chat.streaming_iterator import A2AModelResponseIterator


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
