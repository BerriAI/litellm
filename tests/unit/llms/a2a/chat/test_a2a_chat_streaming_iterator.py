"""Tests for litellm/llms/a2a/chat/streaming_iterator.py."""

import pytest

from litellm.llms.a2a.chat.streaming_iterator import A2AModelResponseIterator
from litellm.llms.a2a.common_utils import A2AError


def _iterator(lines: list[str]) -> A2AModelResponseIterator:
    return A2AModelResponseIterator(streaming_response=iter(lines), sync_stream=True)


def test_a_jsonrpc_error_in_the_stream_fails_the_call():
    """An agent that answers message/stream with a JSON-RPC error (Microsoft Foundry replies -32004
    "operation not supported") must fail the call with that message instead of ending an empty stream."""
    iterator = _iterator(
        ['{"jsonrpc":"2.0","id":"1","error":{"code":-32004,"message":"This operation is not supported"}}']
    )

    with pytest.raises(A2AError, match="This operation is not supported"):
        next(iterator)


def test_a_completed_task_chunk_yields_its_text_and_stops():
    iterator = _iterator(
        [
            '{"jsonrpc":"2.0","id":"1","result":{"kind":"task","status":{"state":"completed"},'
            '"artifacts":[{"parts":[{"kind":"text","text":"7"}]}]}}'
        ]
    )

    chunk = next(iterator)

    assert chunk["text"] == "7"
    assert chunk["is_finished"] is True
    assert chunk["finish_reason"] == "stop"
