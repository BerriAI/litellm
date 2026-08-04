"""
Tests for restamping the public model on Anthropic Messages streaming chunks.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.anthropic_endpoints.streaming_model_restamp import (
    restamp_anthropic_stream_chunk_model,
)
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing


def _message_start_frame(model: str) -> bytes:
    payload = {
        "type": "message_start",
        "message": {"id": "msg_1", "type": "message", "role": "assistant", "model": model, "content": []},
    }
    return f"event: message_start\ndata: {json.dumps(payload)}\n\n".encode()


def _proxy_logging_obj_streaming(frames: list[bytes]) -> MagicMock:
    async def _iterator_hook(**_kwargs):
        for frame in frames:
            yield frame

    proxy_logging_obj = MagicMock()
    proxy_logging_obj.async_post_call_streaming_iterator_hook = _iterator_hook
    proxy_logging_obj.async_post_call_streaming_hook = AsyncMock(side_effect=lambda **kwargs: kwargs["response"])
    return proxy_logging_obj


def _model_from_frame(frame: bytes | str) -> str:
    text = frame.decode("utf-8") if isinstance(frame, bytes) else frame
    data_line = next(line for line in text.split("\n") if line.startswith("data:"))
    return json.loads(data_line[len("data:") :])["message"]["model"]


def test_restamps_sse_bytes_frame():
    restamped = restamp_anthropic_stream_chunk_model(
        _message_start_frame("claude-haiku-4-5-20251001"), "claude-auto-1"
    )

    assert isinstance(restamped, bytes)
    assert _model_from_frame(restamped) == "claude-auto-1"
    assert b"event: message_start" in restamped


def test_restamps_event_dict():
    chunk = {"type": "message_start", "message": {"id": "msg_1", "model": "claude-sonnet-4-6"}}

    restamped = restamp_anthropic_stream_chunk_model(chunk, "claude-auto-2")

    assert restamped == {"type": "message_start", "message": {"id": "msg_1", "model": "claude-auto-2"}}
    assert chunk["message"]["model"] == "claude-sonnet-4-6"


@pytest.mark.parametrize(
    "chunk",
    [
        b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"text":"hi"}}\n\n',
        {"type": "content_block_delta", "delta": {"text": "hi"}},
        {"type": "message_start", "message": "not-a-dict"},
        b"event: message_start\ndata: not-json\n\n",
        b"data: [DONE]\n\n",
    ],
)
def test_leaves_chunks_without_a_model_untouched(chunk):
    assert restamp_anthropic_stream_chunk_model(chunk, "claude-auto-1") == chunk


@pytest.mark.asyncio
async def test_sse_generator_publishes_requested_model_on_message_start():
    """The message_start event reports the requested model, not the provider's."""
    delta_frame = b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"text":"hi"}}\n\n'
    proxy_logging_obj = _proxy_logging_obj_streaming([_message_start_frame("claude-haiku-4-5-20251001"), delta_frame])

    chunks = [
        chunk
        async for chunk in ProxyBaseLLMRequestProcessing.async_sse_data_generator(
            response=MagicMock(),
            user_api_key_dict=MagicMock(),
            request_data={"model": "claude-auto-1"},
            proxy_logging_obj=proxy_logging_obj,
            restamp_model="claude-auto-1",
        )
    ]

    assert _model_from_frame(chunks[0]) == "claude-auto-1"
    assert chunks[1] == delta_frame


@pytest.mark.asyncio
async def test_sse_generator_keeps_provider_model_when_restamping_is_off():
    proxy_logging_obj = _proxy_logging_obj_streaming([_message_start_frame("claude-haiku-4-5-20251001")])

    chunks = [
        chunk
        async for chunk in ProxyBaseLLMRequestProcessing.async_sse_data_generator(
            response=MagicMock(),
            user_api_key_dict=MagicMock(),
            request_data={"model": "claude-auto-1"},
            proxy_logging_obj=proxy_logging_obj,
        )
    ]

    assert _model_from_frame(chunks[0]) == "claude-haiku-4-5-20251001"
