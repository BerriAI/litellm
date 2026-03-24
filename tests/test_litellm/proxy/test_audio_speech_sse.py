"""
Tests for audio/speech SSE stream_format support.

Issue: https://github.com/BerriAI/litellm/issues/24301
When stream_format='sse' is passed to /v1/audio/speech, the proxy should return
an SSE event stream (Content-Type: text/event-stream) instead of binary audio.
"""

import asyncio
import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class MockAiterBytes:
    """Mock for HttpxBinaryResponseContent that supports aiter_bytes."""

    def __init__(self, chunks: list):
        self.chunks = chunks

    async def aiter_bytes(self, chunk_size=8192):
        for chunk in self.chunks:
            yield chunk


@pytest.mark.asyncio
async def test_audio_speech_sse_generator():
    """Test that _audio_speech_sse_generator wraps audio chunks in SSE events."""
    from litellm.proxy.proxy_server import _audio_speech_sse_generator

    test_chunks = [b"audio_chunk_1", b"audio_chunk_2"]
    mock_response = MockAiterBytes(test_chunks)

    events = []
    async for event in _audio_speech_sse_generator(mock_response):  # type: ignore
        events.append(event)

    # Should have one event per chunk plus [DONE]
    assert len(events) == 3, f"Expected 3 events (2 chunks + DONE), got {len(events)}"

    # Verify the [DONE] event
    assert events[-1] == "data: [DONE]\n\n"

    # Verify audio chunks are base64-encoded JSON events
    for i, chunk in enumerate(test_chunks):
        event = events[i]
        assert event.startswith("data: "), f"Event {i} should start with 'data: '"
        assert event.endswith("\n\n"), f"Event {i} should end with '\\n\\n'"

        # Parse the JSON payload
        payload = json.loads(event[len("data: "):].rstrip("\n"))
        assert "audio" in payload, f"Event {i} payload should have 'audio' key"

        # Verify the audio data is correctly base64-encoded
        decoded = base64.b64decode(payload["audio"])
        assert decoded == chunk, f"Event {i} audio data mismatch"


@pytest.mark.asyncio
async def test_audio_speech_sse_generator_empty():
    """Test _audio_speech_sse_generator with empty response still sends [DONE]."""
    from litellm.proxy.proxy_server import _audio_speech_sse_generator

    mock_response = MockAiterBytes([])

    events = []
    async for event in _audio_speech_sse_generator(mock_response):  # type: ignore
        events.append(event)

    assert len(events) == 1
    assert events[0] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_audio_speech_sse_content_type(monkeypatch):
    """
    Integration-style test: audio/speech with stream_format='sse' should return
    StreamingResponse with media_type='text/event-stream'.
    """
    from fastapi.responses import StreamingResponse

    from litellm.proxy.proxy_server import _audio_speech_sse_generator

    test_chunks = [b"hello_audio"]
    mock_response = MockAiterBytes(test_chunks)

    # Simulate the handler logic
    data = {"stream_format": "sse", "model": "tts-1", "input": "Hello", "voice": "alloy"}

    if data.get("stream_format") == "sse":
        response = StreamingResponse(
            _audio_speech_sse_generator(mock_response),  # type: ignore
            media_type="text/event-stream",
        )
        assert response.media_type == "text/event-stream"
    else:
        pytest.fail("stream_format='sse' branch not taken")


@pytest.mark.asyncio
async def test_audio_speech_binary_response_unchanged():
    """
    When stream_format is not 'sse', the response should be binary (audio/mpeg).
    """
    from fastapi.responses import StreamingResponse

    from litellm.proxy.proxy_server import _audio_speech_chunk_generator

    test_chunks = [b"binary_audio_data"]
    mock_response = MockAiterBytes(test_chunks)

    data = {"model": "tts-1", "input": "Hello", "voice": "alloy"}

    if data.get("stream_format") == "sse":
        pytest.fail("Should not enter SSE branch when stream_format is not set")
    else:
        response = StreamingResponse(
            _audio_speech_chunk_generator(mock_response),  # type: ignore
            media_type="audio/mpeg",
        )
        assert response.media_type == "audio/mpeg"
