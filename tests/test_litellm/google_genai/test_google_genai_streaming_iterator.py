import json
from unittest.mock import MagicMock

import pytest

from litellm.google_genai.streaming_iterator import (
    AsyncGoogleGenAIGenerateContentStreamingIterator,
    GoogleGenAIGenerateContentStreamingIterator,
)
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


def _sse_event(payload: dict) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def _chunk_bytes(data: bytes, size: int) -> list[bytes]:
    return [data[i : i + size] for i in range(0, len(data), size)]


def _build_iterator(stream_bytes, *, is_async: bool):
    mock_response = MagicMock()
    if is_async:

        async def _aiter_bytes():
            for chunk in stream_bytes:
                yield chunk

        mock_response.aiter_bytes = _aiter_bytes
        cls = AsyncGoogleGenAIGenerateContentStreamingIterator
    else:
        mock_response.iter_bytes.return_value = iter(stream_bytes)
        cls = GoogleGenAIGenerateContentStreamingIterator

    return cls(
        response=mock_response,
        model="gemini-3.1-pro-preview",
        logging_obj=MagicMock(spec=LiteLLMLoggingObj),
        generate_content_provider_config=MagicMock(),
        litellm_metadata={},
        custom_llm_provider="gemini",
    )


def _parse_event(frame: bytes) -> dict:
    assert frame.startswith(b"data: ")
    assert frame.endswith(b"\n\n")
    return json.loads(frame[len(b"data: ") : -2])


@pytest.mark.asyncio
async def test_async_iterator_preserves_unicode_line_separators():
    """Regression for LIT-3775: U+2028/U+2029/U+0085 inside ``data:`` JSON.

    Gemini emits these raw in thinking/text content. httpx ``aiter_lines``
    splits on them (``str.splitlines``) and the event got rejoined with ``\\n``,
    producing invalid JSON the SDK could not parse.
    """
    text = "first second thirdfourth"
    payload = {"candidates": [{"content": {"parts": [{"text": text}]}}]}
    wire = _sse_event(payload)
    # Split the event into small byte chunks so a separator never aligns with a
    # chunk boundary; this is what the upstream HTTP stream looks like.
    iterator = _build_iterator(_chunk_bytes(wire, 7), is_async=True)

    frames = [frame async for frame in iterator]

    assert len(frames) == 1
    parsed = _parse_event(frames[0])
    assert parsed["candidates"][0]["content"]["parts"][0]["text"] == text


@pytest.mark.asyncio
async def test_async_iterator_reassembles_inline_data_split_across_chunks():
    """Large inlineData payloads must stay in one event even when chunked."""
    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"inlineData": {"mimeType": "image/jpeg", "data": "A" * 20000}}
                    ]
                }
            }
        ]
    }
    wire = _sse_event(payload)
    iterator = _build_iterator(_chunk_bytes(wire, 1024), is_async=True)

    frames = [frame async for frame in iterator]

    assert len(frames) == 1
    parsed = _parse_event(frames[0])
    inline = parsed["candidates"][0]["content"]["parts"][0]["inlineData"]
    assert inline["mimeType"] == "image/jpeg"
    assert inline["data"] == "A" * 20000


def test_sync_iterator_reassembles_inline_data_split_across_chunks():
    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"inlineData": {"mimeType": "image/jpeg", "data": "A" * 20000}}
                    ]
                }
            }
        ]
    }
    wire = _sse_event(payload)
    iterator = _build_iterator(_chunk_bytes(wire, 1024), is_async=False)

    frames = list(iterator)

    assert len(frames) == 1
    assert (
        _parse_event(frames[0])["candidates"][0]["content"]["parts"][0]["inlineData"][
            "data"
        ]
        == "A" * 20000
    )


@pytest.mark.asyncio
async def test_async_iterator_splits_multiple_events():
    first = _sse_event({"candidates": [{"content": {"parts": [{"text": "one"}]}}]})
    second = _sse_event({"candidates": [{"content": {"parts": [{"text": "two"}]}}]})
    # Feed both events as a single byte blob to prove the buffer splits on the
    # frame delimiter rather than relying on chunk boundaries.
    iterator = _build_iterator([first + second], is_async=True)

    frames = [frame async for frame in iterator]

    assert [
        _parse_event(f)["candidates"][0]["content"]["parts"][0]["text"] for f in frames
    ] == [
        "one",
        "two",
    ]


@pytest.mark.asyncio
async def test_async_iterator_preserves_multi_field_and_comment_events():
    multi_field = b'event: message\ndata: {"text":"hi"}\n\n'
    comment = b": keepalive\n\n"
    iterator = _build_iterator([multi_field, comment], is_async=True)

    frames = [frame async for frame in iterator]

    assert frames == [multi_field, comment]


@pytest.mark.asyncio
async def test_async_iterator_flushes_trailing_event_without_delimiter():
    """A final event missing its trailing delimiter must still be emitted."""
    tail = b'data: {"text":"last"}'
    iterator = _build_iterator(_chunk_bytes(tail, 5), is_async=True)

    frames = [frame async for frame in iterator]

    assert frames == [tail]


def test_sync_iterator_flushes_trailing_event_without_delimiter():
    tail = b'data: {"text":"last"}'
    iterator = _build_iterator(_chunk_bytes(tail, 5), is_async=False)

    assert list(iterator) == [tail]
