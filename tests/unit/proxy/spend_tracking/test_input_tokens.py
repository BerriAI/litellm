"""Tests for input-token counting shared across the reservation path's models."""

from __future__ import annotations

import base64
from types import MappingProxyType
from typing import Final

import pytest

from litellm.proxy.spend_tracking.input_tokens import (
    TOKENIZE_OFF_EVENT_LOOP_MIN_CHARS,
    _count_text_tokens,
    count_input_tokens,
    count_input_tokens_for_model,
)

CL100K_MODEL: Final = "gpt-4"


@pytest.mark.asyncio
async def test_large_input_is_still_counted() -> None:
    request_body: Final = {
        "model": CL100K_MODEL,
        "messages": [{"role": "user", "content": "x" * (TOKENIZE_OFF_EVENT_LOOP_MIN_CHARS + 1)}],
    }

    counts: Final = await count_input_tokens(request_body=request_body, raw_body=None, models=(CL100K_MODEL,))

    assert counts[CL100K_MODEL] == count_input_tokens_for_model(request_body=request_body, model=CL100K_MODEL)
    assert isinstance(counts, MappingProxyType)


def test_input_audio_requests_reserve_at_least_the_serialised_fallback() -> None:
    """
    Budget reservation counts an ``input_audio`` block as a size-derived
    estimate at a deliberately low bitrate, priced at the text rate. Before
    #38459 the same request raised inside ``token_counter`` and reserved
    from the serialised-messages fallback, which tokenises the base64
    payload itself. Floor audio-bearing requests at that fallback so a
    caller cannot be admitted against a budget more cheaply than before
    the blocks became countable (compressed audio carries far more duration
    per byte than the estimate assumes).
    """
    model: Final = "gpt-4o-audio-preview"
    audio_b64: Final = base64.b64encode(bytes(range(256)) * 400).decode()
    audio_messages: Final = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Transcribe this recording."},
                {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "mp3"}},
            ],
        }
    ]
    text_messages: Final = [{"role": "user", "content": "Transcribe this recording."}]

    audio_count: Final = count_input_tokens_for_model(request_body={"messages": audio_messages}, model=model)
    fallback: Final = _count_text_tokens(model=model, text=audio_messages)
    text_count: Final = count_input_tokens_for_model(request_body={"messages": text_messages}, model=model)

    assert audio_count is not None, "an audio-bearing request must still be countable"
    assert fallback > 0, "the serialised fallback must see the base64 payload"
    assert audio_count >= fallback, (
        f"audio request reserved {audio_count} tokens, below the pre-#38459 fallback of {fallback}"
    )
    assert text_count is not None
    assert text_count < fallback, "a text-only request must not be floored"
