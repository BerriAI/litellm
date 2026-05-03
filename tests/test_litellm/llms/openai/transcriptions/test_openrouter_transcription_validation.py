"""OpenRouter does not expose /audio/transcriptions; fail fast with a clear error."""

import io

import pytest

import litellm
from litellm.exceptions import UnsupportedParamsError


def test_should_raise_unsupported_params_error_for_openrouter_transcription_sync():
    with pytest.raises(UnsupportedParamsError) as exc_info:
        litellm.transcription(
            model="openrouter/openai/whisper-1",
            file=io.BytesIO(b"fake"),
        )
    msg = str(exc_info.value)
    assert "OpenRouter" in msg
    assert "audio transcription" in msg.lower()
    assert "whisper-1" in msg


@pytest.mark.asyncio
async def test_should_raise_unsupported_params_error_for_openrouter_transcription_async():
    with pytest.raises(UnsupportedParamsError) as exc_info:
        await litellm.atranscription(
            model="openrouter/openai/whisper-1",
            file=io.BytesIO(b"fake"),
        )
    msg = str(exc_info.value)
    assert "OpenRouter" in msg
    assert "audio transcription" in msg.lower()
    assert "whisper-1" in msg
