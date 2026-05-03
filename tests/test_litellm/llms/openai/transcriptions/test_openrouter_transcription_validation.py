"""OpenRouter does not expose /audio/transcriptions; fail fast with a clear error."""

import io

import pytest

import litellm
from litellm.exceptions import UnsupportedParamsError
from litellm.llms.openrouter.transcription_validation import (
    ensure_audio_transcription_supported,
)


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


def test_should_no_op_ensure_audio_transcription_supported_for_non_openrouter_provider():
    """Early return path must not raise (covers openrouter transcription_validation guard)."""
    ensure_audio_transcription_supported(
        model="whisper-1",
        custom_llm_provider="openai",
    )
    ensure_audio_transcription_supported(
        model="deployment-whisper",
        custom_llm_provider="azure",
    )


def test_should_raise_ensure_audio_transcription_supported_when_openrouter():
    with pytest.raises(UnsupportedParamsError) as exc_info:
        ensure_audio_transcription_supported(
            model="openrouter/openai/whisper-1",
            custom_llm_provider="openrouter",
        )
    msg = str(exc_info.value)
    assert "OpenRouter" in msg
    assert "whisper-1" in msg
