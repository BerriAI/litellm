import pytest

import litellm
from litellm import transcription


def test_fireworks_audio_transcription_unsupported():
    """
    Fireworks AI deprecated audio inference on 2026-06-10, so transcription must be
    rejected locally with a clear error before any outbound request is attempted.

    Regression for https://github.com/BerriAI/litellm/issues/30916
    """
    with pytest.raises(litellm.BadRequestError) as exc_info:
        transcription(
            model="fireworks_ai/whisper-v3",
            file=("audio.wav", b"not a real audio file", "audio/wav"),
        )
    assert "audio transcription" in str(exc_info.value).lower()
