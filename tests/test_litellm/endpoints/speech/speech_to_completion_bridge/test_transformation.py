from typing import Final
from unittest.mock import MagicMock

import pytest

import litellm
from litellm.constants import OPENAI_CHAT_COMPLETION_PARAMS
from litellm.endpoints.speech.speech_to_completion_bridge.transformation import (
    SpeechToCompletionBridgeTransformationHandler,
)

GEMINI_TTS_MODEL: Final = "gemini-3.1-flash-tts-preview"


def _bridge_request(response_format: str | None) -> dict:
    optional_params: Final = {"temperature": 0.4} if response_format is None else {"temperature": 0.4, "response_format": response_format}
    return SpeechToCompletionBridgeTransformationHandler().transform_request(
        model=GEMINI_TTS_MODEL,
        input="Hello from LiteLLM",
        voice="Kore",
        optional_params=optional_params,
        litellm_params={},
        headers={},
        litellm_logging_obj=MagicMock(),
        custom_llm_provider="gemini",
    )


@pytest.mark.parametrize("response_format", ["wav", "mp3", "pcm", None])
def test_gemini_tts_request_keeps_speech_response_format_out_of_chat_params(response_format: str | None) -> None:
    request: Final = _bridge_request(response_format)

    assert "response_format" not in request
    assert request["audio"] == {"voice": "Kore", "format": "pcm16"}
    assert request["temperature"] == 0.4
    assert request["modalities"] == ["audio"]

    gemini_params: Final = litellm.get_optional_params(
        model=GEMINI_TTS_MODEL,
        custom_llm_provider="gemini",
        **{param: value for param, value in request.items() if param in OPENAI_CHAT_COMPLETION_PARAMS},
    )
    assert gemini_params["speechConfig"] == {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}}}
    assert "responseMimeType" not in gemini_params


def test_non_gemini_request_forwards_speech_response_format_as_audio_format() -> None:
    request: Final = SpeechToCompletionBridgeTransformationHandler().transform_request(
        model="gpt-4o-audio-preview",
        input="Hello from LiteLLM",
        voice="alloy",
        optional_params={"response_format": "wav"},
        litellm_params={},
        headers={},
        litellm_logging_obj=MagicMock(),
        custom_llm_provider="openai",
    )

    assert "response_format" not in request
    assert request["audio"] == {"voice": "alloy", "format": "wav"}
