import base64
from typing import Final
from unittest.mock import MagicMock

import pytest

import litellm
from litellm.constants import OPENAI_CHAT_COMPLETION_PARAMS
from litellm.endpoints.speech.speech_to_completion_bridge.transformation import (
    SpeechToCompletionBridgeTransformationHandler,
)
from litellm.types.utils import ChatCompletionAudioResponse, Choices, Message, ModelResponse

pytestmark = pytest.mark.usefixtures("local_model_cost_map")

GEMINI_TTS_MODEL: Final = "gemini-3.1-flash-tts-preview"
PCM_BYTES: Final = b"\x01\x02\x03\x04" * 6


def _model_response(model: str, pcm: bytes) -> ModelResponse:
    audio: Final = ChatCompletionAudioResponse(data=base64.b64encode(pcm).decode(), expires_at=0, transcript="hello")
    return ModelResponse(model=model, choices=[Choices(message=Message(content=None, audio=audio))])


def _bridge_request(response_format: str | None) -> dict:
    optional_params: Final = (
        {"temperature": 0.4} if response_format is None else {"temperature": 0.4, "response_format": response_format}
    )
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


@pytest.mark.parametrize("response_format", ["wav", "pcm", None])
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


def test_gemini_request_keeps_voice_beside_nested_speech_config() -> None:
    request: Final = SpeechToCompletionBridgeTransformationHandler().transform_request(
        model=GEMINI_TTS_MODEL,
        input="Bonjour",
        voice={"speech_config": {"language_code": "fr-FR"}, "voice": "Kore"},
        optional_params={},
        litellm_params={},
        headers={},
        litellm_logging_obj=MagicMock(),
        custom_llm_provider="gemini",
    )

    assert request["audio"]["speech_config"] == {
        "languageCode": "fr-FR",
        "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}},
    }


def test_gemini_request_keeps_voice_object_for_a_model_alias_map_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "model_alias_map", {"voice-alias": GEMINI_TTS_MODEL})

    request: Final = SpeechToCompletionBridgeTransformationHandler().transform_request(
        model="voice-alias",
        input="Hello there",
        voice={"voice": "Puck"},
        optional_params={},
        litellm_params={},
        headers={},
        litellm_logging_obj=MagicMock(),
        custom_llm_provider="gemini",
    )

    assert request["audio"] == {
        "speech_config": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Puck"}}},
        "format": "pcm16",
    }


@pytest.mark.parametrize("wrapper", (None, "speech_config", "speechConfig"))
def test_gemini_request_keeps_voice_and_language_without_audio_format(wrapper: str | None) -> None:
    voice: Final = {"language_code": "fr-FR", "voice": "Kore", "format": "pcm16"}
    request: Final = SpeechToCompletionBridgeTransformationHandler().transform_request(
        model=GEMINI_TTS_MODEL,
        input="Bonjour",
        voice=voice if wrapper is None else {wrapper: voice},
        optional_params={},
        litellm_params={},
        headers={},
        litellm_logging_obj=MagicMock(),
        custom_llm_provider="gemini",
    )

    assert request["audio"]["speech_config"] == {
        "languageCode": "fr-FR",
        "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}},
    }


@pytest.mark.parametrize(
    ("drop_params", "voice_config"),
    [
        (True, {"prebuiltVoiceConfig": {"voiceName": "Kore"}}),
        (False, {"prebuiltVoiceConfig": {"voiceName": "Kore", "style": "calm"}, "tone": "warm"}),
    ],
)
def test_gemini_request_drops_unknown_nested_voice_fields_only_with_drop_params(
    drop_params: bool, voice_config: dict[str, object]
) -> None:
    request: Final = SpeechToCompletionBridgeTransformationHandler().transform_request(
        model=GEMINI_TTS_MODEL,
        input="Bonjour",
        voice={"voice_config": {"prebuilt_voice_config": {"voice_name": "Kore", "style": "calm"}, "tone": "warm"}},
        optional_params={},
        litellm_params={"drop_params": drop_params},
        headers={},
        litellm_logging_obj=MagicMock(),
        custom_llm_provider="gemini",
    )

    assert request["audio"]["speech_config"] == {"voiceConfig": voice_config}


def test_gemini_request_reads_snake_case_wrapper_beside_null_camel_case_wrapper() -> None:
    request: Final = SpeechToCompletionBridgeTransformationHandler().transform_request(
        model=GEMINI_TTS_MODEL,
        input="Bonjour",
        voice={"speechConfig": None, "speech_config": {"voice_name": "Kore"}},
        optional_params={},
        litellm_params={},
        headers={},
        litellm_logging_obj=MagicMock(),
        custom_llm_provider="gemini",
    )

    assert request["audio"]["speech_config"] == {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}}}


@pytest.mark.parametrize("response_format", ["mp3", "flac", "opus", "aac"])
def test_gemini_tts_request_rejects_formats_gemini_cannot_produce(response_format: str) -> None:
    with pytest.raises(litellm.BadRequestError) as excinfo:
        _bridge_request(response_format)

    assert excinfo.value.status_code == 400
    assert response_format in str(excinfo.value)
    assert "pcm" in str(excinfo.value)
    assert "wav" in str(excinfo.value)


def test_gemini_tts_pcm_response_returns_raw_pcm_bytes() -> None:
    response: Final = SpeechToCompletionBridgeTransformationHandler().transform_response(
        model_response=_model_response(GEMINI_TTS_MODEL, PCM_BYTES),
        model=GEMINI_TTS_MODEL,
        response_format="pcm",
    )

    assert response.response.content == PCM_BYTES
    assert response.response.headers["content-type"] == "audio/pcm"


@pytest.mark.parametrize("response_format", ["wav", None])
def test_gemini_tts_wav_and_default_responses_wrap_pcm_in_wav(response_format: str | None) -> None:
    response: Final = SpeechToCompletionBridgeTransformationHandler().transform_response(
        model_response=_model_response(GEMINI_TTS_MODEL, PCM_BYTES),
        model=GEMINI_TTS_MODEL,
        response_format=response_format,
    )

    body: Final = response.response.content
    assert body[:4] == b"RIFF"
    assert body[8:12] == b"WAVE"
    assert body[44:] == PCM_BYTES
    assert response.response.headers["content-type"] == "audio/wav"


def test_non_gemini_response_keeps_original_bytes_and_mpeg_content_type() -> None:
    response: Final = SpeechToCompletionBridgeTransformationHandler().transform_response(
        model_response=_model_response("gpt-4o-audio-preview", PCM_BYTES),
        model="gpt-4o-audio-preview",
        response_format="mp3",
    )

    assert response.response.content == PCM_BYTES
    assert response.response.headers["content-type"] == "audio/mpeg"


@pytest.mark.parametrize(
    ("requested_model", "served_model"),
    [(GEMINI_TTS_MODEL, "served-speech-alias"), ("requested-speech-alias", GEMINI_TTS_MODEL)],
)
def test_gemini_response_wraps_wav_when_either_model_name_is_gemini_tts(requested_model: str, served_model: str) -> None:
    response: Final = SpeechToCompletionBridgeTransformationHandler().transform_response(
        model_response=_model_response(served_model, PCM_BYTES),
        model=requested_model,
        response_format=None,
    )

    assert response.response.content[:4] == b"RIFF"
    assert response.response.headers["content-type"] == "audio/wav"
