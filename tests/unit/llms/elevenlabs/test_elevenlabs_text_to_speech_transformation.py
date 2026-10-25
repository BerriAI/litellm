import json
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.llms.elevenlabs.text_to_speech.transformation import (
    ElevenLabsTextToSpeechConfig,
)


def test_should_encode_elevenlabs_voice_id_path_segment():
    config = ElevenLabsTextToSpeechConfig()

    url = config.get_complete_url(
        model="elevenlabs/tts",
        api_base="https://api.elevenlabs.io",
        litellm_params={
            config.ELEVENLABS_VOICE_ID_KEY: "voice/../../models?x=1#frag",
        },
    )

    assert url == "https://api.elevenlabs.io/v1/text-to-speech/voice%2F..%2F..%2Fmodels%3Fx%3D1%23frag"


def test_should_reject_dot_segment_elevenlabs_voice_id():
    config = ElevenLabsTextToSpeechConfig()

    with pytest.raises(ValueError, match="voice_id cannot be a dot path segment"):
        config.get_complete_url(
            model="elevenlabs/tts",
            api_base="https://api.elevenlabs.io",
            litellm_params={config.ELEVENLABS_VOICE_ID_KEY: ".."},
        )


def test_speech_keeps_an_internal_prefixed_kwarg_out_of_the_elevenlabs_request(respx_mock: respx.MockRouter) -> None:
    api_base: Final = "http://localhost:12346"
    mock_route: Final = respx_mock.post(url__regex=rf"{api_base}/v1/text-to-speech/.*").mock(
        return_value=httpx.Response(status_code=200, content=b"audio", headers={"content-type": "audio/mpeg"})
    )

    litellm.speech(
        model="elevenlabs/eleven_multilingual_v2",
        input="hi",
        voice="21m00Tcm4TlvDq8ikWAM",
        api_base=api_base,
        api_key="fake_elevenlabs_api_key",
        _litellm_undeclared_sentinel="internal",
    )

    assert mock_route.called
    sent: Final = json.loads(respx_mock.calls[0].request.content)
    assert "_litellm_undeclared_sentinel" not in sent, sent
    assert sent["text"] == "hi"



def test_map_openai_params_wav_format() -> None:
    config = ElevenLabsTextToSpeechConfig()
    kwargs: dict = {}
    voice, mapped_params = config.map_openai_params(
        model="elevenlabs/tts",
        optional_params={"response_format": "wav"},
        voice="21m00Tcm4TlvDq8ikWAM",
        kwargs=kwargs,
    )
    query_params = kwargs.get(config.ELEVENLABS_QUERY_PARAMS_KEY, {})
    assert query_params.get("output_format") == "wav_44100_16"

    # Also test case-insensitivity
    kwargs_upper: dict = {}
    voice, mapped_params = config.map_openai_params(
        model="elevenlabs/tts",
        optional_params={"response_format": "WAV"},
        voice="21m00Tcm4TlvDq8ikWAM",
        kwargs=kwargs_upper,
    )
    query_params_upper = kwargs_upper.get(config.ELEVENLABS_QUERY_PARAMS_KEY, {})
    assert query_params_upper.get("output_format") == "wav_44100_16"


def test_speech_sets_elevenlabs_output_format_to_wav(respx_mock: respx.MockRouter) -> None:
    api_base: Final = "http://localhost:12346"
    mock_route: Final = respx_mock.post(url__regex=rf"{api_base}/v1/text-to-speech/.*").mock(
        return_value=httpx.Response(status_code=200, content=b"audio_bytes", headers={"content-type": "audio/wav"})
    )

    litellm.speech(
        model="elevenlabs/eleven_multilingual_v2",
        input="hello",
        voice="21m00Tcm4TlvDq8ikWAM",
        response_format="wav",
        api_base=api_base,
        api_key="fake_elevenlabs_api_key",
    )

    assert mock_route.called
    request = respx_mock.calls[0].request
    assert request.url.params.get("output_format") == "wav_44100_16"


def test_get_complete_url_with_wav_output_format() -> None:
    config = ElevenLabsTextToSpeechConfig()

    url = config.get_complete_url(
        model="elevenlabs/tts",
        api_base="https://api.elevenlabs.io",
        litellm_params={
            config.ELEVENLABS_VOICE_ID_KEY: "21m00Tcm4TlvDq8ikWAM",
            config.ELEVENLABS_QUERY_PARAMS_KEY: {"output_format": "wav_44100_16"},
        },
    )

    assert url == "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM?output_format=wav_44100_16"
