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
