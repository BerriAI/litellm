import pytest

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

    assert (
        url
        == "https://api.elevenlabs.io/v1/text-to-speech/voice%2F..%2F..%2Fmodels%3Fx%3D1%23frag"
    )


def test_should_reject_dot_segment_elevenlabs_voice_id():
    config = ElevenLabsTextToSpeechConfig()

    with pytest.raises(ValueError, match="voice_id cannot be a dot path segment"):
        config.get_complete_url(
            model="elevenlabs/tts",
            api_base="https://api.elevenlabs.io",
            litellm_params={config.ELEVENLABS_VOICE_ID_KEY: ".."},
        )
