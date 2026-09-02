import pytest

from litellm.llms.gandr.text_to_speech.transformation import (
    GandrTextToSpeechConfig,
)


def test_gandr_default_url():
    config = GandrTextToSpeechConfig()

    url = config.get_complete_url(
        model="gandr/mia",
        api_base=None,
        litellm_params={},
    )

    assert url == "https://tts.gandr.ai/v1/audio/speech"


def test_gandr_custom_api_base():
    config = GandrTextToSpeechConfig()

    url = config.get_complete_url(
        model="gandr/mia",
        api_base="https://tts.gandr.ai/v1",
        litellm_params={},
    )

    assert url == "https://tts.gandr.ai/v1/audio/speech"


def test_gandr_untrusted_api_base_rejects_server_key(monkeypatch):
    monkeypatch.setenv("GANDR_API_KEY", "gnd_server_secret")
    config = GandrTextToSpeechConfig()

    with pytest.raises(ValueError, match="caller-supplied api_base"):
        config.validate_environment(
            headers={},
            model="gandr/mia",
            api_key=None,
            api_base="https://attacker.example.com/v1",
        )


def test_gandr_custom_api_base_with_explicit_key(monkeypatch):
    monkeypatch.setenv("GANDR_API_KEY", "gnd_server_secret")
    config = GandrTextToSpeechConfig()

    headers = config.validate_environment(
        headers={},
        model="gandr/mia",
        api_key="gnd_caller_key",
        api_base="https://staging.example.com/v1",
    )

    assert headers["Authorization"] == "Bearer gnd_caller_key"


def test_gandr_supported_openai_params():
    config = GandrTextToSpeechConfig()

    assert config.get_supported_openai_params(model="gandr/mia") == [
        "voice",
        "response_format",
        "speed",
    ]


def test_gandr_map_openai_params_default_format():
    config = GandrTextToSpeechConfig()

    voice, mapped = config.map_openai_params(
        model="gandr/mia",
        optional_params={"speed": 1.1},
        voice="alloy",
    )

    assert voice == "alloy"
    assert mapped["response_format"] == "wav"
    assert mapped["speed"] == 1.1


def test_gandr_map_openai_params_passthrough():
    config = GandrTextToSpeechConfig()

    voice, mapped = config.map_openai_params(
        model="gandr/mia",
        optional_params={"response_format": "pcm", "speed": 0.8},
        voice="gandr-dane",
    )

    assert voice == "gandr-dane"
    assert mapped["response_format"] == "pcm"
    assert mapped["speed"] == 0.8


def test_gandr_map_openai_params_requires_voice():
    config = GandrTextToSpeechConfig()

    with pytest.raises(ValueError, match="Gandr voice is required"):
        config.map_openai_params(
            model="gandr/mia",
            optional_params={},
            voice=None,
        )


def test_gandr_transform_request_body():
    config = GandrTextToSpeechConfig()

    data = config.transform_text_to_speech_request(
        model="tts-1",
        input="Hello from Gandr",
        voice="alloy",
        optional_params={"response_format": "wav", "speed": 1.0},
        litellm_params={},
        headers={},
    )

    assert data["dict_body"] == {
        "input": "Hello from Gandr",
        "model": "tts-1",
        "voice": "alloy",
        "response_format": "wav",
        "speed": 1.0,
    }
    assert data["headers"]["Content-Type"] == "application/json"
