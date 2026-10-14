import json
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.llms.gradium.common_utils import GradiumException
from litellm.llms.gradium.text_to_speech.transformation import (
    GradiumTextToSpeechConfig,
)

_AUDIO: Final = b"RIFFgradium-audio"
_URL: Final = "https://api.gradium.ai/api/post/speech/tts"


@pytest.fixture
def gradium_env(monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("GRADIUM_API_KEY", "gd_test_key")


def _sent_body(route) -> dict:
    return json.loads(route.calls[-1].request.content)


def test_maps_openai_voice_names_to_gradium_voice_ids():
    config = GradiumTextToSpeechConfig()

    voice, _ = config.map_openai_params(model="default", optional_params={}, voice="alloy")

    assert voice == "YTpq7expH9539ERJ"


def test_passes_unknown_voices_through_as_raw_gradium_ids():
    config = GradiumTextToSpeechConfig()

    voice, _ = config.map_openai_params(model="default", optional_params={}, voice="w9V1722uEmTkWqnR")

    assert voice == "w9V1722uEmTkWqnR"


def test_requires_a_voice():
    config = GradiumTextToSpeechConfig()

    with pytest.raises(ValueError, match="voice_id is required"):
        config.map_openai_params(model="default", optional_params={}, voice=None)


@pytest.mark.parametrize(
    ("response_format", "expected"),
    [("wav", "wav"), ("pcm", "pcm_24000"), ("opus", "opus")],
)
def test_maps_supported_openai_response_formats(response_format: str, expected: str):
    config = GradiumTextToSpeechConfig()

    _, params = config.map_openai_params(
        model="default",
        optional_params={"response_format": response_format},
        voice="alloy",
    )

    assert params["output_format"] == expected


@pytest.mark.parametrize("response_format", ["mp3", "aac", "flac"])
def test_rejects_formats_gradium_cannot_encode(response_format: str):
    config = GradiumTextToSpeechConfig()

    with pytest.raises(litellm.UnsupportedParamsError, match="does not support response_format"):
        config.map_openai_params(
            model="default",
            optional_params={"response_format": response_format},
            voice="alloy",
        )


def test_drops_unencodable_format_when_drop_params_is_set():
    config = GradiumTextToSpeechConfig()

    _, params = config.map_openai_params(
        model="default",
        optional_params={"response_format": "mp3"},
        voice="alloy",
        drop_params=True,
    )

    assert "output_format" not in params


def test_drops_openai_only_speed_and_instructions():
    config = GradiumTextToSpeechConfig()

    _, params = config.map_openai_params(
        model="default",
        optional_params={"speed": 2.0, "instructions": "whisper it"},
        voice="alloy",
    )

    assert "speed" not in params
    assert "instructions" not in params


def test_advertises_the_openai_params_it_honors():
    config = GradiumTextToSpeechConfig()

    assert config.get_supported_openai_params(model="default") == ["voice", "response_format"]


def test_maps_upstream_failures_to_a_gradium_exception():
    config = GradiumTextToSpeechConfig()

    error = config.get_error_class(error_message="unknown voice", status_code=422, headers={})

    assert isinstance(error, GradiumException)
    assert error.status_code == 422


def test_requires_an_api_key():
    config = GradiumTextToSpeechConfig()

    with pytest.raises(ValueError, match="Gradium API key is required"):
        config.validate_environment(headers={}, model="default", api_key=None)


def test_sends_the_api_key_as_x_api_key():
    config = GradiumTextToSpeechConfig()

    headers = config.validate_environment(headers={}, model="default", api_key="gd_abc")

    assert headers["x-api-key"] == "gd_abc"


@pytest.mark.parametrize(
    "api_base",
    [
        "https://eu.api.gradium.ai",
        "https://eu.api.gradium.ai/",
        # Gradium documents its regional bases with /api already appended.
        "https://eu.api.gradium.ai/api",
        "https://eu.api.gradium.ai/api/",
    ],
)
def test_builds_the_rest_tts_url_without_doubling_the_api_segment(api_base: str):
    config = GradiumTextToSpeechConfig()

    url = config.get_complete_url(model="default", api_base=api_base, litellm_params={})

    assert url == "https://eu.api.gradium.ai/api/post/speech/tts"


def test_defaults_to_the_latency_routed_endpoint():
    config = GradiumTextToSpeechConfig()

    url = config.get_complete_url(model="default", api_base=None, litellm_params={})

    assert url == "https://api.gradium.ai/api/post/speech/tts"


def test_requests_raw_audio_rather_than_a_json_message_stream():
    config = GradiumTextToSpeechConfig()

    data = config.transform_text_to_speech_request(
        model="default",
        input="bonjour",
        voice="YTpq7expH9539ERJ",
        optional_params={},
        litellm_params={},
        headers={},
    )

    assert data["dict_body"]["only_audio"] is True


def test_omits_model_name_for_the_default_model():
    config = GradiumTextToSpeechConfig()

    data = config.transform_text_to_speech_request(
        model="default",
        input="bonjour",
        voice="YTpq7expH9539ERJ",
        optional_params={},
        litellm_params={},
        headers={},
    )

    assert "model_name" not in data["dict_body"]


def test_raises_on_gradium_plain_text_errors():
    config = GradiumTextToSpeechConfig()
    raw_response = httpx.Response(
        500,
        text="error from server 42: unknown voice",
        request=httpx.Request("POST", _URL),
    )

    with pytest.raises(GradiumException, match="unknown voice"):
        config.transform_text_to_speech_response(model="default", raw_response=raw_response, logging_obj=None)


def test_speech_calls_the_gradium_rest_endpoint(gradium_env):
    with respx.mock:
        route = respx.post(_URL).mock(return_value=httpx.Response(200, content=_AUDIO))

        response = litellm.speech(
            model="gradium/default",
            input="Bonjour",
            voice="alloy",
            response_format="wav",
        )

        assert response.content == _AUDIO
        assert route.calls[-1].request.headers["x-api-key"] == "gd_test_key"
        assert _sent_body(route) == {
            "text": "Bonjour",
            "voice_id": "YTpq7expH9539ERJ",
            "output_format": "wav",
            "only_audio": True,
        }


def test_speech_forwards_a_non_default_model_name(gradium_env):
    with respx.mock:
        route = respx.post(_URL).mock(return_value=httpx.Response(200, content=_AUDIO))

        litellm.speech(model="gradium/pro-voice-clone", input="Bonjour", voice="alloy")

        assert _sent_body(route)["model_name"] == "pro-voice-clone"


@pytest.mark.asyncio
async def test_aspeech_calls_the_gradium_rest_endpoint(gradium_env):
    with respx.mock:
        respx.post(_URL).mock(return_value=httpx.Response(200, content=_AUDIO))

        response = await litellm.aspeech(model="gradium/default", input="Bonjour", voice="alloy")

        assert response.content == _AUDIO


def test_speech_rejects_a_missing_voice(gradium_env):
    with pytest.raises(ValueError, match="voice_id is required"):
        litellm.speech(model="gradium/default", input="Bonjour", voice=None)


def test_speech_rejects_a_missing_voice_without_a_provider_config(gradium_env, monkeypatch):
    """`speech()` guards the voice itself on the path where no config was resolved."""
    monkeypatch.setattr(
        litellm.utils.ProviderConfigManager,
        "get_provider_text_to_speech_config",
        staticmethod(lambda **_: None),
    )

    with pytest.raises(litellm.BadRequestError, match="must resolve to a Gradium voice id"):
        litellm.speech(model="gradium/default", input="Bonjour", voice=None)


def test_speech_honors_an_explicit_api_base_and_api_key(gradium_env):
    with respx.mock:
        route = respx.post("https://eu.api.gradium.ai/api/post/speech/tts").mock(
            return_value=httpx.Response(200, content=_AUDIO)
        )

        litellm.speech(
            model="gradium/default",
            input="Bonjour",
            voice="alloy",
            api_base="https://eu.api.gradium.ai",
            api_key="gd_explicit",
        )

        assert route.calls[-1].request.headers["x-api-key"] == "gd_explicit"


def test_speech_falls_back_to_its_own_config(gradium_env, monkeypatch):
    """`speech()` instantiates the config itself when the manager returns none."""
    monkeypatch.setattr(
        litellm.utils.ProviderConfigManager,
        "get_provider_text_to_speech_config",
        staticmethod(lambda **_: None),
    )

    with respx.mock:
        respx.post(_URL).mock(return_value=httpx.Response(200, content=_AUDIO))

        response = litellm.speech(model="gradium/default", input="Bonjour", voice="alloy")

        assert response.content == _AUDIO


def test_speech_tracks_cost_per_character(gradium_env, monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")

    with respx.mock:
        respx.post(_URL).mock(return_value=httpx.Response(200, content=_AUDIO))

        response = litellm.speech(model="gradium/default", input="a" * 100, voice="alloy")

        assert response._hidden_params["response_cost"] == pytest.approx(100 * 6.9e-05)
