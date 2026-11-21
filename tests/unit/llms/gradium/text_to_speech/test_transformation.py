from types import MappingProxyType
from typing import Final

import httpx
import pytest
import respx
from pydantic import TypeAdapter

import litellm
from litellm.llms.gradium.common_utils import GradiumException
from litellm.llms.gradium.text_to_speech.transformation import (
    GradiumTextToSpeechConfig,
)

_AUDIO: Final = b"RIFFgradium-audio"
_API_BASE: Final = "https://eu.api.gradium.ai"
_API_KEY: Final = "gd_test_key"
_URL: Final = f"{_API_BASE}/api/speech/tts"
_JSON_BODY: Final = TypeAdapter(dict[str, object])


@pytest.fixture
def gradium_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)


def _sent_body(route: respx.Route) -> dict[str, object]:
    return _JSON_BODY.validate_json(route.calls[-1].request.content)


def test_maps_openai_voice_names_to_gradium_voice_ids():
    config: Final = GradiumTextToSpeechConfig()

    voice, _ = config.map_openai_params(model="default", optional_params={}, voice="alloy")

    assert voice == "YTpq7expH9539ERJ"


@pytest.mark.parametrize("voice", ["Alloy", " alloy ", "ALLOY"])
def test_maps_openai_voice_names_regardless_of_case_and_surrounding_spaces(voice: str):
    config: Final = GradiumTextToSpeechConfig()

    mapped, _ = config.map_openai_params(model="default", optional_params={}, voice=voice)

    assert mapped == "YTpq7expH9539ERJ"


def test_maps_every_current_openai_voice_name_to_its_own_gradium_voice():
    config: Final = GradiumTextToSpeechConfig()
    openai_voices: Final = tuple("alloy ash ballad coral echo fable nova onyx sage shimmer verse marin cedar".split())

    mapped: Final = tuple(
        config.map_openai_params(model="default", optional_params={}, voice=name)[0] for name in openai_voices
    )

    assert not set(mapped) & set(openai_voices)
    assert len(set(mapped)) == len(openai_voices)


def test_passes_unknown_voices_through_as_raw_gradium_ids():
    config: Final = GradiumTextToSpeechConfig()

    voice, _ = config.map_openai_params(model="default", optional_params={}, voice="w9V1722uEmTkWqnR")

    assert voice == "w9V1722uEmTkWqnR"


def test_requires_a_voice():
    config: Final = GradiumTextToSpeechConfig()

    with pytest.raises(litellm.BadRequestError, match="voice_id is required"):
        config.map_openai_params(model="default", optional_params={}, voice=None)


@pytest.mark.parametrize(
    ("response_format", "expected"),
    [
        ("wav", "wav"),
        ("pcm", "pcm_24000"),
        ("opus", "opus"),
        ("ulaw_8000", "ulaw_8000"),
        ("alaw_8000", "alaw_8000"),
        ("pcm_16000", "pcm_16000"),
    ],
)
def test_maps_supported_openai_response_formats(response_format: str, expected: str):
    config: Final = GradiumTextToSpeechConfig()

    _, params = config.map_openai_params(
        model="default",
        optional_params={"response_format": response_format},
        voice="alloy",
    )

    assert params["output_format"] == expected


@pytest.mark.parametrize("response_format", ["mp3", "aac", "flac"])
def test_rejects_formats_gradium_cannot_encode(response_format: str):
    config: Final = GradiumTextToSpeechConfig()

    with pytest.raises(litellm.UnsupportedParamsError, match="does not support response_format"):
        config.map_openai_params(
            model="default",
            optional_params={"response_format": response_format},
            voice="alloy",
        )


def test_drops_unencodable_format_when_drop_params_is_set():
    config: Final = GradiumTextToSpeechConfig()

    _, params = config.map_openai_params(
        model="default",
        optional_params={"response_format": "mp3"},
        voice="alloy",
        drop_params=True,
    )

    assert "output_format" not in params


@pytest.mark.parametrize(("param", "value"), [("speed", 2.0), ("speed", 0.5), ("instructions", "whisper it")])
def test_rejects_openai_only_params_gradium_cannot_honor(param: str, value: object):
    config: Final = GradiumTextToSpeechConfig()

    with pytest.raises(litellm.UnsupportedParamsError, match=f"does not support {param}=") as excinfo:
        config.map_openai_params(model="default", optional_params={param: value}, voice="alloy")

    assert "response_format values" not in str(excinfo.value)


@pytest.mark.parametrize("speed", [1.0, 1])
def test_accepts_the_default_speed_without_sending_it(speed: float):
    config: Final = GradiumTextToSpeechConfig()

    _, params = config.map_openai_params(model="default", optional_params={"speed": speed}, voice="alloy")

    assert params == {}


def test_drops_unsupported_params_when_the_deployment_sets_drop_params():
    config: Final = GradiumTextToSpeechConfig()

    _, params = config.map_openai_params(
        model="default",
        optional_params={"response_format": "mp3", "speed": 2.0, "instructions": "whisper it"},
        voice="alloy",
        kwargs={"drop_params": True},
    )

    assert params == {}


def test_advertises_the_openai_params_it_honors():
    config: Final = GradiumTextToSpeechConfig()

    assert config.get_supported_openai_params(model="default") == ["voice", "response_format"]


def test_maps_upstream_failures_to_a_gradium_exception():
    config: Final = GradiumTextToSpeechConfig()

    error: Final = config.get_error_class(error_message="unknown voice", status_code=422, headers={})

    assert isinstance(error, GradiumException)
    assert error.status_code == 422


def test_requires_an_api_key():
    config: Final = GradiumTextToSpeechConfig()

    with pytest.raises(litellm.AuthenticationError, match="Gradium API key is required"):
        config.validate_environment(headers={}, model="default", api_key=None)


def test_sends_the_api_key_as_x_api_key():
    config: Final = GradiumTextToSpeechConfig()

    headers: Final = config.validate_environment(headers={}, model="default", api_key="gd_abc")

    assert headers["x-api-key"] == "gd_abc"


@pytest.mark.parametrize(
    "api_base",
    [
        "https://eu.api.gradium.ai",
        "https://eu.api.gradium.ai/",
        "https://eu.api.gradium.ai/api",
        "https://eu.api.gradium.ai/api/",
    ],
)
def test_builds_the_rest_tts_url_without_doubling_the_api_segment(api_base: str):
    config: Final = GradiumTextToSpeechConfig()

    url: Final = config.get_complete_url(model="default", api_base=api_base, litellm_params={})

    assert url == "https://eu.api.gradium.ai/api/speech/tts"


def test_requires_an_api_base():
    config: Final = GradiumTextToSpeechConfig()

    with pytest.raises(litellm.APIConnectionError, match="Gradium API base is required"):
        config.get_complete_url(model="default", api_base=None, litellm_params={})


def test_ignores_gradium_environment_variables(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GRADIUM_API_KEY", "gd_from_env")
    monkeypatch.setenv("GRADIUM_API_BASE", "https://eu.api.gradium.ai")
    config: Final = GradiumTextToSpeechConfig()

    with pytest.raises(litellm.AuthenticationError, match="Gradium API key is required"):
        config.validate_environment(headers={}, model="default", api_key=None)
    with pytest.raises(litellm.APIConnectionError, match="Gradium API base is required"):
        config.get_complete_url(model="default", api_base=None, litellm_params={})


def test_requests_raw_audio_rather_than_a_json_message_stream():
    config: Final = GradiumTextToSpeechConfig()

    data: Final = config.transform_text_to_speech_request(
        model="default",
        input="bonjour",
        voice="YTpq7expH9539ERJ",
        optional_params={},
        litellm_params={},
        headers={},
    )

    assert data["dict_body"]["only_audio"] is True


def test_omits_model_name_for_the_default_model():
    config: Final = GradiumTextToSpeechConfig()

    data: Final = config.transform_text_to_speech_request(
        model="default",
        input="bonjour",
        voice="YTpq7expH9539ERJ",
        optional_params={},
        litellm_params={},
        headers={},
    )

    assert "model_name" not in data["dict_body"]


def test_raises_on_gradium_plain_text_errors():
    config: Final = GradiumTextToSpeechConfig()
    raw_response: Final = httpx.Response(
        500,
        text="error from server 42: unknown voice",
        request=httpx.Request("POST", _URL),
    )

    with pytest.raises(GradiumException, match="unknown voice"):
        config.transform_text_to_speech_response(model="default", raw_response=raw_response, logging_obj=None)


def test_speech_calls_the_gradium_rest_endpoint(gradium_env: None):
    with respx.mock:
        route: Final = respx.post(_URL).mock(return_value=httpx.Response(200, content=_AUDIO))

        response: Final = litellm.speech(
            model="gradium/default",
            input="Bonjour",
            voice="alloy",
            response_format="wav",
            api_base=_API_BASE,
            api_key=_API_KEY,
        )

        assert response.content == _AUDIO
        assert route.calls[-1].request.headers["x-api-key"] == "gd_test_key"
        assert _sent_body(route) == {
            "text": "Bonjour",
            "voice_id": "YTpq7expH9539ERJ",
            "output_format": "wav",
            "only_audio": True,
        }


def test_speech_forwards_a_non_default_model_name(gradium_env: None):
    with respx.mock:
        route: Final = respx.post(_URL).mock(return_value=httpx.Response(200, content=_AUDIO))

        litellm.speech(model="gradium/next-model", input="Bonjour", voice="alloy", api_base=_API_BASE, api_key=_API_KEY)

        assert _sent_body(route)["model_name"] == "next-model"


def test_speech_forwards_gradium_request_fields_the_caller_sets(gradium_env: None):
    with respx.mock:
        route: Final = respx.post(_URL).mock(return_value=httpx.Response(200, content=_AUDIO))

        litellm.speech(
            model="gradium/default",
            input="Bonjour",
            voice="alloy",
            api_base=_API_BASE,
            api_key=_API_KEY,
            json_config={"padding_bonus": -1.5},
            drop_params=True,
            metadata={"team": "voice"},
            user="caller-1",
        )

        assert _sent_body(route) == {
            "text": "Bonjour",
            "voice_id": "YTpq7expH9539ERJ",
            "output_format": "wav",
            "only_audio": True,
            "json_config": {"padding_bonus": -1.5},
        }


def test_speech_forwards_json_config_sent_in_extra_body(gradium_env: None):
    with respx.mock:
        route: Final = respx.post(_URL).mock(return_value=httpx.Response(200, content=_AUDIO))

        litellm.speech(
            model="gradium/default",
            input="Bonjour",
            voice="alloy",
            api_base=_API_BASE,
            api_key=_API_KEY,
            extra_body={"json_config": '{"padding_bonus": 2.0}', "only_audio": False},
        )

        assert _sent_body(route)["json_config"] == '{"padding_bonus": 2.0}'
        assert _sent_body(route)["only_audio"] is True


def test_speech_sends_the_extra_body_json_config_when_both_are_set(gradium_env: None):
    with respx.mock:
        route: Final = respx.post(_URL).mock(return_value=httpx.Response(200, content=_AUDIO))

        litellm.speech(
            model="gradium/default",
            input="Bonjour",
            voice="alloy",
            api_base=_API_BASE,
            api_key=_API_KEY,
            json_config={"padding_bonus": -1.0},
            extra_body={"json_config": {"padding_bonus": 2.0}},
        )

        assert _sent_body(route)["json_config"] == {"padding_bonus": 2.0}


@pytest.mark.parametrize("extra_body", ["oops", ["json_config"], 5])
def test_speech_ignores_an_extra_body_that_is_not_an_object(gradium_env: None, extra_body: object):
    with respx.mock:
        route: Final = respx.post(_URL).mock(return_value=httpx.Response(200, content=_AUDIO))

        litellm.speech(
            model="gradium/default",
            input="Bonjour",
            voice="alloy",
            api_base=_API_BASE,
            api_key=_API_KEY,
            json_config={"padding_bonus": -1.0},
            extra_body=extra_body,
        )

        assert _sent_body(route)["json_config"] == {"padding_bonus": -1.0}


def test_speech_forwards_json_config_from_a_read_only_extra_body(gradium_env: None):
    with respx.mock:
        route: Final = respx.post(_URL).mock(return_value=httpx.Response(200, content=_AUDIO))

        litellm.speech(
            model="gradium/default",
            input="Bonjour",
            voice="alloy",
            api_base=_API_BASE,
            api_key=_API_KEY,
            extra_body=MappingProxyType({"json_config": {"padding_bonus": 2.0}}),
        )

        assert _sent_body(route)["json_config"] == {"padding_bonus": 2.0}


def test_speech_keeps_top_level_json_config_when_extra_body_sets_it_to_none(gradium_env: None):
    with respx.mock:
        route: Final = respx.post(_URL).mock(return_value=httpx.Response(200, content=_AUDIO))

        litellm.speech(
            model="gradium/default",
            input="Bonjour",
            voice="alloy",
            api_base=_API_BASE,
            api_key=_API_KEY,
            json_config={"padding_bonus": -1.0},
            extra_body={"json_config": None},
        )

        assert _sent_body(route)["json_config"] == {"padding_bonus": -1.0}


@pytest.mark.parametrize(
    ("credentials", "error", "status_code"),
    [
        ({"api_base": _API_BASE}, litellm.AuthenticationError, 401),
        ({"api_key": _API_KEY}, litellm.APIConnectionError, 500),
    ],
)
def test_speech_reports_missing_credentials_without_a_server_traceback(
    gradium_env: None, credentials: dict[str, str], error: type[Exception], status_code: int
):
    with pytest.raises(error, match="Set `api_") as excinfo:
        litellm.speech(model="gradium/default", input="Bonjour", voice="alloy", **credentials)

    assert getattr(excinfo.value, "status_code", None) == status_code
    assert "Traceback" not in str(excinfo.value)


@pytest.mark.asyncio
async def test_aspeech_calls_the_gradium_rest_endpoint(gradium_env: None):
    with respx.mock:
        route: Final = respx.post(_URL).mock(return_value=httpx.Response(200, content=_AUDIO))

        response: Final = await litellm.aspeech(
            model="gradium/default",
            input="Bonjour",
            voice="alloy",
            api_base=_API_BASE,
            api_key=_API_KEY,
            json_config={"padding_bonus": -1.0},
        )

        assert response.content == _AUDIO
        assert route.calls[-1].request.headers["x-api-key"] == _API_KEY
        assert _sent_body(route) == {
            "text": "Bonjour",
            "voice_id": "YTpq7expH9539ERJ",
            "output_format": "wav",
            "only_audio": True,
            "json_config": {"padding_bonus": -1.0},
        }


@pytest.mark.parametrize("voice", [None, " "])
def test_speech_rejects_a_missing_voice_with_a_400(gradium_env: None, voice: str | None):
    with pytest.raises(litellm.BadRequestError, match="voice_id is required") as excinfo:
        litellm.speech(model="gradium/default", input="Bonjour", voice=voice, api_base=_API_BASE, api_key=_API_KEY)

    assert excinfo.value.status_code == 400


@pytest.mark.parametrize(
    ("response_format", "gradium_content_type", "expected_content_type"),
    [
        ("ulaw_8000", "application/octet-stream", "audio/basic"),
        ("alaw_8000", "application/octet-stream", "audio/x-alaw-basic"),
        ("pcm_16000", "application/octet-stream", "audio/pcm"),
        ("pcm_16000", "Application/Octet-Stream; charset=binary", "audio/pcm"),
        ("wav", "audio/wav", "audio/wav"),
    ],
)
def test_speech_labels_raw_audio_with_its_media_type(
    gradium_env: None, response_format: str, gradium_content_type: str, expected_content_type: str
):
    with respx.mock:
        route: Final = respx.post(_URL).mock(
            return_value=httpx.Response(200, content=_AUDIO, headers={"content-type": gradium_content_type})
        )

        response: Final = litellm.speech(
            model="gradium/default",
            input="Bonjour",
            voice="alloy",
            response_format=response_format,
            api_base=_API_BASE,
            api_key=_API_KEY,
        )

        assert _sent_body(route)["output_format"] == response_format
        assert response.content_type == expected_content_type


@pytest.mark.parametrize("content_type", ["application/json", "text/html; charset=utf-8"])
@pytest.mark.asyncio
async def test_aspeech_rejects_a_success_response_that_is_not_audio(gradium_env: None, content_type: str):
    with respx.mock:
        respx.post(_URL).mock(
            return_value=httpx.Response(200, content=b'{"text": "Bonjour"}', headers={"content-type": content_type})
        )

        with pytest.raises(litellm.BadGatewayError) as raised:
            await litellm.aspeech(
                model="gradium/default",
                input="Bonjour",
                voice="alloy",
                response_format="pcm",
                api_base=_API_BASE,
                api_key=_API_KEY,
            )

        assert raised.value.status_code == 502
        assert content_type.split(";")[0] in str(raised.value)


def test_speech_honors_an_explicit_api_base_and_api_key(gradium_env: None):
    with respx.mock:
        route: Final = respx.post("https://eu.api.gradium.ai/api/speech/tts").mock(
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


def test_speech_tracks_cost_per_character(gradium_env: None, local_model_cost_map: None):
    with respx.mock:
        respx.post(_URL).mock(return_value=httpx.Response(200, content=_AUDIO))

        response: Final = litellm.speech(
            model="gradium/default", input="a" * 100, voice="alloy", api_base=_API_BASE, api_key=_API_KEY
        )

        price_per_character: Final = litellm.model_cost["gradium/default"]["input_cost_per_character"]
        assert price_per_character > 0
        assert response._hidden_params["response_cost"] == pytest.approx(100 * price_per_character)


@pytest.mark.parametrize("model", ["gradium/gradium-tts-beta", "gradium/gradium-tts-202602"])
def test_speech_prices_other_gradium_models_like_the_default_one(
    gradium_env: None, local_model_cost_map: None, model: str
):
    with respx.mock:
        respx.post(_URL).mock(return_value=httpx.Response(200, content=_AUDIO))

        response: Final = litellm.speech(
            model=model, input="Bonjour", voice="alloy", api_base=_API_BASE, api_key=_API_KEY
        )

        price_per_character: Final = litellm.model_cost[model]["input_cost_per_character"]
        assert price_per_character == litellm.model_cost["gradium/default"]["input_cost_per_character"]
        assert response._hidden_params["response_cost"] == pytest.approx(len("Bonjour") * price_per_character)


def test_speech_bills_whitespace_like_gradium_does(gradium_env: None, local_model_cost_map: None):
    spoken_text: Final = "Oui.     Non.     Oui."
    with respx.mock:
        respx.post(_URL).mock(return_value=httpx.Response(200, content=_AUDIO))

        response: Final = litellm.speech(
            model="gradium/default", input=spoken_text, voice="alloy", api_base=_API_BASE, api_key=_API_KEY
        )

        price_per_character: Final = litellm.model_cost["gradium/default"]["input_cost_per_character"]
        assert response._hidden_params["response_cost"] == pytest.approx(len(spoken_text) * price_per_character)
