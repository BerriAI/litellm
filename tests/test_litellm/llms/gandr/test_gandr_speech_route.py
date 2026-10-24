"""
Route and edge coverage for the Gandr text to speech provider.

The transformation tests next to this file cover the happy paths. These
tests reach the remaining branches: voice objects, kwargs passthrough,
server-managed keys, the error class, extra_body merging, the response
wrapper, the provider config registry and the litellm.speech() route with
the HTTP handler mocked.
"""

import json
from unittest.mock import MagicMock

import httpx
import pytest
import respx

import litellm
from litellm import speech
from litellm.llms.gandr.common_utils import GandrException
from litellm.llms.gandr.text_to_speech.transformation import (
    GandrTextToSpeechConfig,
    _as_voice_str,
    _resolve_voice,
)
from litellm.types.llms.openai import HttpxBinaryResponseContent
from litellm.utils import ProviderConfigManager


def test_resolve_voice_accepts_strings_and_voice_objects():
    assert _resolve_voice("gandr-mia") == "gandr-mia"
    assert _resolve_voice("   ") is None
    assert _resolve_voice({"voice_id": "gandr-ava"}) == "gandr-ava"
    assert _resolve_voice({"id": "gandr-leo"}) == "gandr-leo"
    assert _resolve_voice({"name": "gandr-dane"}) == "gandr-dane"
    assert _resolve_voice({"voice_id": "", "name": "gandr-jenny"}) == "gandr-jenny"
    assert _resolve_voice({"other": "x"}) is None
    assert _resolve_voice(42) is None


def test_as_voice_str():
    assert _as_voice_str("gandr-mia") == "gandr-mia"
    assert _as_voice_str("") is None
    assert _as_voice_str(None) is None
    assert _as_voice_str(7) is None


def test_map_openai_params_voice_object_and_voice_id_fallback():
    config = GandrTextToSpeechConfig()

    voice, mapped = config.map_openai_params(
        model="gandr/mia",
        optional_params={"voice_id": "gandr-lewis"},
        voice={"voice_id": "gandr-ava"},
    )
    assert voice == "gandr-ava"
    assert "voice_id" not in mapped

    voice, mapped = config.map_openai_params(
        model="gandr/mia",
        optional_params={"voice_id": "gandr-lewis"},
        voice=None,
    )
    assert voice == "gandr-lewis"
    assert mapped["response_format"] == "wav"


def test_map_openai_params_drops_unparseable_speed_and_passes_kwargs():
    config = GandrTextToSpeechConfig()

    voice, mapped = config.map_openai_params(
        model="gandr/mia",
        optional_params={"speed": "fast", "response_format": "mp3", "unused": None},
        voice="gandr-mia",
        kwargs={
            "temperature": 0.4,
            "model": "should-be-ignored",
            "user": "should-be-ignored",
            "extra_body": {"ignored": True},
            "metadata": {"ignored": True},
            "empty": None,
        },
    )

    assert voice == "gandr-mia"
    assert "speed" not in mapped
    assert "unused" not in mapped
    assert mapped["response_format"] == "mp3"
    assert mapped["temperature"] == 0.4
    for reserved in ("model", "user", "extra_body", "metadata", "empty"):
        assert reserved not in mapped


def test_validate_environment_uses_server_key_for_default_base(monkeypatch):
    monkeypatch.setattr(litellm, "api_key", None)
    monkeypatch.setenv("GANDR_API_KEY", "gnd_server_secret")
    monkeypatch.delenv("GANDR_API_BASE", raising=False)
    config = GandrTextToSpeechConfig()

    headers = config.validate_environment(headers={}, model="gandr/mia")
    assert headers["Authorization"] == "Bearer gnd_server_secret"
    assert headers["Content-Type"] == "application/json"

    headers = config.validate_environment(
        headers={}, model="gandr/mia", api_base="https://tts.gandr.ai/v1/"
    )
    assert headers["Authorization"] == "Bearer gnd_server_secret"


def test_validate_environment_trusts_operator_api_base(monkeypatch):
    monkeypatch.setattr(litellm, "api_key", None)
    monkeypatch.setenv("GANDR_API_KEY", "gnd_server_secret")
    monkeypatch.setenv("GANDR_API_BASE", "https://tts.internal.example/v1")
    config = GandrTextToSpeechConfig()

    headers = config.validate_environment(
        headers={}, model="gandr/mia", api_base="https://tts.internal.example/v1"
    )
    assert headers["Authorization"] == "Bearer gnd_server_secret"

    with pytest.raises(ValueError, match="caller-supplied api_base"):
        config.validate_environment(
            headers={}, model="gandr/mia", api_base="https://elsewhere.example/v1"
        )


def test_validate_environment_requires_some_key(monkeypatch):
    monkeypatch.setattr(litellm, "api_key", None)
    monkeypatch.delenv("GANDR_API_KEY", raising=False)
    config = GandrTextToSpeechConfig()

    with pytest.raises(ValueError, match="Gandr API key is required"):
        config.validate_environment(headers={}, model="gandr/mia")


def test_get_error_class_returns_gandr_exception():
    config = GandrTextToSpeechConfig()

    error = config.get_error_class(
        error_message="quota exhausted", status_code=402, headers={"x-test": "1"}
    )

    assert isinstance(error, GandrException)
    assert error.status_code == 402
    assert "quota exhausted" in str(error)


def test_transform_request_merges_extra_body_and_skips_none():
    config = GandrTextToSpeechConfig()

    data = config.transform_text_to_speech_request(
        model="tts-1",
        input="Order 4471 is ready",
        voice="gandr-mia",
        optional_params={
            "temperature": 0.3,
            "dropped": None,
            "extra_body": {"cfg": 0.6, "ignored": None},
        },
        litellm_params={},
        headers={},
    )

    body = data["dict_body"]
    assert body["response_format"] == "wav"
    assert body["speed"] == 1.0
    assert body["temperature"] == 0.3
    assert body["cfg"] == 0.6
    assert "dropped" not in body
    assert "ignored" not in body
    assert "extra_body" not in body


def test_transform_response_wraps_binary_content():
    config = GandrTextToSpeechConfig()
    raw = httpx.Response(200, content=b"RIFF....WAVE", headers={"Content-Type": "audio/wav"})

    wrapped = config.transform_text_to_speech_response(
        model="gandr/mia", raw_response=raw, logging_obj=MagicMock()
    )

    assert isinstance(wrapped, HttpxBinaryResponseContent)
    assert wrapped.content == b"RIFF....WAVE"


def test_get_complete_url_honours_operator_base(monkeypatch):
    monkeypatch.setenv("GANDR_API_BASE", "https://tts.internal.example/v1/")
    config = GandrTextToSpeechConfig()

    assert (
        config.get_complete_url(model="gandr/mia", api_base=None, litellm_params={})
        == "https://tts.internal.example/v1/audio/speech"
    )


def test_provider_config_registry_returns_gandr_config():
    config = ProviderConfigManager.get_provider_text_to_speech_config(
        model="gandr/mia", provider=litellm.LlmProviders.GANDR
    )

    assert isinstance(config, GandrTextToSpeechConfig)
    assert litellm.LlmProviders("gandr") is litellm.LlmProviders.GANDR


def test_speech_routes_to_gandr_endpoint(respx_mock: respx.MockRouter):
    route = respx_mock.post("https://tts.gandr.ai/v1/audio/speech").mock(
        return_value=httpx.Response(
            200, content=b"fake wav bytes", headers={"Content-Type": "audio/wav"}
        )
    )

    response = speech(
        model="gandr/mia",
        voice="gandr-mia",
        input="Your order number is 4471.",
        api_key="gnd_test_key",
        response_format="wav",
    )

    assert route.called
    request = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer gnd_test_key"
    assert request.headers["Content-Type"] == "application/json"
    body = json.loads(request.content)
    assert body == {
        "input": "Your order number is 4471.",
        "model": "mia",
        "voice": "gandr-mia",
        "response_format": "wav",
        "speed": 1.0,
    }
    assert response.content == b"fake wav bytes"


def test_speech_honours_caller_api_base_with_explicit_key(respx_mock: respx.MockRouter):
    route = respx_mock.post("https://staging.example.com/v1/audio/speech").mock(
        return_value=httpx.Response(200, content=b"pcm", headers={"Content-Type": "audio/pcm"})
    )

    response = speech(
        model="gandr/mia",
        voice="gandr-ava",
        input="Hello.",
        api_key="gnd_caller_key",
        api_base="https://staging.example.com/v1",
        response_format="pcm",
    )

    assert route.called
    assert route.calls.last.request.headers["Authorization"] == "Bearer gnd_caller_key"
    assert json.loads(route.calls.last.request.content)["response_format"] == "pcm"
    assert response.content == b"pcm"


def test_speech_gandr_requires_a_voice(respx_mock: respx.MockRouter):
    route = respx_mock.post("https://tts.gandr.ai/v1/audio/speech")

    # The litellm client wrapper may re-raise inside its own exception
    # types; the message is the stable contract.
    with pytest.raises(Exception, match="Gandr voice is required"):
        speech(model="gandr/mia", input="No voice given", api_key="gnd_test_key")
    assert not route.called
