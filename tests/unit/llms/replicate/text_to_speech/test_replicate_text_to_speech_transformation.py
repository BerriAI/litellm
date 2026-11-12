from typing import Final
from unittest.mock import Mock

import httpx
import pytest

from litellm.llms.replicate.common_utils import ReplicateError
from litellm.llms.replicate.text_to_speech.transformation import ReplicateTextToSpeechConfig
from litellm.types.llms.openai import HttpxBinaryResponseContent
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

API_BASE: Final = "https://api.replicate.com/v1"
MODEL: Final = "xai/grok-text-to-speech"
PREDICTIONS_URL: Final = f"{API_BASE}/models/{MODEL}/predictions"
AUDIO_URL: Final = "https://replicate.delivery/out/speech.mp3"


class _FakeClient:
    def __init__(self, *responses: httpx.Response) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(self, url: str, headers: dict[str, str] | None = None, params: object = None) -> httpx.Response:
        self.calls.append(url)
        return self.responses.pop(0)


class _FakeAsyncClient(_FakeClient):
    async def get(self, url: str, headers: dict[str, str] | None = None, params: object = None) -> httpx.Response:  # pyright: ignore[reportIncompatibleMethodOverride]  # async twin of the sync fake
        return super().get(url, headers=headers, params=params)


def _prediction_response(body: dict[str, object]) -> httpx.Response:
    request = httpx.Request("POST", PREDICTIONS_URL, headers={"Authorization": "Bearer k", "Prefer": "wait"})
    return httpx.Response(201, json=body, request=request)


def _audio(content: bytes, content_type: str = "audio/mpeg") -> httpx.Response:
    return httpx.Response(
        200, content=content, headers={"content-type": content_type}, request=httpx.Request("GET", AUDIO_URL)
    )


def test_provider_config_is_registered():
    config = ProviderConfigManager.get_provider_text_to_speech_config(model=MODEL, provider=LlmProviders.REPLICATE)
    assert isinstance(config, ReplicateTextToSpeechConfig)


def test_map_openai_params_keeps_voice_and_speed_only():
    voice, params = ReplicateTextToSpeechConfig().map_openai_params(
        MODEL, {"speed": 1.2, "response_format": "mp3", "instructions": "cheerful"}, voice="ara"
    )
    assert voice == "ara"
    assert params == {"speed": 1.2}
    assert ReplicateTextToSpeechConfig().map_openai_params(MODEL, {}, voice=None) == (None, {})


def test_validate_environment_sets_bearer_and_prefer_wait():
    headers = ReplicateTextToSpeechConfig().validate_environment({}, MODEL, api_key="r8_key")
    assert headers["Authorization"] == "Bearer r8_key"
    assert headers["Prefer"] == "wait"


def test_get_complete_url():
    config = ReplicateTextToSpeechConfig()
    assert config.get_complete_url(MODEL, None, {}) == PREDICTIONS_URL
    assert (
        config.get_complete_url(MODEL, "https://proxy.internal", {})
        == f"https://proxy.internal/v1/models/{MODEL}/predictions"
    )


@pytest.mark.parametrize(
    ("model", "voice_key"),
    [
        (MODEL, "voice"),
        ("minimax/speech-02-turbo", "voice_id"),
        ("inworld/realtime-tts-2", "voice_id"),
        ("qwen/qwen3-tts", "speaker"),
    ],
)
def test_request_body_uses_model_family_voice_key(model: str, voice_key: str):
    request = ReplicateTextToSpeechConfig().transform_text_to_speech_request(
        model, "hello there", "Wise_Woman", {"speed": 1.1}, {}, {}
    )
    assert request == {"dict_body": {"input": {"text": "hello there", voice_key: "Wise_Woman", "speed": 1.1}}}


def test_request_body_without_voice():
    request = ReplicateTextToSpeechConfig().transform_text_to_speech_request(MODEL, "hello", None, {}, {}, {})
    assert request == {"dict_body": {"input": {"text": "hello"}}}


def test_response_downloads_audio_from_output_url():
    client = _FakeClient(_audio(b"mp3-bytes"))
    config = ReplicateTextToSpeechConfig(sync_client_factory=lambda: client)  # pyright: ignore[reportArgumentType]  # fake stands in for HTTPHandler
    result = config.transform_text_to_speech_response(
        MODEL, _prediction_response({"id": "p1", "status": "succeeded", "output": AUDIO_URL}), Mock()
    )
    assert isinstance(result, HttpxBinaryResponseContent)
    assert result.content == b"mp3-bytes"
    assert result.response.headers["content-type"] == "audio/mpeg"
    assert client.calls == [AUDIO_URL]


@pytest.mark.asyncio
async def test_async_response_downloads_audio_with_async_client():
    client = _FakeAsyncClient(_audio(b"wav-bytes", "audio/wav"))
    config = ReplicateTextToSpeechConfig(async_client_factory=lambda: client)  # pyright: ignore[reportArgumentType]  # fake stands in for AsyncHTTPHandler
    result = await config.async_transform_text_to_speech_response(
        MODEL, _prediction_response({"id": "p1", "status": "succeeded", "output": [AUDIO_URL]}), Mock()
    )
    assert result.content == b"wav-bytes"
    assert result.response.headers["content-type"] == "audio/wav"


def test_failed_prediction_raises():
    config = ReplicateTextToSpeechConfig(sync_client_factory=lambda: _FakeClient())  # pyright: ignore[reportArgumentType]  # fake stands in for HTTPHandler
    with pytest.raises(ReplicateError, match="text too long"):
        config.transform_text_to_speech_response(
            MODEL, _prediction_response({"id": "p1", "status": "failed", "error": "text too long"}), Mock()
        )
    with pytest.raises(ReplicateError, match="without audio output"):
        config.transform_text_to_speech_response(
            MODEL, _prediction_response({"id": "p1", "status": "succeeded", "output": None}), Mock()
        )
