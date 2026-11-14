from typing import Final
from unittest.mock import Mock

import httpx
import pytest

import litellm
from litellm.llms.replicate.common_utils import ReplicateError
from litellm.llms.replicate.text_to_speech.transformation import ReplicateTextToSpeechConfig
from litellm.types.llms.openai import HttpxBinaryResponseContent
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

API_BASE: Final = "https://api.replicate.com/v1"
MODEL: Final = "xai/grok-text-to-speech"
PREDICTIONS_URL: Final = f"{API_BASE}/models/{MODEL}/predictions"
AUDIO_URL: Final = "https://replicate.delivery/out/speech.mp3"


@pytest.fixture(autouse=True)
def _skip_url_dns_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "user_url_validation", False)


class _FakeClient:
    def __init__(self, *responses: httpx.Response) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(
        self, url: str, headers: dict[str, str] | None = None, params: object = None, **kwargs: object
    ) -> httpx.Response:
        self.calls.append(url)
        return self.responses.pop(0)


class _FakeAsyncClient(_FakeClient):
    async def get(
        self, url: str, headers: dict[str, str] | None = None, params: object = None, **kwargs: object
    ) -> httpx.Response:  # pyright: ignore[reportIncompatibleMethodOverride]  # async twin of the sync fake
        return super().get(url, headers=headers, params=params, **kwargs)


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


def test_map_openai_params_passes_extra_body_as_native_input():
    _, params = ReplicateTextToSpeechConfig().map_openai_params(
        MODEL, {"speed": 0.9}, voice="ara", kwargs={"extra_body": {"output_format": "wav", "speed": 2}}
    )
    assert params == {"output_format": "wav", "speed": 0.9}
    _, ignored = ReplicateTextToSpeechConfig().map_openai_params(MODEL, {}, voice=None, kwargs={"extra_body": "x"})
    assert ignored == {}


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
        ("deployments/minimax/my-tts", "voice_id"),
        ("qwen/qwen3-tts:0123abcd", "speaker"),
    ],
)
def test_request_body_uses_model_family_voice_key(model: str, voice_key: str):
    request = ReplicateTextToSpeechConfig().transform_text_to_speech_request(
        model, "hello there", "Wise_Woman", {"speed": 1.1}, {}, {}
    )
    assert request["dict_body"]["input"] == {"text": "hello there", voice_key: "Wise_Woman", "speed": 1.1}


def test_request_body_without_voice():
    request = ReplicateTextToSpeechConfig().transform_text_to_speech_request(MODEL, "hello", None, {}, {}, {})
    assert request == {"dict_body": {"input": {"text": "hello"}}}


def test_request_input_wins_over_extra_body_text():
    _, params = ReplicateTextToSpeechConfig().map_openai_params(
        MODEL, {}, voice="ara", kwargs={"extra_body": {"text": "a much longer text that is not billed"}}
    )
    request = ReplicateTextToSpeechConfig().transform_text_to_speech_request(MODEL, "hello", "ara", params, {}, {})
    assert request["dict_body"]["input"] == {"text": "hello", "voice": "ara"}


def test_response_downloads_audio_from_output_url():
    client = _FakeClient(_audio(b"mp3-bytes"))
    config = ReplicateTextToSpeechConfig(sync_client_factory=lambda _ssl: client)  # pyright: ignore[reportArgumentType]  # fake stands in for HTTPHandler
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
    config = ReplicateTextToSpeechConfig(async_client_factory=lambda _ssl: client)  # pyright: ignore[reportArgumentType]  # fake stands in for AsyncHTTPHandler
    result = await config.async_transform_text_to_speech_response(
        MODEL, _prediction_response({"id": "p1", "status": "succeeded", "output": [AUDIO_URL]}), Mock()
    )
    assert result.content == b"wav-bytes"
    assert result.response.headers["content-type"] == "audio/wav"


def test_response_client_uses_the_request_ssl_verify():
    seen: list[object] = []

    def factory(ssl_verify: object) -> _FakeClient:
        seen.append(ssl_verify)
        return _FakeClient(_audio(b"mp3-bytes"))

    config = ReplicateTextToSpeechConfig(sync_client_factory=factory)  # pyright: ignore[reportArgumentType]  # fake stands in for HTTPHandler
    logging_obj = Mock()
    logging_obj.litellm_params = {"ssl_verify": "/certs/gateway.pem"}
    config.transform_text_to_speech_response(
        MODEL, _prediction_response({"id": "p1", "status": "succeeded", "output": AUDIO_URL}), logging_obj
    )
    assert seen == ["/certs/gateway.pem"]


def test_failed_prediction_raises():
    config = ReplicateTextToSpeechConfig(sync_client_factory=lambda _ssl: _FakeClient())  # pyright: ignore[reportArgumentType]  # fake stands in for HTTPHandler
    with pytest.raises(ReplicateError, match="text too long"):
        config.transform_text_to_speech_response(
            MODEL, _prediction_response({"id": "p1", "status": "failed", "error": "text too long"}), Mock()
        )
    with pytest.raises(ReplicateError, match="without audio output"):
        config.transform_text_to_speech_response(
            MODEL, _prediction_response({"id": "p1", "status": "succeeded", "output": None}), Mock()
        )


def test_response_usage_comes_from_prediction_token_metrics():
    config = ReplicateTextToSpeechConfig(sync_client_factory=lambda _ssl: _FakeClient(_audio(b"mp3-bytes")))  # pyright: ignore[reportArgumentType]  # fake stands in for HTTPHandler
    result = config.transform_text_to_speech_response(
        MODEL,
        _prediction_response(
            {
                "id": "p1",
                "status": "succeeded",
                "output": AUDIO_URL,
                "metrics": {"predict_time": 2.3, "token_input_count": 8, "token_output_count": 32},
            }
        ),
        Mock(),
    )
    assert result.usage is not None
    assert (result.usage.prompt_tokens, result.usage.completion_tokens, result.usage.total_tokens) == (8, 32, 40)
    assert result.logging_summary()["usage"]["completion_tokens"] == 32


@pytest.mark.parametrize(
    "metrics",
    [
        {"predict_time": 2.3, "token_input_count": 8, "token_output_count": 32},
        {"predict_time": 2.3, "input_token_count": 8, "output_token_count": 32},
    ],
)
def test_response_usage_accepts_both_replicate_metric_spellings(metrics: dict[str, object]):
    config = ReplicateTextToSpeechConfig(sync_client_factory=lambda _ssl: _FakeClient(_audio(b"mp3-bytes")))  # pyright: ignore[reportArgumentType]  # fake stands in for HTTPHandler
    result = config.transform_text_to_speech_response(
        MODEL,
        _prediction_response({"id": "p1", "status": "succeeded", "output": AUDIO_URL, "metrics": metrics}),
        Mock(),
    )
    assert result.usage is not None
    assert (result.usage.prompt_tokens, result.usage.completion_tokens) == (8, 32)


@pytest.mark.parametrize("metrics", [None, {"predict_time": 2.3}, {"token_input_count": 0, "token_output_count": -1}])
def test_response_without_token_metrics_has_no_usage(metrics: dict[str, object] | None):
    config = ReplicateTextToSpeechConfig(sync_client_factory=lambda _ssl: _FakeClient(_audio(b"mp3-bytes")))  # pyright: ignore[reportArgumentType]  # fake stands in for HTTPHandler
    body: dict[str, object] = {"id": "p1", "status": "succeeded", "output": AUDIO_URL}
    result = config.transform_text_to_speech_response(
        MODEL, _prediction_response(body if metrics is None else {**body, "metrics": metrics}), Mock()
    )
    assert result.usage is None
