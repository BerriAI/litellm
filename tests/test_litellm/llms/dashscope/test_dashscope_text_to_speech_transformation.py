"""
Unit tests for DashScope text-to-speech support (qwen3-tts-vc).
"""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from litellm.llms.dashscope.text_to_speech.transformation import (
    DEFAULT_API_BASE,
    DashScopeTextToSpeechConfig,
)


@pytest.fixture
def config() -> DashScopeTextToSpeechConfig:
    return DashScopeTextToSpeechConfig()


def test_provider_routing():
    from litellm.utils import ProviderConfigManager, get_llm_provider
    from litellm.types.utils import LlmProviders

    _, provider, _, _ = get_llm_provider("dashscope/qwen3-tts-vc")
    assert provider == "dashscope"

    cfg = ProviderConfigManager.get_provider_text_to_speech_config(
        "qwen3-tts-vc", LlmProviders.DASHSCOPE
    )
    assert isinstance(cfg, DashScopeTextToSpeechConfig)


def test_get_complete_url_default(config: DashScopeTextToSpeechConfig):
    assert config.get_complete_url("qwen3-tts-vc", None, {}) == DEFAULT_API_BASE


def test_validate_environment_requires_key(config: DashScopeTextToSpeechConfig):
    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY is not set"):
        config.validate_environment(headers={}, model="qwen3-tts-vc", api_key=None)

    headers = config.validate_environment(headers={}, model="qwen3-tts-vc", api_key="sk-x")
    assert headers["Authorization"] == "Bearer sk-x"
    assert headers["Content-Type"] == "application/json"


def test_map_openai_params_drops_unsupported(config: DashScopeTextToSpeechConfig):
    voice, params = config.map_openai_params(
        model="qwen3-tts-vc",
        optional_params={"response_format": "mp3", "speed": 1.0, "extra_body": {"language_type": "English"}},
        voice="my-voice",
    )
    assert voice == "my-voice"
    assert "response_format" not in params
    assert "speed" not in params
    assert params["language_type"] == "English"


def test_transform_request_shape(config: DashScopeTextToSpeechConfig):
    data = config.transform_text_to_speech_request(
        model="qwen3-tts-vc",
        input="hello world",
        voice="my-voice",
        optional_params={"language_type": "English"},
        litellm_params={},
        headers={},
    )
    body = data["dict_body"]
    assert body["model"] == "qwen3-tts-vc"
    assert body["input"]["text"] == "hello world"
    assert body["input"]["voice"] == "my-voice"
    assert body["input"]["language_type"] == "English"


def test_transform_response_downloads_audio(config: DashScopeTextToSpeechConfig):
    raw = MagicMock(spec=httpx.Response)
    raw.status_code = 200
    raw.json.return_value = {"output": {"audio": {"url": "https://cdn/audio.wav"}}}

    audio_resp = httpx.Response(
        status_code=200,
        headers={"content-type": "audio/wav"},
        content=b"RIFFfakeaudio",
        request=httpx.Request("GET", "https://cdn/audio.wav"),
    )

    with patch("httpx.get", return_value=audio_resp) as mock_get:
        result = config.transform_text_to_speech_response(
            model="qwen3-tts-vc", raw_response=raw, logging_obj=MagicMock()
        )
        mock_get.assert_called_once_with("https://cdn/audio.wav", timeout=60.0)

    assert result.content == b"RIFFfakeaudio"


def test_transform_response_raises_without_audio_url(config: DashScopeTextToSpeechConfig):
    raw = MagicMock(spec=httpx.Response)
    raw.status_code = 200
    raw.headers = httpx.Headers({})
    raw.json.return_value = {"output": {}}
    with pytest.raises(Exception, match="No audio url"):
        config.transform_text_to_speech_response(
            model="qwen3-tts-vc", raw_response=raw, logging_obj=MagicMock()
        )


def test_speech_dispatches_to_dashscope_handler():
    import litellm

    fake_audio = MagicMock()

    with patch.object(
        litellm.main.base_llm_http_handler,
        "text_to_speech_handler",
        return_value=fake_audio,
    ) as mock_handler:
        result = litellm.speech(
            model="dashscope/qwen3-tts-vc",
            input="hello world",
            voice="my-cloned-voice",
            api_key="sk-test",
        )

    assert result is fake_audio
    mock_handler.assert_called_once()
    kwargs = mock_handler.call_args.kwargs
    assert kwargs["custom_llm_provider"] == "dashscope"
    assert kwargs["model"] == "qwen3-tts-vc"
    assert kwargs["voice"] == "my-cloned-voice"
    assert isinstance(kwargs["text_to_speech_provider_config"], DashScopeTextToSpeechConfig)
    assert kwargs["litellm_params"]["api_key"] == "sk-test"
