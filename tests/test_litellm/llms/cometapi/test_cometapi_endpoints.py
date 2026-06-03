import io
import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

import litellm
from litellm.images.main import image_generation
from litellm.llms.cometapi.embed.transformation import CometAPIEmbeddingConfig
from litellm.llms.cometapi.image_generation.transformation import (
    CometAPIImageGenerationConfig,
)
from litellm.main import amoderation, moderation, speech, transcription
from litellm.types.utils import ImageResponse, TranscriptionResponse


def _clear_cometapi_env(monkeypatch):
    monkeypatch.delenv("COMETAPI_KEY", raising=False)
    monkeypatch.delenv("COMETAPI_API_KEY", raising=False)
    monkeypatch.delenv("COMETAPI_BASE_URL", raising=False)
    monkeypatch.delenv("COMETAPI_API_BASE", raising=False)
    monkeypatch.setattr(litellm, "cometapi_key", None, raising=False)


def _pollute_openai_globals(monkeypatch):
    monkeypatch.setattr(litellm, "api_key", "openai-global-key", raising=False)
    monkeypatch.setattr(litellm, "openai_key", "openai-provider-key", raising=False)
    monkeypatch.setattr(litellm, "api_base", "https://openai.invalid/v1", raising=False)


@pytest.mark.parametrize(
    ("api_base", "expected"),
    [
        (None, "https://api.cometapi.com/v1/embeddings"),
        ("https://api.cometapi.com", "https://api.cometapi.com/v1/embeddings"),
        ("https://api.cometapi.com/v1", "https://api.cometapi.com/v1/embeddings"),
        (
            "https://api.cometapi.com/v1/embeddings",
            "https://api.cometapi.com/v1/embeddings",
        ),
    ],
)
def test_cometapi_embedding_url_normalization(api_base, expected, monkeypatch):
    _clear_cometapi_env(monkeypatch)

    assert (
        CometAPIEmbeddingConfig().get_complete_url(
            api_base=api_base,
            api_key=None,
            model="text-embedding-3-small",
            optional_params={},
            litellm_params={},
        )
        == expected
    )


def test_cometapi_embedding_uses_api_key_alias(monkeypatch):
    _clear_cometapi_env(monkeypatch)
    monkeypatch.setenv("COMETAPI_API_KEY", "comet-env-key")

    headers = CometAPIEmbeddingConfig().validate_environment(
        headers={},
        model="text-embedding-3-small",
        messages=[],
        optional_params={},
        litellm_params={},
    )

    assert headers["Authorization"] == "Bearer comet-env-key"


def test_cometapi_embedding_missing_key_fails(monkeypatch):
    _clear_cometapi_env(monkeypatch)

    with pytest.raises(ValueError, match="COMETAPI_KEY or COMETAPI_API_KEY"):
        CometAPIEmbeddingConfig().validate_environment(
            headers={},
            model="text-embedding-3-small",
            messages=[],
            optional_params={},
            litellm_params={},
        )


def test_cometapi_image_generation_url_normalization():
    config = CometAPIImageGenerationConfig()

    assert (
        config.get_complete_url(
            api_base="https://api.cometapi.com/v1",
            api_key=None,
            model="gpt-image-1",
            optional_params={},
            litellm_params={},
        )
        == "https://api.cometapi.com/v1/images/generations"
    )
    assert (
        config.get_complete_url(
            api_base="https://api.cometapi.com/v1/images/generations",
            api_key=None,
            model="gpt-image-1",
            optional_params={},
            litellm_params={},
        )
        == "https://api.cometapi.com/v1/images/generations"
    )


def test_cometapi_image_generation_maps_new_openai_params(monkeypatch):
    _clear_cometapi_env(monkeypatch)
    _pollute_openai_globals(monkeypatch)

    with patch(
        "litellm.images.main.llm_http_handler.image_generation_handler",
        return_value=ImageResponse(data=[{"url": "https://example.com/image.png"}]),
    ) as mock_image_handler:
        image_generation(
            model="cometapi/gpt-image-1",
            prompt="A small comet over a clean API diagram",
            api_key="comet-explicit-key",
            background="transparent",
            output_compression=70,
            output_format="png",
            size="1024x1024",
        )

    call_kwargs = mock_image_handler.call_args.kwargs
    assert call_kwargs["api_key"] == "comet-explicit-key"
    assert call_kwargs["custom_llm_provider"] == "cometapi"
    assert call_kwargs["litellm_params"]["api_base"] is None
    assert (
        call_kwargs["image_generation_optional_request_params"]["background"]
        == "transparent"
    )
    assert (
        call_kwargs["image_generation_optional_request_params"]["output_compression"]
        == 70
    )
    assert (
        call_kwargs["image_generation_optional_request_params"]["output_format"]
        == "png"
    )


def test_cometapi_speech_uses_cometapi_key_and_base(monkeypatch):
    _clear_cometapi_env(monkeypatch)
    _pollute_openai_globals(monkeypatch)
    monkeypatch.setenv("COMETAPI_API_KEY", "comet-env-key")

    with patch("litellm.main.openai_chat_completions.audio_speech") as mock_speech:
        mock_speech.return_value = b"audio"
        speech(model="cometapi/tts-1", input="hello", voice="alloy")

    call_kwargs = mock_speech.call_args.kwargs
    assert call_kwargs["model"] == "tts-1"
    assert call_kwargs["api_key"] == "comet-env-key"
    assert call_kwargs["api_base"] == "https://api.cometapi.com/v1"


def test_cometapi_transcription_uses_cometapi_key_and_base(monkeypatch):
    _clear_cometapi_env(monkeypatch)
    _pollute_openai_globals(monkeypatch)

    audio_file = io.BytesIO(b"not-real-audio")
    audio_file.name = "sample.wav"

    with patch(
        "litellm.main.openai_audio_transcriptions.audio_transcriptions",
        return_value=TranscriptionResponse(text="hello"),
    ) as mock_transcription:
        transcription(
            model="cometapi/whisper-1",
            file=audio_file,
            api_key="comet-explicit-key",
        )

    call_kwargs = mock_transcription.call_args.kwargs
    assert call_kwargs["model"] == "whisper-1"
    assert call_kwargs["api_key"] == "comet-explicit-key"
    assert call_kwargs["api_base"] == "https://api.cometapi.com/v1"


class _ModerationResponse:
    def model_dump(self):
        return {
            "id": "modr-test",
            "model": "omni-moderation-latest",
            "results": [
                {
                    "flagged": False,
                    "categories": {},
                    "category_scores": {},
                    "category_applied_input_types": {},
                }
            ],
        }


class _Moderations:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _ModerationResponse()


class _OpenAIClient:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.moderations = _Moderations()
        self.instances.append(self)


def test_cometapi_moderation_uses_cometapi_key_and_base(monkeypatch):
    _clear_cometapi_env(monkeypatch)
    _pollute_openai_globals(monkeypatch)

    _OpenAIClient.instances = []
    with patch("litellm.main.openai.OpenAI", _OpenAIClient):
        moderation(
            model="cometapi/omni-moderation-latest",
            input="hello",
            api_key="comet-explicit-key",
        )

    assert _OpenAIClient.instances[0].kwargs == {
        "api_key": "comet-explicit-key",
        "base_url": "https://api.cometapi.com/v1",
    }
    assert _OpenAIClient.instances[0].moderations.calls[0]["model"] == (
        "omni-moderation-latest"
    )


class _AsyncModerations:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return _ModerationResponse()


class _AsyncOpenAIClient:
    def __init__(self):
        self.moderations = _AsyncModerations()


@pytest.mark.asyncio
async def test_cometapi_amoderation_uses_cometapi_key_and_base(monkeypatch):
    _clear_cometapi_env(monkeypatch)
    _pollute_openai_globals(monkeypatch)

    fake_client = _AsyncOpenAIClient()
    with patch(
        "litellm.main.openai_chat_completions._get_openai_client",
        return_value=fake_client,
    ) as mock_get_client:
        await amoderation(
            model="cometapi/omni-moderation-latest",
            input="hello",
            api_key="comet-explicit-key",
        )

    assert mock_get_client.call_args.kwargs["api_key"] == "comet-explicit-key"
    assert mock_get_client.call_args.kwargs["api_base"] == "https://api.cometapi.com/v1"
    assert fake_client.moderations.calls[0]["model"] == "omni-moderation-latest"
