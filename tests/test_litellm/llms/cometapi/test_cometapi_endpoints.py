import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

import litellm
from litellm import completion
from litellm.images.main import image_generation
from litellm.llms.cometapi.common_utils import (
    CometAPIException,
    get_cometapi_api_base,
    get_cometapi_api_key,
    get_cometapi_complete_url,
)
from litellm.llms.cometapi.embed.transformation import CometAPIEmbeddingConfig
from litellm.llms.cometapi.image_generation.transformation import (
    CometAPIImageGenerationConfig,
)
from litellm.main import amoderation, embedding, moderation, speech, transcription
from litellm.types.utils import EmbeddingResponse, ImageResponse, TranscriptionResponse


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
        (
            "https://proxy.example.com/openai/v1",
            "https://proxy.example.com/openai/v1/embeddings",
        ),
        (
            "https://proxy.example.com/vertex/openai/v1",
            "https://proxy.example.com/vertex/openai/v1/embeddings",
        ),
        (
            "https://proxy.example.com/api/v2/openai/v1",
            "https://proxy.example.com/api/v2/openai/v1/embeddings",
        ),
        (
            "https://proxy.example.com/openai/v1/embeddings",
            "https://proxy.example.com/openai/v1/embeddings",
        ),
    ],
)
def test_cometapi_embedding_url_normalization(api_base, expected, monkeypatch):
    _clear_cometapi_env(monkeypatch)
    request_api_key = "comet-explicit-key" if api_base else None

    assert (
        CometAPIEmbeddingConfig().get_complete_url(
            api_base=api_base,
            api_key=request_api_key,
            model="text-embedding-3-small",
            optional_params={},
            litellm_params={},
        )
        == expected
    )


def test_cometapi_key_and_base_precedence(monkeypatch):
    _clear_cometapi_env(monkeypatch)
    _pollute_openai_globals(monkeypatch)
    monkeypatch.setenv("COMETAPI_KEY", "comet-env-key")
    monkeypatch.setenv("COMETAPI_BASE_URL", "https://proxy.example.com/openai/v1")

    assert get_cometapi_api_key("explicit-key") == "explicit-key"
    assert get_cometapi_api_key() == "comet-env-key"
    assert (
        get_cometapi_api_base("https://explicit.example.com/v1", api_key="explicit-key")
        == "https://explicit.example.com/v1"
    )
    assert get_cometapi_api_base() == "https://proxy.example.com/openai/v1"


def test_cometapi_custom_base_requires_request_key(monkeypatch):
    _clear_cometapi_env(monkeypatch)
    _pollute_openai_globals(monkeypatch)
    monkeypatch.setenv("COMETAPI_KEY", "comet-env-key")

    with pytest.raises(ValueError, match="api_base requires an explicit api_key"):
        get_cometapi_api_base("https://attacker.example.com/v1")


def test_cometapi_key_and_base_ignore_litellm_globals(monkeypatch):
    _clear_cometapi_env(monkeypatch)
    _pollute_openai_globals(monkeypatch)

    assert get_cometapi_api_key() is None
    assert get_cometapi_api_base() == "https://api.cometapi.com/v1"


def test_cometapi_complete_url_preserves_existing_v1_path(monkeypatch):
    _clear_cometapi_env(monkeypatch)

    assert (
        get_cometapi_complete_url(
            "https://proxy.example.com/openai/v1",
            "embeddings",
            api_key="comet-explicit-key",
        )
        == "https://proxy.example.com/openai/v1/embeddings"
    )


@pytest.mark.parametrize(
    "api_base",
    [
        "https://proxy.example.com/openai/v1beta",
        "https://proxy.example.com/openai/v1beta/embeddings",
        "https://proxy.example.com/openai/v10",
        "https://proxy.example.com/openai/embeddings",
    ],
)
def test_cometapi_complete_url_rejects_non_v1_paths(api_base, monkeypatch):
    _clear_cometapi_env(monkeypatch)

    with pytest.raises(
        ValueError,
        match="CometAPI OpenAI-compatible endpoints require a /v1 api_base",
    ):
        get_cometapi_complete_url(api_base, "embeddings", api_key="comet-explicit-key")


@pytest.mark.parametrize(
    "endpoint",
    [
        "",
        "/",
        "embeddings?foo=bar",
        "embeddings#frag",
        "http://evil.test/x",
        "../embeddings",
    ],
)
def test_cometapi_complete_url_rejects_invalid_endpoints(endpoint, monkeypatch):
    _clear_cometapi_env(monkeypatch)

    with pytest.raises(ValueError, match="CometAPI endpoint must be a non-empty path"):
        get_cometapi_complete_url(
            "https://api.cometapi.com/v1",
            endpoint,
            api_key="comet-explicit-key",
        )


@pytest.mark.parametrize(
    "api_base",
    [
        "https://api.cometapi.com/v1?tenant=test",
        "https://api.cometapi.com/v1#fragment",
    ],
)
def test_cometapi_complete_url_rejects_api_base_query_or_fragment(
    api_base, monkeypatch
):
    _clear_cometapi_env(monkeypatch)

    with pytest.raises(
        ValueError, match="CometAPI api_base must not include query or fragment"
    ):
        get_cometapi_complete_url(api_base, "embeddings", api_key="comet-explicit-key")


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


def test_cometapi_embedding_main_uses_cometapi_key_and_base(monkeypatch):
    _clear_cometapi_env(monkeypatch)
    _pollute_openai_globals(monkeypatch)
    monkeypatch.setenv("COMETAPI_API_KEY", "comet-env-key")
    monkeypatch.setenv("COMETAPI_BASE_URL", "https://proxy.example.com/openai/v1")

    with patch(
        "litellm.main.base_llm_http_handler.embedding",
        return_value=EmbeddingResponse(),
    ) as mock_embedding:
        embedding(model="cometapi/text-embedding-3-small", input=["hello"])

    call_kwargs = mock_embedding.call_args.kwargs
    assert call_kwargs["model"] == "text-embedding-3-small"
    assert call_kwargs["api_key"] == "comet-env-key"
    assert call_kwargs["api_base"] == "https://proxy.example.com/openai/v1"


def test_cometapi_completion_uses_cometapi_key_and_base(monkeypatch):
    _clear_cometapi_env(monkeypatch)
    _pollute_openai_globals(monkeypatch)
    monkeypatch.setenv("COMETAPI_API_KEY", "comet-env-key")
    monkeypatch.setenv("COMETAPI_BASE_URL", "https://proxy.example.com/openai/v1")
    mock_response = MagicMock()

    with patch(
        "litellm.main.base_llm_http_handler.completion",
        return_value=mock_response,
    ) as mock_completion:
        response = completion(
            model="cometapi/gpt-5.5",
            messages=[{"role": "user", "content": "hello"}],
        )

    call_kwargs = mock_completion.call_args.kwargs
    assert response is mock_response
    assert call_kwargs["model"] == "gpt-5.5"
    assert call_kwargs["api_key"] == "comet-env-key"
    assert call_kwargs["api_base"] == "https://proxy.example.com/openai/v1"
    assert call_kwargs["custom_llm_provider"] == "cometapi"


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
            api_key="comet-explicit-key",
            model="gpt-image-2",
            optional_params={},
            litellm_params={},
        )
        == "https://api.cometapi.com/v1/images/generations"
    )
    assert (
        config.get_complete_url(
            api_base="https://api.cometapi.com/v1/images/generations",
            api_key="comet-explicit-key",
            model="gpt-image-2",
            optional_params={},
            litellm_params={},
        )
        == "https://api.cometapi.com/v1/images/generations"
    )


def test_cometapi_image_generation_validate_environment_uses_api_key_alias(monkeypatch):
    _clear_cometapi_env(monkeypatch)
    monkeypatch.setenv("COMETAPI_API_KEY", "comet-env-key")

    headers = CometAPIImageGenerationConfig().validate_environment(
        headers={},
        model="gpt-image-2",
        messages=[],
        optional_params={},
        litellm_params={},
    )

    assert headers["Authorization"] == "Bearer comet-env-key"
    assert headers["Content-Type"] == "application/json"


def test_cometapi_image_generation_missing_key_fails_closed(monkeypatch):
    _clear_cometapi_env(monkeypatch)
    _pollute_openai_globals(monkeypatch)

    with pytest.raises(ValueError, match="COMETAPI_KEY or COMETAPI_API_KEY"):
        CometAPIImageGenerationConfig().validate_environment(
            headers={},
            model="gpt-image-2",
            messages=[],
            optional_params={},
            litellm_params={},
        )


def test_cometapi_image_generation_missing_key_does_not_call_handler(monkeypatch):
    _clear_cometapi_env(monkeypatch)
    _pollute_openai_globals(monkeypatch)

    with patch(
        "litellm.images.main.llm_http_handler.image_generation_handler"
    ) as mock_image_handler:
        with pytest.raises(
            litellm.APIConnectionError, match="COMETAPI_KEY or COMETAPI_API_KEY"
        ):
            image_generation(
                model="cometapi/gpt-image-2",
                prompt="A small comet over a clean API diagram",
            )

    mock_image_handler.assert_not_called()


def test_cometapi_image_generation_custom_base_requires_request_key(monkeypatch):
    _clear_cometapi_env(monkeypatch)
    _pollute_openai_globals(monkeypatch)
    monkeypatch.setenv("COMETAPI_KEY", "comet-env-key")

    with patch(
        "litellm.images.main.llm_http_handler.image_generation_handler"
    ) as mock_image_handler:
        with pytest.raises(
            litellm.APIConnectionError,
            match="api_base requires an explicit api_key",
        ):
            image_generation(
                model="cometapi/gpt-image-2",
                prompt="A small comet over a clean API diagram",
                api_base="https://attacker.example.com/v1",
            )

    mock_image_handler.assert_not_called()


def test_cometapi_image_generation_maps_new_openai_params(monkeypatch):
    _clear_cometapi_env(monkeypatch)
    _pollute_openai_globals(monkeypatch)

    with patch(
        "litellm.images.main.llm_http_handler.image_generation_handler",
        return_value=ImageResponse(data=[{"url": "https://example.com/image.png"}]),
    ) as mock_image_handler:
        image_generation(
            model="cometapi/gpt-image-2",
            prompt="A small comet over a clean API diagram",
            api_base="https://proxy.example.com/openai/v1",
            api_key="comet-explicit-key",
            output_compression=70,
            output_format="png",
            size="1024x1024",
        )

    call_kwargs = mock_image_handler.call_args.kwargs
    assert call_kwargs["api_key"] == "comet-explicit-key"
    assert call_kwargs["custom_llm_provider"] == "cometapi"
    assert call_kwargs["litellm_params"]["api_base"] == (
        "https://proxy.example.com/openai/v1"
    )
    assert "api_key" not in call_kwargs["litellm_params"]
    assert (
        call_kwargs["image_generation_optional_request_params"]["output_compression"]
        == 70
    )
    assert (
        call_kwargs["image_generation_optional_request_params"]["output_format"]
        == "png"
    )


def test_cometapi_image_generation_normalizes_null_usage_fields():
    raw_response = httpx.Response(
        200,
        json={
            "created": 123,
            "data": [{"url": "https://example.com/image.png"}],
            "usage": {
                "input_tokens": 12,
                "input_tokens_details": {
                    "image_tokens": None,
                    "text_tokens": None,
                },
                "output_tokens": 100,
                "total_tokens": 112,
            },
        },
        request=httpx.Request("POST", "https://api.cometapi.com/v1/images/generations"),
    )

    response = CometAPIImageGenerationConfig().transform_image_generation_response(
        model="gpt-image-2",
        raw_response=raw_response,
        model_response=ImageResponse(),
        logging_obj=MagicMock(),
        request_data={"prompt": "A small comet"},
        optional_params={
            "output_format": "png",
            "quality": "high",
            "size": "1024x1024",
        },
        litellm_params={},
        encoding=None,
    )

    assert response.data[0].url == "https://example.com/image.png"
    assert response.usage.input_tokens_details.image_tokens == 0
    assert response.usage.input_tokens_details.text_tokens == 0
    assert response.usage.input_tokens == 12
    assert response.usage.output_tokens == 100
    assert response.usage.total_tokens == 112
    assert response.output_format == "png"
    assert response.quality == "high"
    assert response.size == "1024x1024"


def test_cometapi_image_generation_handles_missing_usage():
    raw_response = httpx.Response(
        200,
        json={
            "created": 123,
            "data": [{"url": "https://example.com/image.png"}],
        },
        request=httpx.Request("POST", "https://api.cometapi.com/v1/images/generations"),
    )

    response = CometAPIImageGenerationConfig().transform_image_generation_response(
        model="gpt-image-2",
        raw_response=raw_response,
        model_response=ImageResponse(),
        logging_obj=MagicMock(),
        request_data={"prompt": "A small comet"},
        optional_params={},
        litellm_params={},
        encoding=None,
    )

    assert response.data[0].url == "https://example.com/image.png"
    assert response.usage.input_tokens == 0
    assert response.usage.input_tokens_details.image_tokens == 0
    assert response.usage.input_tokens_details.text_tokens == 0
    assert response.usage.output_tokens == 0
    assert response.usage.total_tokens == 0


def test_cometapi_image_generation_normalizes_null_usage_totals():
    raw_response = httpx.Response(
        200,
        json={
            "created": 123,
            "data": [{"url": "https://example.com/image.png"}],
            "usage": {
                "input_tokens": None,
                "input_tokens_details": None,
                "output_tokens": None,
                "total_tokens": None,
            },
        },
        request=httpx.Request("POST", "https://api.cometapi.com/v1/images/generations"),
    )

    response = CometAPIImageGenerationConfig().transform_image_generation_response(
        model="gpt-image-2",
        raw_response=raw_response,
        model_response=ImageResponse(),
        logging_obj=MagicMock(),
        request_data={"prompt": "A small comet"},
        optional_params={},
        litellm_params={},
        encoding=None,
    )

    assert response.usage.input_tokens == 0
    assert response.usage.input_tokens_details.image_tokens == 0
    assert response.usage.input_tokens_details.text_tokens == 0
    assert response.usage.output_tokens == 0
    assert response.usage.total_tokens == 0


def test_cometapi_image_generation_raises_provider_error_on_error_response():
    raw_response = httpx.Response(
        500,
        json={
            "error": {
                "message": "Transparent background is not supported for this model.",
                "type": "image_generation_user_error",
                "param": "background",
                "code": "invalid_value",
            }
        },
        request=httpx.Request("POST", "https://api.cometapi.com/v1/images/generations"),
    )

    with pytest.raises(
        CometAPIException,
        match="Transparent background is not supported for this model.",
    ):
        CometAPIImageGenerationConfig().transform_image_generation_response(
            model="gpt-image-2",
            raw_response=raw_response,
            model_response=ImageResponse(),
            logging_obj=MagicMock(),
            request_data={"prompt": "A small comet"},
            optional_params={},
            litellm_params={},
            encoding=None,
        )


def test_cometapi_speech_uses_cometapi_key_and_base(monkeypatch):
    _clear_cometapi_env(monkeypatch)
    _pollute_openai_globals(monkeypatch)
    monkeypatch.setenv("COMETAPI_API_KEY", "comet-env-key")
    monkeypatch.setenv("COMETAPI_BASE_URL", "https://api.cometapi.com/v1")

    with patch("litellm.main.openai_chat_completions.audio_speech") as mock_speech:
        mock_speech.return_value = b"audio"
        speech(model="cometapi/tts-1", input="hello", voice="alloy")

    call_kwargs = mock_speech.call_args.kwargs
    assert call_kwargs["model"] == "tts-1"
    assert call_kwargs["api_key"] == "comet-env-key"
    assert call_kwargs["api_base"] == "https://api.cometapi.com/v1"


def test_cometapi_speech_uses_dynamic_api_key(monkeypatch):
    _clear_cometapi_env(monkeypatch)

    with (
        patch(
            "litellm.main.get_llm_provider",
            return_value=("tts-1", "cometapi", "dynamic-comet-key", None),
        ),
        patch("litellm.main.openai_chat_completions.audio_speech") as mock_speech,
    ):
        mock_speech.return_value = b"audio"
        speech(model="cometapi/tts-1", input="hello", voice="alloy")

    assert mock_speech.call_args.kwargs["api_key"] == "dynamic-comet-key"


def test_cometapi_speech_missing_key_does_not_call_openai_audio(monkeypatch):
    _clear_cometapi_env(monkeypatch)
    _pollute_openai_globals(monkeypatch)

    with patch("litellm.main.openai_chat_completions.audio_speech") as mock_speech:
        with pytest.raises(ValueError, match="COMETAPI_KEY or COMETAPI_API_KEY"):
            speech(model="cometapi/tts-1", input="hello", voice="alloy")

    mock_speech.assert_not_called()


def test_cometapi_speech_custom_base_requires_request_key(monkeypatch):
    _clear_cometapi_env(monkeypatch)
    _pollute_openai_globals(monkeypatch)
    monkeypatch.setenv("COMETAPI_KEY", "comet-env-key")

    with patch("litellm.main.openai_chat_completions.audio_speech") as mock_speech:
        with pytest.raises(ValueError, match="api_base requires an explicit api_key"):
            speech(
                model="cometapi/tts-1",
                input="hello",
                voice="alloy",
                api_base="https://attacker.example.com/v1",
            )

    mock_speech.assert_not_called()


def test_cometapi_speech_requires_voice(monkeypatch):
    _clear_cometapi_env(monkeypatch)

    with pytest.raises(litellm.BadRequestError, match="'voice' is required"):
        speech(model="cometapi/tts-1", input="hello", api_key="comet-explicit-key")


def test_cometapi_transcription_uses_cometapi_key_and_base(monkeypatch):
    _clear_cometapi_env(monkeypatch)
    _pollute_openai_globals(monkeypatch)
    monkeypatch.setenv("COMETAPI_BASE_URL", "https://api.cometapi.com/v1")

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


def test_cometapi_transcription_custom_base_requires_request_key(monkeypatch):
    _clear_cometapi_env(monkeypatch)
    _pollute_openai_globals(monkeypatch)
    monkeypatch.setenv("COMETAPI_KEY", "comet-env-key")

    audio_file = io.BytesIO(b"not-real-audio")
    audio_file.name = "sample.wav"

    with patch(
        "litellm.main.openai_audio_transcriptions.audio_transcriptions"
    ) as mock_transcription:
        with pytest.raises(ValueError, match="api_base requires an explicit api_key"):
            transcription(
                model="cometapi/whisper-1",
                file=audio_file,
                api_base="https://attacker.example.com/v1",
            )

    mock_transcription.assert_not_called()


def test_openai_transcription_fallback_is_unchanged(monkeypatch):
    _clear_cometapi_env(monkeypatch)
    _pollute_openai_globals(monkeypatch)

    audio_file = io.BytesIO(b"not-real-audio")
    audio_file.name = "sample.wav"

    with patch(
        "litellm.main.openai_audio_transcriptions.audio_transcriptions",
        return_value=TranscriptionResponse(text="hello"),
    ) as mock_transcription:
        transcription(
            model="whisper-1",
            file=audio_file,
            api_key="openai-explicit-key",
        )

    call_kwargs = mock_transcription.call_args.kwargs
    assert call_kwargs["model"] == "whisper-1"
    assert call_kwargs["api_key"] == "openai-explicit-key"
    assert call_kwargs["api_base"] == "https://openai.invalid/v1"


def test_openai_speech_fallback_is_unchanged(monkeypatch):
    _clear_cometapi_env(monkeypatch)
    _pollute_openai_globals(monkeypatch)

    with patch("litellm.main.openai_chat_completions.audio_speech") as mock_speech:
        mock_speech.return_value = b"audio"
        speech(
            model="tts-1",
            input="hello",
            voice="alloy",
            api_key="openai-explicit-key",
        )

    call_kwargs = mock_speech.call_args.kwargs
    assert call_kwargs["model"] == "tts-1"
    assert call_kwargs["api_key"] == "openai-explicit-key"
    assert call_kwargs["api_base"] == "https://openai.invalid/v1"
    assert call_kwargs["organization"] is None
    assert call_kwargs["project"] is None


def test_provider_endpoint_matrix_only_updates_cometapi():
    support_path = Path(__file__).parents[4] / "provider_endpoints_support.json"
    providers = json.loads(support_path.read_text())["providers"]

    for endpoint in (
        "image_generations",
        "audio_transcriptions",
        "audio_speech",
        "moderations",
    ):
        assert providers["cometapi"]["endpoints"][endpoint] is True


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
    monkeypatch.setenv("COMETAPI_BASE_URL", "https://api.cometapi.com/v1")

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


def test_cometapi_moderation_accepts_bare_model_with_custom_provider(monkeypatch):
    _clear_cometapi_env(monkeypatch)
    _pollute_openai_globals(monkeypatch)
    monkeypatch.setenv("COMETAPI_BASE_URL", "https://api.cometapi.com/v1")

    _OpenAIClient.instances = []
    with patch("litellm.main.openai.OpenAI", _OpenAIClient):
        moderation(
            model="omni-moderation-latest",
            custom_llm_provider="cometapi",
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


def test_cometapi_moderation_missing_key_does_not_create_openai_client(monkeypatch):
    _clear_cometapi_env(monkeypatch)
    _pollute_openai_globals(monkeypatch)

    _OpenAIClient.instances = []
    with patch("litellm.main.openai.OpenAI", _OpenAIClient):
        with pytest.raises(ValueError, match="COMETAPI_KEY or COMETAPI_API_KEY"):
            moderation(model="cometapi/omni-moderation-latest", input="hello")

    assert _OpenAIClient.instances == []


def test_cometapi_moderation_custom_base_requires_request_key(monkeypatch):
    _clear_cometapi_env(monkeypatch)
    _pollute_openai_globals(monkeypatch)
    monkeypatch.setenv("COMETAPI_KEY", "comet-env-key")

    _OpenAIClient.instances = []
    with patch("litellm.main.openai.OpenAI", _OpenAIClient):
        with pytest.raises(ValueError, match="api_base requires an explicit api_key"):
            moderation(
                model="cometapi/omni-moderation-latest",
                input="hello",
                api_base="https://attacker.example.com/v1",
            )

    assert _OpenAIClient.instances == []


def test_openai_moderation_fallback_is_unchanged(monkeypatch):
    _clear_cometapi_env(monkeypatch)
    _pollute_openai_globals(monkeypatch)

    _OpenAIClient.instances = []
    with patch("litellm.main.openai.OpenAI", _OpenAIClient):
        moderation(model="omni-moderation-latest", input="hello")

    assert _OpenAIClient.instances[0].kwargs == {
        "api_key": "openai-global-key",
    }
    assert _OpenAIClient.instances[0].moderations.calls[0]["model"] == (
        "omni-moderation-latest"
    )


def test_openai_moderation_without_model_fallback_is_unchanged(monkeypatch):
    _clear_cometapi_env(monkeypatch)
    _pollute_openai_globals(monkeypatch)

    _OpenAIClient.instances = []
    with patch("litellm.main.openai.OpenAI", _OpenAIClient):
        moderation(input="hello")

    assert _OpenAIClient.instances[0].kwargs == {
        "api_key": "openai-global-key",
    }
    assert "model" not in _OpenAIClient.instances[0].moderations.calls[0]


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
    monkeypatch.setenv("COMETAPI_BASE_URL", "https://api.cometapi.com/v1")

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


@pytest.mark.asyncio
async def test_openai_amoderation_fallback_is_unchanged(monkeypatch):
    _clear_cometapi_env(monkeypatch)
    _pollute_openai_globals(monkeypatch)

    fake_client = _AsyncOpenAIClient()
    with patch(
        "litellm.main.openai_chat_completions._get_openai_client",
        return_value=fake_client,
    ) as mock_get_client:
        await amoderation(
            model="omni-moderation-latest",
            input="hello",
            api_key="openai-explicit-key",
        )

    assert mock_get_client.call_args.kwargs["api_key"] == "openai-explicit-key"
    assert mock_get_client.call_args.kwargs["api_base"] is None
    assert fake_client.moderations.calls[0]["model"] == "omni-moderation-latest"
