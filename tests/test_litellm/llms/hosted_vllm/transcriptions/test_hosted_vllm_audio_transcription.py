import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../../../.."))

import litellm
from litellm.llms.hosted_vllm.transcriptions.transformation import (
    HostedVLLMAudioTranscriptionConfig,
)
from litellm.types.utils import ImageResponse, TranscriptionResponse


def _complete_url(api_base: str | None) -> str:
    return HostedVLLMAudioTranscriptionConfig().get_complete_url(
        api_base=api_base,
        api_key=None,
        model="whisper-1",
        optional_params={},
        litellm_params={},
    )


class TestHostedVLLMTranscriptionUrl:
    @pytest.mark.parametrize(
        "api_base,expected",
        [
            (
                "http://vllm.example.com",
                "http://vllm.example.com/v1/audio/transcriptions",
            ),
            (
                "http://vllm.example.com/",
                "http://vllm.example.com/v1/audio/transcriptions",
            ),
            (
                "http://vllm.example.com/v1",
                "http://vllm.example.com/v1/audio/transcriptions",
            ),
            (
                "http://vllm.example.com/v1/",
                "http://vllm.example.com/v1/audio/transcriptions",
            ),
            (
                "http://proxy.example.com/qwen3-asr/v1",
                "http://proxy.example.com/qwen3-asr/v1/audio/transcriptions",
            ),
            (
                "http://vllm.example.com/v1/audio/transcriptions",
                "http://vllm.example.com/v1/audio/transcriptions",
            ),
        ],
    )
    def test_get_complete_url(self, api_base: str, expected: str) -> None:
        assert _complete_url(api_base) == expected

    def test_get_complete_url_requires_api_base(self) -> None:
        with pytest.raises(ValueError, match="api_base must be provided"):
            _complete_url(None)


class TestAtranscriptionCustomLlmProvider:
    @pytest.mark.asyncio
    async def test_unprefixed_model_uses_custom_llm_provider(self) -> None:
        """
        Proxy/router deployments often register model=qwen3-asr-0.6b with
        custom_llm_provider=hosted_vllm (no hosted_vllm/ prefix). Chat honors
        that field; atranscription used to ignore it and raise
        'LLM Provider NOT provided'.
        """
        with patch(
            "litellm.main.transcription",
            return_value=TranscriptionResponse(text="hello"),
        ) as mock_transcription:
            response = await litellm.atranscription(
                model="qwen3-asr-0.6b",
                file=b"fake-audio",
                custom_llm_provider="hosted_vllm",
                api_base="http://vllm.example.com/v1",
            )

        assert response.text == "hello"
        mock_transcription.assert_called()

    @pytest.mark.asyncio
    async def test_unprefixed_model_without_provider_still_fails(self) -> None:
        with pytest.raises(Exception, match="LLM Provider NOT provided"):
            await litellm.atranscription(
                model="qwen3-asr-0.6b",
                file=b"fake-audio",
                api_base="http://vllm.example.com/v1",
            )

    @pytest.mark.asyncio
    async def test_aspeech_unprefixed_model_uses_custom_llm_provider(self) -> None:
        with patch("litellm.main.speech", return_value=MagicMock()) as mock_speech:
            await litellm.aspeech(
                model="tts-model",
                input="hello",
                voice="alloy",
                custom_llm_provider="hosted_vllm",
                api_base="http://vllm.example.com/v1",
            )

        mock_speech.assert_called()


class TestAimageGenerationCustomLlmProvider:
    @pytest.mark.asyncio
    async def test_unprefixed_model_uses_custom_llm_provider(self) -> None:
        mock_response = ImageResponse(
            created=1234567890,
            data=[{"url": "https://example.com/image.png"}],
        )
        with patch(
            "litellm.images.main.image_generation",
            return_value=mock_response,
        ) as mock_image_generation:
            response = await litellm.aimage_generation(
                model="unprefixed-image-model",
                prompt="a cat",
                custom_llm_provider="openai",
                api_base="http://vllm.example.com/v1",
            )

        assert response.data is not None
        mock_image_generation.assert_called()

    @pytest.mark.asyncio
    async def test_unprefixed_model_without_provider_still_fails(self) -> None:
        with pytest.raises(Exception, match="LLM Provider NOT provided"):
            await litellm.aimage_generation(
                model="unprefixed-image-model",
                prompt="a cat",
                api_base="http://vllm.example.com/v1",
            )
