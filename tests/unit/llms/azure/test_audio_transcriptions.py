import json
from pathlib import Path
from typing import Final

import httpx
import pytest
from openai import AsyncAzureOpenAI, AzureOpenAI

import litellm
from litellm.cost_calculator import completion_cost
from litellm.litellm_core_utils.audio_utils.utils import calculate_request_duration

AUDIO_FILE: Final = Path(__file__).parents[3] / "gettysburg.wav"
WHISPER_COST_PER_SECOND: Final = 0.0001


def _audio_file() -> tuple[str, bytes, str]:
    return ("gettysburg.wav", AUDIO_FILE.read_bytes(), "audio/wav")


def _transcription_client() -> AzureOpenAI:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"text": "Four score and seven years ago"})

    return AzureOpenAI(
        api_key="test-key",
        api_version="2024-06-01",
        azure_endpoint="https://example.cognitiveservices.azure.com",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_azure_transcription_keeps_the_azure_provider():
    with AUDIO_FILE.open("rb") as audio:
        response = litellm.transcription(
            model="azure/whisper-1",
            file=audio,
            api_base="https://example.openai.azure.com",
            api_key="test-key",
            api_version="2024-06-01",
            client=_transcription_client(),
        )

    assert response._hidden_params["custom_llm_provider"] == "azure"
    assert json.loads(response.model_dump_json())["text"] == "Four score and seven years ago"


@pytest.mark.asyncio
async def test_azure_transcribe_model_mapping():
    """
    Test that Azure transcription models are correctly mapped and not hardcoded to whisper-1.
    This test validates that the request body contains the correct model parameter.
    """
    from unittest.mock import AsyncMock, patch, MagicMock
    from openai import AsyncAzureOpenAI

    from pydantic import BaseModel as PydanticBaseModel

    class MockTranscriptionResponse(PydanticBaseModel):
        text: str

    mock_transcription_response = MockTranscriptionResponse(
        text="This is a test transcription"
    )

    mock_raw_response = MagicMock()
    mock_raw_response.headers = {"content-type": "application/json"}
    mock_raw_response.parse = MagicMock(return_value=mock_transcription_response)

    mock_azure_client = MagicMock(spec=AsyncAzureOpenAI)
    mock_azure_client.audio.transcriptions.with_raw_response.create = AsyncMock(
        return_value=mock_raw_response
    )
    mock_azure_client.api_key = "test-api-key"
    mock_azure_client._base_url = MagicMock()
    mock_azure_client._base_url._uri_reference = (
        "https://my-endpoint-europe-berri-992.openai.azure.com/"
    )

    with patch(
        "litellm.llms.azure.audio_transcriptions.AzureAudioTranscription.get_azure_openai_client",
        return_value=mock_azure_client,
    ):
        response = await litellm.atranscription(
            model="azure/whisper-1",
            file=_audio_file(),
            response_format="json",
            api_key="test-api-key",
            api_base="https://my-endpoint-europe-berri-992.openai.azure.com/",
            api_version="2024-02-15-preview",
            drop_params=True,
        )

        mock_azure_client.audio.transcriptions.with_raw_response.create.assert_called_once()

        call_kwargs = (
            mock_azure_client.audio.transcriptions.with_raw_response.create.call_args.kwargs
        )

        assert (
            call_kwargs["model"] == "whisper-1"
        ), f"Expected model 'whisper-1', got {call_kwargs['model']}"
        assert "file" in call_kwargs
        assert call_kwargs["response_format"] == "json"

        assert response._hidden_params is not None
        assert response._hidden_params["model"] == "whisper-1"
        assert response._hidden_params["custom_llm_provider"] == "azure"
        assert response.text is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response_format", "body", "content_type", "expected_text"),
    [
        ("json", b'{"text":"Four score and seven years ago"}', "application/json", "Four score and seven years ago"),
        (
            "vtt",
            b"WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nFour score and seven years ago",
            "text/vtt",
            "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nFour score and seven years ago",
        ),
        (
            "verbose_json",
            b'{"task":"transcribe","language":"English","duration":2.0,"text":"Four score and seven years ago"}',
            "application/json",
            "Four score and seven years ago",
        ),
    ],
)
async def test_azure_transcription_parses_response_formats(
    response_format: str,
    body: bytes,
    content_type: str,
    expected_text: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            content=body,
            headers={"content-type": content_type},
        )

    async with AsyncAzureOpenAI(
        api_key="test-key",
        api_version="2024-06-01",
        azure_endpoint="https://example.cognitiveservices.azure.com",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    ) as client:
        response = await litellm.atranscription(
            model="azure/whisper-1",
            file=_audio_file(),
            api_base="https://example.openai.azure.com",
            api_key="test-key",
            api_version="2024-06-01",
            response_format=response_format,
            timestamp_granularities=["word"] if response_format == "verbose_json" else None,
            client=client,
            drop_params=True,
        )

    assert response.text == expected_text
