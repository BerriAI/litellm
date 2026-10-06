import io
import json
from pathlib import Path
from typing import Final
from unittest.mock import MagicMock

import httpx
import pytest
from openai import AzureOpenAI

import litellm
from litellm.cost_calculator import completion_cost
from litellm.litellm_core_utils.audio_utils.utils import calculate_request_duration
from litellm.llms.azure.audio_transcriptions import AzureAudioTranscription
from litellm.types.utils import TranscriptionResponse

AUDIO_FILE: Final = Path(__file__).parents[3] / "gettysburg.wav"
WHISPER_COST_PER_SECOND: Final = 0.0001


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


@pytest.mark.parametrize("api_version", ["v1", "latest", "preview"])
@pytest.mark.parametrize(
    ("model", "expected_path"),
    [
        ("whisper-1", "/openai/v1/audio/transcriptions"),
        ("gpt-transcribe", "/openai/deployments/gpt-transcribe/audio/transcriptions"),
        ("custom-transcribe-deployment", "/openai/deployments/custom-transcribe-deployment/audio/transcriptions"),
    ],
)
def test_azure_transcription_alias_uses_model_route(
    monkeypatch: pytest.MonkeyPatch, model: str, expected_path: str, api_version: str
) -> None:
    def send_response(request: httpx.Request) -> httpx.Response:
        assert request.url.path == expected_path
        assert request.url.params.get("api-version") == (
            None if expected_path.startswith("/openai/v1/") else litellm.AZURE_DEFAULT_API_VERSION
        )
        return httpx.Response(200, json={"text": "hello"})

    audio_file: Final = io.BytesIO(b"audio")
    audio_file.name = "sample.wav"
    with httpx.Client(transport=httpx.MockTransport(send_response)) as http_client:
        monkeypatch.setattr(litellm, "client_session", http_client)
        monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
        response: Final = AzureAudioTranscription().audio_transcriptions(
            model=model,
            audio_file=audio_file,
            optional_params={"response_format": "json"},
            logging_obj=MagicMock(),
            model_response=TranscriptionResponse(),
            timeout=10,
            max_retries=0,
            api_key="test-key",
            api_base="https://example.openai.azure.com",
            api_version=api_version,
        )

    assert response.text == "hello"
