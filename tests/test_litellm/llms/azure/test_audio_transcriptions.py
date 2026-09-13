import json
from pathlib import Path
from typing import Final

import httpx
import pytest
from openai import AzureOpenAI

import litellm
from litellm.cost_calculator import completion_cost
from litellm.litellm_core_utils.audio_utils.utils import calculate_request_duration

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


def test_azure_ai_transcription_is_priced_at_the_azure_ai_entry():
    with AUDIO_FILE.open("rb") as audio:
        response = litellm.transcription(
            model="azure_ai/whisper",
            file=audio,
            api_base="https://example.cognitiveservices.azure.com",
            api_key="test-key",
            api_version="2024-06-01",
            client=_transcription_client(),
        )
    with AUDIO_FILE.open("rb") as audio:
        duration = calculate_request_duration(audio)

    assert duration is not None and duration > 0
    assert response._hidden_params["custom_llm_provider"] == "azure_ai"
    assert completion_cost(completion_response=response, call_type="transcription") == pytest.approx(
        WHISPER_COST_PER_SECOND * duration
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
