from datetime import datetime, timezone
from typing import Final
from unittest.mock import Mock

import httpx
import pytest
from openai import AsyncOpenAI, OpenAI

import litellm
from litellm.litellm_core_utils.audio_utils.utils import get_audio_file_name
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.openai.transcriptions.handler import OpenAIAudioTranscription
from litellm.types.utils import TranscriptionResponse

_PROVIDER_HEADERS: Final = {"x-request-id": "req_stt", "x-ratelimit-remaining-requests": "41"}


def _transcription_transport() -> httpx.MockTransport:
    return httpx.MockTransport(lambda request: httpx.Response(200, json={"text": "hello"}, headers=_PROVIDER_HEADERS))


def _logging_obj() -> Mock:
    logging_obj = Mock()
    logging_obj.model_call_details = {}
    return logging_obj


def _call_kwargs(logging_obj: Mock) -> dict:
    return {
        "model": "gpt-4o-mini-transcribe",
        "audio_file": ("audio.wav", b"riff-bytes", "audio/wav"),
        "optional_params": {},
        "litellm_params": {},
        "model_response": TranscriptionResponse(),
        "timeout": 10.0,
        "max_retries": 0,
        "logging_obj": logging_obj,
        "api_key": "transport-only",
        "api_base": None,
    }


def _assert_headers_recorded(response: TranscriptionResponse, logging_obj: Mock) -> None:
    assert response.text == "hello"
    assert response._hidden_params["headers"]["x-request-id"] == "req_stt"
    assert response._hidden_params["additional_headers"]["llm_provider-x-request-id"] == "req_stt"
    assert response._hidden_params["additional_headers"]["x-ratelimit-remaining-requests"] == "41"
    assert logging_obj.model_call_details["response_headers"]["x-request-id"] == "req_stt"


def test_audio_transcriptions_records_provider_response_headers():
    logging_obj = _logging_obj()

    with httpx.Client(transport=_transcription_transport()) as http_client:
        response = OpenAIAudioTranscription().audio_transcriptions(
            client=OpenAI(api_key="transport-only", http_client=http_client),
            atranscription=False,
            **_call_kwargs(logging_obj),
        )

    _assert_headers_recorded(response, logging_obj)


@pytest.mark.asyncio
async def test_async_audio_transcriptions_records_provider_response_headers():
    logging_obj = _logging_obj()

    async with httpx.AsyncClient(transport=_transcription_transport()) as http_client:
        response = await OpenAIAudioTranscription().audio_transcriptions(
            client=AsyncOpenAI(api_key="transport-only", http_client=http_client),
            atranscription=True,
            **_call_kwargs(logging_obj),
        )

    _assert_headers_recorded(response, logging_obj)


@pytest.mark.asyncio
async def test_async_audio_transcriptions_logs_the_audio_file_when_the_provider_fails():
    audio_file: Final = ("audio.wav", b"riff-bytes", "audio/wav")
    logging_obj: Final = Logging(
        model="openai/whisper-1",
        messages=[],
        stream=False,
        call_type="atranscription",
        start_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
        litellm_call_id="call-id",
        function_id="function-id",
    )
    failing_transport: Final = httpx.MockTransport(
        lambda request: httpx.Response(500, json={"error": {"message": "provider failure"}})
    )

    async with httpx.AsyncClient(transport=failing_transport) as http_client:
        with pytest.raises(litellm.InternalServerError):
            await litellm.atranscription(
                model="openai/whisper-1",
                file=audio_file,
                client=AsyncOpenAI(api_key="transport-only", http_client=http_client),
                max_retries=0,
                litellm_logging_obj=logging_obj,
            )

    assert logging_obj.model_call_details["input"] == get_audio_file_name(audio_file)
