from datetime import datetime, timezone
from typing import Final, Literal
from unittest.mock import Mock

import httpx
import pytest
from openai import AsyncOpenAI, OpenAI

from litellm import completion_cost
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.openai.cost_calculation import cost_per_second
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


@pytest.mark.parametrize(
    ("response_format", "subtitle", "expected_duration"),
    (
        ("srt", "1\n00:00:00,000 --> 00:00:01,000\nHello\n\n2\n00:00:01,000 --> 00:00:12,500\nWorld\n", 12.5),
        ("vtt", "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello\n\n00:00:01.000 --> 00:00:12.500\nWorld\n", 12.5),
        ("srt", "", None),
        ("text", "Hello world", None),
        ("text", "The timecode is 00:00:12.500", None),
        ("text", "00:00:01.000 --> 00:00:12.500\nHello world", None),
        ("vtt", "WEBVTT\n\n00:01.500 --> 00:02.000\nHello world\n", 2.0),
    ),
    ids=("srt", "vtt", "empty-subtitle", "plain-text", "text-timecode", "text-cue", "short-vtt"),
)
def test_sync_transcription_retains_text_and_charges_subtitle_duration(
    response_format: Literal["srt", "vtt", "text"], subtitle: str, expected_duration: float | None
) -> None:
    logging_obj: Final = Logging(
        model="whisper-1",
        messages=None,
        stream=False,
        call_type="transcription",
        start_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
        litellm_call_id="subtitle-transcription",
        function_id="subtitle-transcription",
    )
    transport: Final = httpx.MockTransport(
        lambda request: httpx.Response(200, text=subtitle, headers={"content-type": "text/plain"})
    )
    with httpx.Client(transport=transport) as http_client:
        response: Final = OpenAIAudioTranscription().audio_transcriptions(
            model="whisper-1",
            audio_file=("audio.wav", b"riff-bytes", "audio/wav"),
            optional_params={"response_format": response_format},
            litellm_params={},
            model_response=TranscriptionResponse(),
            timeout=10.0,
            max_retries=0,
            logging_obj=logging_obj,
            api_key="transport-only",
            api_base=None,
            client=OpenAI(api_key="transport-only", http_client=http_client),
        )

    assert response.model_dump() == TranscriptionResponse(text=subtitle).model_dump()
    assert response._hidden_params.get("audio_transcription_duration") == expected_duration
    assert completion_cost(completion_response=response, model="whisper-1", call_type="transcription") == sum(
        cost_per_second(model="whisper-1", custom_llm_provider="openai", duration=expected_duration or 0.0)
    )


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
