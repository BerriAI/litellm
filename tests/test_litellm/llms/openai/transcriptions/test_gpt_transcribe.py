import io
import json
import wave
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from openai import AsyncOpenAI, AsyncStream, AzureOpenAI

import litellm
from litellm.litellm_core_utils.audio_utils.transcription_streaming import wrap_transcription_stream
from litellm.llms.azure.audio_transcriptions import AzureAudioTranscription
from litellm.llms.openai.transcriptions.gpt_transformation import (
    OpenAIGPTTranscribeAudioTranscriptionConfig,
)
from litellm.llms.openai.transcriptions.handler import OpenAIAudioTranscription
from litellm.main import _validate_gpt_transcription_request
from litellm.types.utils import TranscriptionResponse
from litellm.utils import get_optional_params_transcription


def test_gpt_transcribe_config_uses_native_parameters_and_json():
    config = OpenAIGPTTranscribeAudioTranscriptionConfig()
    supported = config.get_supported_openai_params("gpt-transcribe")
    assert supported == ["prompt", "response_format", "keywords", "languages", "stream"]

    audio_file = io.BytesIO(b"audio")
    request = config.transform_audio_transcription_request(
        model="gpt-transcribe",
        audio_file=audio_file,
        optional_params={"keywords": ["LiteLLM"], "languages": ["en", "fr"], "stream": True},
        litellm_params={},
    )
    assert request.data["response_format"] == "json"
    assert request.data["keywords"] == ["LiteLLM"]
    assert request.data["languages"] == ["en", "fr"]
    assert request.data["stream"] is True


def test_gpt_transcribe_optional_params_are_preserved():
    params = get_optional_params_transcription(
        model="gpt-transcribe",
        custom_llm_provider="openai",
        keywords=["LiteLLM", "Realtime API"],
        languages=["en", "fr"],
        stream=True,
    )
    assert params == {
        "keywords": ["LiteLLM", "Realtime API"],
        "languages": ["en", "fr"],
        "stream": True,
    }


def test_transcription_response_preserves_empty_languages():
    response = TranscriptionResponse(text="hello", languages=[])
    assert response.model_dump()["languages"] == []


@pytest.mark.asyncio
async def test_openai_handler_returns_native_typed_stream():
    async def send_response(request: httpx.Request) -> httpx.Response:
        body = await request.aread()
        assert b'name="keywords[]"' in body
        assert b'name="languages[]"' in body
        assert b'name="stream"' in body
        events = (
            {"type": "transcript.text.delta", "delta": "hello "},
            {
                "type": "transcript.text.done",
                "text": "hello world",
                "languages": [],
                "usage": {
                    "type": "tokens",
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "total_tokens": 12,
                },
            },
        )
        content = "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode()
        return httpx.Response(200, content=content, headers={"content-type": "text/event-stream"})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(send_response))
    openai_client = AsyncOpenAI(api_key="sk-test", base_url="https://example.com/v1", http_client=http_client)
    logging_obj = MagicMock()
    logging_obj.model_call_details = {}
    audio_file = io.BytesIO(b"audio")
    audio_file.name = "sample.wav"

    handler = OpenAIAudioTranscription()
    result = handler.audio_transcriptions(
        model="gpt-transcribe",
        audio_file=audio_file,
        optional_params={"keywords": ["LiteLLM"], "languages": ["en"], "stream": True},
        litellm_params={},
        model_response=TranscriptionResponse(),
        timeout=10,
        max_retries=0,
        logging_obj=logging_obj,
        api_key="sk-test",
        api_base="https://example.com/v1",
        client=openai_client,
        atranscription=True,
        provider_config=OpenAIGPTTranscribeAudioTranscriptionConfig(),
    )
    stream = await result
    assert isinstance(stream, AsyncStream)
    logging_obj.async_success_handler = AsyncMock()
    logging_obj.async_failure_handler = AsyncMock()
    wrapped_stream = wrap_transcription_stream(stream, logging_obj, datetime.now())
    received = [event async for event in wrapped_stream]
    await wrapped_stream.close()
    await openai_client.close()

    assert [event.type for event in received] == ["transcript.text.delta", "transcript.text.done"]
    assert received[-1].languages == []
    logging_obj.async_success_handler.assert_awaited_once()
    logged_response = logging_obj.async_success_handler.await_args.kwargs["result"]
    assert logged_response.text == "hello world"
    assert logged_response.languages == []


@pytest.mark.asyncio
async def test_atranscription_stream_preserves_duration_for_callback_cost():
    async def send_response(request: httpx.Request) -> httpx.Response:
        events = (
            {"type": "transcript.text.delta", "delta": "hello "},
            {
                "type": "transcript.text.done",
                "text": "hello world",
                "usage": {
                    "type": "tokens",
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "total_tokens": 12,
                },
            },
        )
        content = "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode()
        return httpx.Response(200, content=content, headers={"content-type": "text/event-stream"})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(send_response))
    openai_client = AsyncOpenAI(api_key="sk-test", base_url="https://example.com/v1", http_client=http_client)
    logging_obj = MagicMock()
    logging_obj.model_call_details = {}
    logging_obj.async_success_handler = AsyncMock()
    logging_obj.async_failure_handler = AsyncMock()
    audio_file = io.BytesIO()
    with wave.open(audio_file, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 16000)
    audio_file.name = "sample.wav"

    stream = await litellm.atranscription(
        model="openai/gpt-transcribe",
        file=audio_file,
        stream=True,
        client=openai_client,
        litellm_logging_obj=logging_obj,
    )
    received = [event async for event in stream]
    await stream.close()
    await openai_client.close()

    assert [event.type for event in received] == ["transcript.text.delta", "transcript.text.done"]
    logging_obj.async_success_handler.assert_awaited_once()
    logged_response = logging_obj.async_success_handler.await_args.kwargs["result"]
    assert logged_response._hidden_params["audio_transcription_duration"] == pytest.approx(1.0)


def test_gpt_transcribe_rejects_conflicting_language_inputs():
    audio_file = io.BytesIO(b"audio")
    audio_file.name = "sample.wav"
    with pytest.raises(litellm.UnsupportedParamsError, match="cannot be used together"):
        litellm.transcription(
            model="gpt-transcribe",
            file=audio_file,
            language="en",
            languages=["fr"],
            api_key="sk-test",
        )


def test_gpt_transcribe_rejects_whisper_response_formats():
    audio_file = io.BytesIO(b"audio")
    audio_file.name = "sample.wav"
    with pytest.raises(litellm.UnsupportedParamsError, match="only supports response_format='json'"):
        litellm.transcription(
            model="gpt-transcribe",
            file=audio_file,
            response_format="verbose_json",
            api_key="sk-test",
        )


def test_gpt_live_transcribe_rejects_file_transcription():
    audio_file = io.BytesIO(b"audio")
    audio_file.name = "sample.wav"
    with pytest.raises(litellm.UnsupportedParamsError, match="Realtime API"):
        litellm.transcription(
            model="gpt-live-transcribe",
            file=audio_file,
            api_key="sk-test",
        )


def test_azure_async_gpt_transcribe_forwards_v1_api_version():
    handler = AzureAudioTranscription()
    handler.async_audio_transcriptions = MagicMock(return_value=MagicMock())

    handler.audio_transcriptions(
        model="gpt-transcribe",
        audio_file=io.BytesIO(b"audio"),
        optional_params={"stream": True},
        logging_obj=MagicMock(),
        model_response=TranscriptionResponse(),
        timeout=10,
        max_retries=0,
        api_key="sk-test",
        api_base="https://example.openai.azure.com",
        api_version="v1",
        atranscription=True,
    )

    assert handler.async_audio_transcriptions.call_args.kwargs["api_version"] == "v1"


@pytest.mark.parametrize("api_version", [None, "v1", "latest", "preview"])
def test_azure_gpt_transcribe_uses_deployment_scoped_api_version(api_version: str | None):
    resolved_api_version = _validate_gpt_transcription_request(
        model="gpt-transcribe",
        custom_llm_provider="azure",
        language=None,
        languages=None,
        response_format="json",
        api_version=api_version,
    )

    assert resolved_api_version == litellm.AZURE_DEFAULT_API_VERSION


def test_azure_gpt_transcribe_uses_deployment_scoped_route():
    def send_response(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == (
            "https://example.openai.azure.com/openai/deployments/gpt-transcribe/audio/transcriptions"
            f"?api-version={litellm.AZURE_DEFAULT_API_VERSION}"
        )
        return httpx.Response(
            200,
            json={"text": "hello", "languages": [{"code": "en"}], "usage": {"type": "duration", "seconds": 1}},
        )

    http_client = httpx.Client(transport=httpx.MockTransport(send_response))
    client = AzureOpenAI(
        api_key="azure-test-key",
        azure_endpoint="https://example.openai.azure.com",
        api_version=litellm.AZURE_DEFAULT_API_VERSION,
        http_client=http_client,
    )
    audio_file = io.BytesIO(b"audio")
    audio_file.name = "sample.wav"

    response = AzureAudioTranscription().audio_transcriptions(
        model="gpt-transcribe",
        audio_file=audio_file,
        optional_params={"response_format": "json"},
        logging_obj=MagicMock(),
        model_response=TranscriptionResponse(),
        timeout=10,
        max_retries=0,
        api_key="azure-test-key",
        api_base="https://example.openai.azure.com",
        api_version=litellm.AZURE_DEFAULT_API_VERSION,
        client=client,
    )

    assert response.text == "hello"
    assert response.languages is not None
    assert [language.code for language in response.languages] == ["en"]
    client.close()


def test_azure_gpt_transcribe_preserves_dated_api_version():
    resolved_api_version = _validate_gpt_transcription_request(
        model="gpt-transcribe",
        custom_llm_provider="azure",
        language=None,
        languages=None,
        response_format="json",
        api_version="2025-04-01-preview",
    )

    assert resolved_api_version == "2025-04-01-preview"
