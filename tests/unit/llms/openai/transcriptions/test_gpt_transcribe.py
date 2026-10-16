import io
import json
import wave
from collections.abc import Iterator
from datetime import datetime
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from openai import AsyncAzureOpenAI, AsyncOpenAI, AsyncStream, AzureOpenAI, OpenAI

import litellm
from litellm.litellm_core_utils.audio_utils.transcription_streaming import wrap_transcription_stream
from litellm.llms.azure.audio_transcriptions import AzureAudioTranscription
from litellm.llms.openai.transcriptions.gpt_transformation import (
    OpenAIGPTTranscribeAudioTranscriptionConfig,
)
from litellm.llms.openai.transcriptions.handler import OpenAIAudioTranscription
from litellm.types.utils import TranscriptionResponse
from litellm.utils import get_optional_params_transcription


@pytest.fixture
def local_model_cost_map(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    litellm.get_model_info.cache_clear()
    yield
    litellm.get_model_info.cache_clear()


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


def test_sync_transcription_stream_logs_final_text_and_usage_once() -> None:
    def send_response(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                'data: {"type":"transcript.text.delta","delta":"hello "}\n\n'
                'data: {"type":"transcript.text.done","text":"hello world",'
                '"usage":{"type":"duration","seconds":2.5}}\n\n'
            ),
        )

    logging_obj: Final = MagicMock()
    with OpenAI(
        api_key="sk-test",
        base_url="https://example.com/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(send_response)),
    ) as client:
        stream: Final = client.audio.transcriptions.create(
            model="gpt-transcribe", file=("sample.webm", b"audio"), stream=True
        )
        wrapped: Final = wrap_transcription_stream(stream, logging_obj, datetime(2026, 1, 1))
        received: Final = tuple(wrapped)
        wrapped.close()

    assert tuple(event.type for event in received) == ("transcript.text.delta", "transcript.text.done")
    logging_obj.success_handler.assert_called_once()
    logged_response: Final = logging_obj.success_handler.call_args.args[0]
    assert logged_response.text == "hello world"
    assert logged_response.usage.model_dump(exclude_none=True) == {"type": "duration", "seconds": 2.5}
    logging_obj.failure_handler.assert_not_called()


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
async def test_closed_transcription_stream_without_usage_or_duration_does_not_log_success():
    async def send_response(request: httpx.Request) -> httpx.Response:
        events = (
            {"type": "transcript.text.delta", "delta": "hello"},
            {"type": "transcript.text.done", "text": "hello", "usage": {"type": "duration", "seconds": 1}},
        )
        content = "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode()
        return httpx.Response(200, content=content, headers={"content-type": "text/event-stream"})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(send_response))
    client = AsyncOpenAI(api_key="sk-test", base_url="https://example.com/v1", http_client=http_client)
    stream = await client.audio.transcriptions.create(
        model="gpt-transcribe", file=("sample.webm", b"audio"), stream=True
    )
    logging_obj = MagicMock()
    logging_obj.async_success_handler = AsyncMock()
    logging_obj.async_failure_handler = AsyncMock()
    wrapped_stream = wrap_transcription_stream(stream, logging_obj, datetime.now())

    async for event in wrapped_stream:
        assert event.type == "transcript.text.delta"
        break
    await wrapped_stream.close()
    await client.close()

    logging_obj.handle_sync_success_callbacks_for_async_calls.assert_not_called()
    logging_obj.async_success_handler.assert_not_awaited()
    logging_obj.async_failure_handler.assert_not_awaited()


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
    from litellm.litellm_core_utils.litellm_logging import Logging

    logging_obj = Logging(
        model="gpt-transcribe", messages=[], stream=True, call_type="atranscription",
        start_time=datetime.now(), litellm_call_id="transcription-cost-test", function_id="transcription-cost-test",
    )
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
    logged_response = logging_obj.model_call_details["async_complete_streaming_response"]
    assert logging_obj.model_call_details["response_cost"] == pytest.approx(0.000075)
    assert logging_obj.model_call_details["standard_logging_object"]["response_cost"] == pytest.approx(0.000075)
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


@pytest.mark.parametrize("model", ["gpt-transcribe", "azure/gpt-transcribe"])
def test_gpt_transcribe_rejects_whisper_response_formats(local_model_cost_map: None, model: str) -> None:
    audio_file = io.BytesIO(b"audio")
    audio_file.name = "sample.wav"
    with pytest.raises(litellm.UnsupportedParamsError, match="only supports response_format='json'"):
        litellm.transcription(
            model=model,
            file=audio_file,
            response_format="verbose_json",
            api_key="sk-test",
        )


def test_gpt_live_transcribe_rejects_file_transcription(local_model_cost_map: None) -> None:
    audio_file = io.BytesIO(b"audio")
    audio_file.name = "sample.wav"
    with pytest.raises(litellm.UnsupportedParamsError, match="Realtime API"):
        litellm.transcription(
            model="gpt-live-transcribe",
            file=audio_file,
            api_key="sk-test",
        )


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


@pytest.mark.asyncio
async def test_azure_gpt_transcribe_sends_language_hints_in_sdk_extra_body():
    async def send_response(request: httpx.Request) -> httpx.Response:
        body = await request.aread()
        assert b'name="keywords[]"' in body
        assert b'name="languages[]"' in body
        return httpx.Response(200, json={"text": "hello"})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(send_response))
    client = AsyncAzureOpenAI(
        api_key="azure-test-key",
        azure_endpoint="https://example.openai.azure.com",
        api_version="2025-04-01-preview",
        http_client=http_client,
    )
    audio_file = io.BytesIO(b"audio")
    audio_file.name = "sample.wav"

    response = await AzureAudioTranscription().audio_transcriptions(
        model="gpt-transcribe",
        audio_file=audio_file,
        optional_params={"keywords": ["LiteLLM"], "languages": ["en"]},
        logging_obj=MagicMock(),
        model_response=TranscriptionResponse(),
        timeout=10,
        max_retries=0,
        api_key="azure-test-key",
        api_base="https://example.openai.azure.com",
        api_version="2025-04-01-preview",
        client=client,
        atranscription=True,
    )

    assert response.text == "hello"
    await client.close()
