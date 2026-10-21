import io
import json
import wave
from collections.abc import Callable, Iterator
from datetime import datetime
from typing import Final, NamedTuple
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from openai import AsyncAzureOpenAI, AsyncOpenAI, AsyncStream, AzureOpenAI, OpenAI

import litellm
from litellm.litellm_core_utils.audio_utils.transcription_streaming import wrap_transcription_stream
from litellm.llms.azure.audio_transcriptions import AzureAudioTranscription
from litellm.llms.custom_httpx.http_handler import HTTPHandler
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
        optional_params={"keywords": ["LiteLLM"], "languages": ["en", "fr"], "stream": True, "response_format": "json"},
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
    assert json.loads(response.model_dump_json())["languages"] == []


def test_sync_transcription_stream_logs_final_text_and_usage_once() -> None:
    def send_response(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                'data: {"type":"transcript.text.delta","delta":"hello "}\n\n'
                'data: {"type":"transcript.text.done","text":"hello world",'
                '"usage":{"type":"duration","seconds":2.5},"languages":[{"code":"en"}]}\n\n'
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
    assert json.loads(logged_response.model_dump_json())["languages"] == [{"code": "en"}]
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
    assert logged_response.languages == ()


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


@pytest.mark.parametrize("model", ["gpt-transcribe", "azure/gpt-transcribe"])
def test_gpt_transcribe_rejects_whisper_response_formats(local_model_cost_map: None, model: str) -> None:
    audio_file = io.BytesIO(b"audio")
    audio_file.name = "sample.wav"
    with pytest.raises(litellm.UnsupportedParamsError, match="only supports response_format 'json' or 'text'"):
        litellm.transcription(
            model=model,
            file=audio_file,
            response_format="verbose_json",
            api_key="sk-test",
        )


@pytest.mark.parametrize("drop_globally", (True, False))
def test_gpt_transcribe_drops_whisper_response_formats_under_drop_params(
    local_model_cost_map: None, monkeypatch: pytest.MonkeyPatch, drop_globally: bool
) -> None:
    sent_bodies: Final[list[bytes]] = []  # mutable-ok: collects what the mock upstream received

    def send_response(request: httpx.Request) -> httpx.Response:
        sent_bodies.append(request.read())
        return httpx.Response(200, json={"text": "hello", "usage": {"type": "duration", "seconds": 1}})

    monkeypatch.setattr(litellm, "drop_params", drop_globally)
    audio_file: Final = io.BytesIO(b"audio")
    audio_file.name = "sample.wav"
    with OpenAI(
        api_key="sk-test",
        base_url="https://example.com/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(send_response)),
    ) as client:
        response: Final = litellm.transcription(
            model="gpt-transcribe",
            file=audio_file,
            response_format="verbose_json",
            client=client,
            **({} if drop_globally else {"drop_params": True}),
        )

    assert response.text == "hello"
    assert len(sent_bodies) == 1
    assert b"verbose_json" not in sent_bodies[0]


def _openai_transcription_client(transport: httpx.MockTransport) -> OpenAI:
    return OpenAI(api_key="sk-test", base_url="https://example.com/v1", http_client=httpx.Client(transport=transport))


def _azure_transcription_client(transport: httpx.MockTransport) -> AzureOpenAI:
    return AzureOpenAI(
        api_key="azure-test-key",
        azure_endpoint="https://example.openai.azure.com",
        api_version=litellm.AZURE_DEFAULT_API_VERSION,
        http_client=httpx.Client(transport=transport),
    )


@pytest.mark.parametrize(
    ("model", "make_client"),
    (("gpt-transcribe", _openai_transcription_client), ("azure/gpt-transcribe", _azure_transcription_client)),
)
def test_gpt_transcribe_sends_text_response_format_to_the_provider(
    local_model_cost_map: None, model: str, make_client: Callable[[httpx.MockTransport], OpenAI | AzureOpenAI]
) -> None:
    sent_bodies: Final[list[bytes]] = []  # mutable-ok: collects what the mock upstream received

    def send_response(request: httpx.Request) -> httpx.Response:
        sent_bodies.append(request.read())
        return httpx.Response(200, text="hello", headers={"content-type": "text/plain"})

    audio_file: Final = io.BytesIO(b"audio")
    audio_file.name = "sample.wav"
    with make_client(httpx.MockTransport(send_response)) as client:
        response: Final = litellm.transcription(
            model=model,
            file=audio_file,
            response_format="text",
            api_key="sk-test",
            api_base="https://example.openai.azure.com",
            api_version=litellm.AZURE_DEFAULT_API_VERSION,
            client=client,
        )

    assert response.text == "hello"
    assert len(sent_bodies) == 1
    assert b'name="response_format"\r\n\r\ntext\r\n' in sent_bodies[0]


def test_file_transcription_reaches_models_whose_cost_map_lists_other_endpoints(
    local_model_cost_map: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent_requests: Final[list[httpx.Request]] = []  # mutable-ok: collects what the mock upstreams received

    def send_gemini_response(request: httpx.Request) -> httpx.Response:
        sent_requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "interaction-1",
                "status": "completed",
                "steps": [
                    {"type": "model_generation", "content": [{"type": "text", "text": "hello", "annotations": []}]}
                ],
            },
        )

    def send_whisper_response(request: httpx.Request) -> httpx.Response:
        sent_requests.append(request)
        return httpx.Response(200, json={"text": "hello"})

    gemini_response: Final = litellm.transcription(
        model="gemini/gemini-2.5-flash",
        file=_one_second_wav(),
        api_key="gemini-test-key",
        client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(send_gemini_response))),
    )
    monkeypatch.setattr(litellm, "client_session", httpx.Client(transport=httpx.MockTransport(send_whisper_response)))
    whisper_response: Final = litellm.transcription(
        model="openai/gpt-realtime-whisper",
        file=_one_second_wav(),
        api_key="self-hosted-whisper-key",
        api_base="https://whisper.self-hosted.example/v1",
    )

    assert gemini_response.text == "hello"
    assert whisper_response.text == "hello"
    assert tuple(str(request.url.host) for request in sent_requests) == (
        "generativelanguage.googleapis.com",
        "whisper.self-hosted.example",
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


def _one_second_wav() -> io.BytesIO:
    audio_file: Final = io.BytesIO()
    with wave.open(audio_file, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 16000)
    audio_file.name = "sample.wav"
    audio_file.seek(0)
    return audio_file


class _LoggedSpend(NamedTuple):
    cost: float | None
    usage: dict[str, object] | None


def _logged_spend(kwargs: dict[str, object]) -> _LoggedSpend:
    payload: Final = kwargs["standard_logging_object"]
    return _LoggedSpend(payload["response_cost"], payload["metadata"]["usage_object"])


class _SuccessSpendRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.async_spends: Final[asyncio.Queue[_LoggedSpend]] = asyncio.Queue()
        self.sync_spends: Final[queue.Queue[_LoggedSpend]] = queue.Queue()

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
        await self.async_spends.put(_logged_spend(kwargs))

    def log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
        self.sync_spends.put(_logged_spend(kwargs))


@pytest.mark.asyncio
async def test_atranscription_bills_the_transcript_when_the_model_ignores_stream(
    local_model_cost_map: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def send_response(request: httpx.Request) -> httpx.Response:
        return _duration_priced_transcript(streamed=False)

    recorder: Final = _SuccessSpendRecorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    client: Final = AsyncOpenAI(
        api_key="sk-test",
        base_url="https://example.com/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(send_response)),
    )
    await litellm.atranscription(model="openai/whisper-1", file=_one_second_wav(), client=client)
    unstreamed_spend: Final = await asyncio.wait_for(recorder.async_spends.get(), timeout=5)
    ignored_stream: Final = await litellm.atranscription(
        model="openai/whisper-1", file=_one_second_wav(), stream=True, client=client
    )
    ignored_stream_spend: Final = await asyncio.wait_for(recorder.async_spends.get(), timeout=5)
    await client.close()

    assert isinstance(ignored_stream, TranscriptionResponse)
    assert ignored_stream.text == "hello world"
    assert unstreamed_spend.cost is not None
    assert unstreamed_spend.cost > 0
    assert unstreamed_spend.usage
    assert ignored_stream_spend == unstreamed_spend


def test_azure_transcription_bills_the_transcript_when_the_deployment_ignores_stream(
    local_model_cost_map: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def send_response(request: httpx.Request) -> httpx.Response:
        return _duration_priced_transcript(streamed=False)

    recorder: Final = _SuccessSpendRecorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    client: Final = AzureOpenAI(
        api_key="azure-test-key",
        azure_endpoint="https://example.openai.azure.com",
        api_version=litellm.AZURE_DEFAULT_API_VERSION,
        http_client=httpx.Client(transport=httpx.MockTransport(send_response)),
    )
    azure_params: Final = {
        "client": client,
        "api_key": "azure-test-key",
        "api_base": "https://example.openai.azure.com",
        "api_version": litellm.AZURE_DEFAULT_API_VERSION,
    }
    litellm.transcription(model="azure/whisper-1", file=_one_second_wav(), **azure_params)
    unstreamed_spend: Final = recorder.sync_spends.get(timeout=5)
    ignored_stream: Final = litellm.transcription(
        model="azure/whisper-1", file=_one_second_wav(), stream=True, **azure_params
    )
    ignored_stream_spend: Final = recorder.sync_spends.get(timeout=5)
    client.close()

    assert isinstance(ignored_stream, TranscriptionResponse)
    assert ignored_stream.text == "hello world"
    assert unstreamed_spend.cost is not None
    assert unstreamed_spend.cost > 0
    assert unstreamed_spend.usage
    assert ignored_stream_spend == unstreamed_spend


def _duration_priced_transcript(streamed: bool) -> httpx.Response:
    usage: Final = {"type": "duration", "seconds": 2}
    if not streamed:
        return httpx.Response(200, json={"text": "hello world", "usage": usage})
    events: Final = (
        {"type": "transcript.text.delta", "delta": "hello world"},
        {"type": "transcript.text.done", "text": "hello world", "usage": usage},
    )
    return httpx.Response(
        200,
        content="".join(f"data: {json.dumps(event)}\n\n" for event in events).encode(),
        headers={"content-type": "text/event-stream"},
    )


def test_whisper_stream_request_answered_with_json_keeps_the_unstreamed_response_shape(
    local_model_cost_map: None,
) -> None:
    verbose_transcript: Final = {
        "task": "transcribe",
        "language": "english",
        "duration": 1.0,
        "text": "hello world",
        "usage": {"type": "duration", "seconds": 1},
    }
    with _openai_transcription_client(
        httpx.MockTransport(lambda request: httpx.Response(200, json=verbose_transcript))
    ) as client:
        unstreamed: Final = litellm.transcription(model="openai/whisper-1", file=_one_second_wav(), client=client)
        ignored_stream: Final = litellm.transcription(
            model="openai/whisper-1", file=_one_second_wav(), stream=True, client=client
        )

    assert isinstance(ignored_stream, TranscriptionResponse)
    assert json.loads(ignored_stream.model_dump_json()) == json.loads(unstreamed.model_dump_json())


@pytest.mark.parametrize("requested_stream", ("true", "True", "1"))
def test_transcription_stream_requested_as_a_form_string_streams_like_a_bool(
    local_model_cost_map: None, requested_stream: str
) -> None:
    sent_bodies: Final[list[bytes]] = []  # mutable-ok: collects what the mock upstream received

    def send_response(request: httpx.Request) -> httpx.Response:
        sent_bodies.append(request.read())
        return _duration_priced_transcript(streamed=_asks_for_stream(sent_bodies[-1]))

    with _openai_transcription_client(httpx.MockTransport(send_response)) as client:
        stream: Final = litellm.transcription(
            model="openai/gpt-transcribe", file=_one_second_wav(), stream=requested_stream, client=client
        )
        events: Final = [event.type for event in stream]

    assert events == ["transcript.text.delta", "transcript.text.done"]
    assert len(sent_bodies) == 1


def _asks_for_stream(request_body: bytes) -> bool:
    return b'name="stream"\r\n\r\ntrue' in request_body


@pytest.mark.asyncio
async def test_atranscription_stream_bills_the_provider_duration_like_the_unstreamed_call(
    local_model_cost_map: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def send_response(request: httpx.Request) -> httpx.Response:
        return _duration_priced_transcript(streamed=_asks_for_stream(await request.aread()))

    recorder: Final = _SuccessSpendRecorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    client: Final = AsyncOpenAI(
        api_key="sk-test",
        base_url="https://example.com/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(send_response)),
    )
    await litellm.atranscription(model="openai/gpt-transcribe", file=_one_second_wav(), client=client)
    unstreamed_spend: Final = await asyncio.wait_for(recorder.async_spends.get(), timeout=5)
    stream: Final = await litellm.atranscription(
        model="openai/gpt-transcribe", file=_one_second_wav(), stream=True, client=client
    )
    events: Final = [event.type async for event in stream]
    streamed_spend: Final = await asyncio.wait_for(recorder.async_spends.get(), timeout=5)
    await client.close()

    assert events == ["transcript.text.delta", "transcript.text.done"]
    assert unstreamed_spend.cost is not None
    assert unstreamed_spend.cost > 0
    assert unstreamed_spend.usage
    assert streamed_spend == unstreamed_spend


def test_transcription_stream_bills_the_provider_duration_like_the_unstreamed_call(
    local_model_cost_map: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def send_response(request: httpx.Request) -> httpx.Response:
        return _duration_priced_transcript(streamed=_asks_for_stream(request.read()))

    recorder: Final = _SuccessSpendRecorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    client: Final = OpenAI(
        api_key="sk-test",
        base_url="https://example.com/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(send_response)),
    )
    litellm.transcription(model="openai/gpt-transcribe", file=_one_second_wav(), client=client)
    unstreamed_spend: Final = recorder.sync_spends.get(timeout=5)
    stream: Final = litellm.transcription(
        model="openai/gpt-transcribe", file=_one_second_wav(), stream=True, client=client
    )
    events: Final = [event.type for event in stream]
    streamed_spend: Final = recorder.sync_spends.get(timeout=5)
    client.close()

    assert events == ["transcript.text.delta", "transcript.text.done"]
    assert unstreamed_spend.cost is not None
    assert unstreamed_spend.cost > 0
    assert unstreamed_spend.usage
    assert streamed_spend == unstreamed_spend


class _LoggedResponseRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.responses: Final[queue.Queue[object]] = queue.Queue()

    def log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
        self.responses.put(kwargs["standard_logging_object"]["response"])


@pytest.mark.parametrize(
    "provider_body",
    ({"text": "hello"}, {"text": "hello", "languages": ["en"]}, {"text": "hello", "languages": None}),
)
def test_transcription_without_parsed_languages_has_no_languages_key(
    local_model_cost_map: None, monkeypatch: pytest.MonkeyPatch, provider_body: dict[str, object]
) -> None:
    recorder: Final = _LoggedResponseRecorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    with _openai_transcription_client(
        httpx.MockTransport(lambda request: httpx.Response(200, json=provider_body))
    ) as client:
        response: Final = litellm.transcription(model="openai/whisper-1", file=_one_second_wav(), client=client)
    logged_response: Final = recorder.responses.get(timeout=5)

    assert response.text == "hello"
    assert "languages" not in json.loads(response.model_dump_json())
    assert isinstance(logged_response, dict)
    assert "languages" not in logged_response


def test_unstreamed_transcription_sends_no_stream_field(local_model_cost_map: None) -> None:
    sent_bodies: Final[list[bytes]] = []  # mutable-ok: collects what the mock upstream received

    def send_response(request: httpx.Request) -> httpx.Response:
        sent_bodies.append(request.read())
        return httpx.Response(200, json={"text": "hello"})

    with _openai_transcription_client(httpx.MockTransport(send_response)) as client:
        response: Final = litellm.transcription(model="openai/whisper-1", file=_one_second_wav(), client=client)

    assert response.text == "hello"
    assert len(sent_bodies) == 1
    assert b'name="stream"' not in sent_bodies[0]


def test_transcription_forwards_both_language_and_languages(local_model_cost_map: None) -> None:
    sent_bodies: Final[list[bytes]] = []  # mutable-ok: collects what the mock upstream received

    def send_response(request: httpx.Request) -> httpx.Response:
        sent_bodies.append(request.read())
        return httpx.Response(200, json={"text": "hello", "usage": {"type": "duration", "seconds": 1}})

    with _openai_transcription_client(httpx.MockTransport(send_response)) as client:
        response: Final = litellm.transcription(
            model="openai/gpt-transcribe", file=_one_second_wav(), language="en", languages=["fr"], client=client
        )

    assert response.text == "hello"
    assert len(sent_bodies) == 1
    assert b'name="language"\r\n\r\nen\r\n' in sent_bodies[0]
    assert b'name="languages[]"\r\n\r\nfr\r\n' in sent_bodies[0]


def _gpt_transcribe_cost_per_second() -> float:
    rate: Final = litellm.model_cost["gpt-transcribe"]["input_cost_per_second"]
    assert isinstance(rate, float)
    return rate


async def _spend_for_one_second_wav(
    monkeypatch: pytest.MonkeyPatch, provider_body: dict[str, object], router: litellm.Router | None
) -> _LoggedSpend:
    recorder: Final = _SuccessSpendRecorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    client: Final = AsyncOpenAI(
        api_key="sk-test",
        base_url="https://example.com/v1",
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=provider_body))
        ),
    )
    if router is None:
        await litellm.atranscription(model="openai/gpt-transcribe", file=_one_second_wav(), client=client)
    else:
        await router.atranscription(model="my-transcribe", file=_one_second_wav(), client=client)
    spend: Final = await asyncio.wait_for(recorder.async_spends.get(), timeout=5)
    await client.close()
    return spend


@pytest.mark.asyncio
async def test_custom_named_gpt_transcribe_deployment_bills_the_provider_seconds(
    local_model_cost_map: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "my-transcribe",
                "litellm_params": {"model": "openai/my-transcribe", "api_key": "sk-test"},
                "model_info": {"base_model": "gpt-transcribe"},
            }
        ]
    )

    spend: Final = await _spend_for_one_second_wav(
        monkeypatch, {"text": "hello", "usage": {"type": "duration", "seconds": 2}}, router
    )

    assert spend.cost == pytest.approx(2 * _gpt_transcribe_cost_per_second())


@pytest.mark.asyncio
async def test_gpt_transcribe_bills_the_file_length_when_the_provider_reports_zero_seconds(
    local_model_cost_map: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    spend: Final = await _spend_for_one_second_wav(
        monkeypatch, {"text": "hello", "usage": {"type": "duration", "seconds": 0}}, None
    )

    assert spend.cost == pytest.approx(_gpt_transcribe_cost_per_second())


@pytest.mark.asyncio
async def test_m4a_transcription_stream_that_ends_early_logs_one_spend_row_like_the_unstreamed_call(
    local_model_cost_map: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def send_response(request: httpx.Request) -> httpx.Response:
        if not _asks_for_stream(await request.aread()):
            return httpx.Response(200, json={"text": "hello"})
        return httpx.Response(
            200,
            content=f"data: {json.dumps({'type': 'transcript.text.delta', 'delta': 'hello'})}\n\n".encode(),
            headers={"content-type": "text/event-stream"},
        )

    recorder: Final = _SuccessSpendRecorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    client: Final = AsyncOpenAI(
        api_key="sk-test",
        base_url="https://example.com/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(send_response)),
    )
    await litellm.atranscription(model="openai/gpt-transcribe", file=("sample.m4a", b"m4a audio"), client=client)
    unstreamed_spend: Final = await asyncio.wait_for(recorder.async_spends.get(), timeout=5)
    stream: Final = await litellm.atranscription(
        model="openai/gpt-transcribe", file=("sample.m4a", b"m4a audio"), stream=True, client=client
    )
    events: Final = [event.type async for event in stream]
    streamed_spend: Final = await asyncio.wait_for(recorder.async_spends.get(), timeout=5)
    await asyncio.sleep(0.2)
    await client.close()

    assert events == ["transcript.text.delta"]
    assert streamed_spend.cost == unstreamed_spend.cost
    assert recorder.async_spends.empty()
