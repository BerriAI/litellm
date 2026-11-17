import base64
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Final

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

import litellm
from litellm.types.llms.openai import HttpxBinaryResponseContent
from litellm.types.utils import CompletionTokensDetailsWrapper, Usage

_SSE_MEDIA_TYPE: Final = "text/event-stream"
_STREAM_FORMAT_KEY: Final = "stream_format"
_AUDIO_MEDIA_TYPES: Final[Mapping[str, str]] = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}


class _SpeechUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_tokens: int
    output_tokens: int


class _SpeechEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: str
    audio: str | None = None
    usage: _SpeechUsage | None = None


def _parse_event(line: bytes) -> _SpeechEvent | None:
    if not line.startswith(b"data:"):
        return None
    try:
        return _SpeechEvent.model_validate_json(line.removeprefix(b"data:").strip())
    except ValidationError:
        return None


class SpeechUsageDecoder:
    """Turns OpenAI's speech SSE events back into the audio bytes and keeps the usage the last event reports."""

    def __init__(self) -> None:
        self._pending = b""
        self.usage: Usage | None = None

    def feed(self, chunk: bytes) -> bytes:
        *lines, self._pending = (self._pending + chunk).split(b"\n")
        return b"".join(self._audio(event) for line in lines if (event := _parse_event(line)) is not None)

    def _audio(self, event: _SpeechEvent) -> bytes:
        if event.usage is not None:
            self.usage = Usage(
                prompt_tokens=event.usage.input_tokens,
                completion_tokens=event.usage.output_tokens,
                total_tokens=event.usage.input_tokens + event.usage.output_tokens,
                completion_tokens_details=CompletionTokensDetailsWrapper(audio_tokens=event.usage.output_tokens),
            )
        return b"" if event.audio is None else base64.b64decode(event.audio)


def _attach_usage(speech_response: HttpxBinaryResponseContent, usage: Usage | None) -> None:
    if usage is None or speech_response.usage is usage:
        return
    speech_response.usage = usage
    speech_response.set_response_cost(None)


class _UsageAsyncByteStream(httpx.AsyncByteStream):
    def __init__(self, stream: httpx.AsyncByteStream, speech_response: HttpxBinaryResponseContent) -> None:
        self._stream: Final = stream
        self._speech_response: Final = speech_response
        self._decoder: Final = SpeechUsageDecoder()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._stream:
            audio: Final = self._decoder.feed(chunk)
            _attach_usage(self._speech_response, self._decoder.usage)
            if audio:
                yield audio

    async def aclose(self) -> None:
        await self._stream.aclose()


class _UsageSyncByteStream(httpx.SyncByteStream):
    def __init__(self, stream: httpx.SyncByteStream, speech_response: HttpxBinaryResponseContent) -> None:
        self._stream: Final = stream
        self._speech_response: Final = speech_response
        self._decoder: Final = SpeechUsageDecoder()

    def __iter__(self) -> Iterator[bytes]:
        for chunk in self._stream:
            audio: Final = self._decoder.feed(chunk)
            _attach_usage(self._speech_response, self._decoder.usage)
            if audio:
                yield audio

    def close(self) -> None:
        self._stream.close()


def bills_audio_output_tokens(model: str) -> bool:
    model_info: Final = litellm.model_cost.get(model)
    return (
        model_info is not None
        and model_info.get("litellm_provider") == "openai"
        and model_info.get("output_cost_per_audio_token") is not None
    )


def wants_usage_stream(model: str, optional_params: Mapping[str, object]) -> bool:
    extra_body: Final = optional_params.get("extra_body")
    caller_asked: Final = _STREAM_FORMAT_KEY in optional_params or (
        isinstance(extra_body, Mapping) and _STREAM_FORMAT_KEY in extra_body
    )
    return bills_audio_output_tokens(model) and not caller_asked


def with_usage_stream(model: str, optional_params: Mapping[str, object]) -> Mapping[str, object]:
    """Asks OpenAI for the SSE form of the audio, the only form that reports the audio tokens it bills."""
    if not wants_usage_stream(model, optional_params):
        return optional_params
    extra_body: Final = optional_params.get("extra_body")
    caller_body: Final[Mapping[str, object]] = extra_body if isinstance(extra_body, Mapping) else {}
    return {**optional_params, "extra_body": {**caller_body, _STREAM_FORMAT_KEY: "sse"}}


def _audio_headers(headers: httpx.Headers, response_format: object) -> httpx.Headers:
    media_type: Final = _AUDIO_MEDIA_TYPES.get(str(response_format), _AUDIO_MEDIA_TYPES["mp3"])
    kept: Final = tuple(
        (name, value) for name, value in headers.multi_items() if name.lower() not in ("content-type", "content-length")
    )
    return httpx.Headers((*kept, ("content-type", media_type)))


def speech_response_with_usage(
    raw_response: httpx.Response, model: str, optional_params: Mapping[str, object]
) -> HttpxBinaryResponseContent:
    """The speech response the caller gets. The SSE answer we asked for is decoded back to audio with its usage."""
    if not wants_usage_stream(model, optional_params) or not raw_response.headers.get("content-type", "").startswith(
        _SSE_MEDIA_TYPE
    ):
        return HttpxBinaryResponseContent(response=raw_response)
    headers: Final = _audio_headers(raw_response.headers, optional_params.get("response_format"))
    if raw_response.is_stream_consumed:
        decoder: Final = SpeechUsageDecoder()
        audio: Final = decoder.feed(raw_response.content + b"\n")
        return HttpxBinaryResponseContent(
            response=httpx.Response(
                status_code=raw_response.status_code, headers=headers, content=audio, request=raw_response.request
            ),
            usage=decoder.usage,
        )
    raw_response.headers = headers
    speech_response: Final = HttpxBinaryResponseContent(response=raw_response)
    if isinstance(raw_response.stream, httpx.AsyncByteStream):
        raw_response.stream = _UsageAsyncByteStream(raw_response.stream, speech_response)
    elif isinstance(raw_response.stream, httpx.SyncByteStream):
        raw_response.stream = _UsageSyncByteStream(raw_response.stream, speech_response)
    return speech_response
