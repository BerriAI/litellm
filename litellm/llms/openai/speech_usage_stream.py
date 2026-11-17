import base64
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Final

import httpx
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from litellm.litellm_core_utils import model_cost_table
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


class _SpeechPricing(BaseModel):
    model_config = ConfigDict(frozen=True)

    litellm_provider: str | None = None
    output_cost_per_audio_token: float | None = None


_EXTRA_BODY: Final = TypeAdapter(dict[str, object])


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


def _decoded_audio(decoder: SpeechUsageDecoder, speech_response: HttpxBinaryResponseContent, chunk: bytes) -> bytes:
    audio: Final = decoder.feed(chunk)
    if decoder.usage is not None and speech_response.usage is not decoder.usage:
        speech_response.usage = decoder.usage  # rebind-ok: the caller already holds this response while it streams
        speech_response.set_response_cost(None)
    return audio


class _UsageAsyncByteStream(httpx.AsyncByteStream):
    def __init__(self, stream: httpx.AsyncByteStream, speech_response: HttpxBinaryResponseContent) -> None:
        self._stream: Final = stream
        self._speech_response: Final = speech_response
        self._decoder: Final = SpeechUsageDecoder()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._stream:
            if audio := _decoded_audio(self._decoder, self._speech_response, chunk):
                yield audio
        if last_audio := _decoded_audio(self._decoder, self._speech_response, b"\n"):
            yield last_audio

    async def aclose(self) -> None:
        await self._stream.aclose()


class _UsageSyncByteStream(httpx.SyncByteStream):
    def __init__(self, stream: httpx.SyncByteStream, speech_response: HttpxBinaryResponseContent) -> None:
        self._stream: Final = stream
        self._speech_response: Final = speech_response
        self._decoder: Final = SpeechUsageDecoder()

    def __iter__(self) -> Iterator[bytes]:
        for chunk in self._stream:
            if audio := _decoded_audio(self._decoder, self._speech_response, chunk):
                yield audio
        if last_audio := _decoded_audio(self._decoder, self._speech_response, b"\n"):
            yield last_audio

    def close(self) -> None:
        self._stream.close()


def bills_audio_output_tokens(model: str) -> bool:
    row: Final = model_cost_table.published().get(model)
    if row is None:
        return False
    try:
        pricing: Final = _SpeechPricing.model_validate(row)
    except ValidationError:
        return False
    return pricing.litellm_provider == "openai" and pricing.output_cost_per_audio_token is not None


def _caller_extra_body(optional_params: Mapping[str, object]) -> Mapping[str, object]:
    return _EXTRA_BODY.validate_python(optional_params.get("extra_body") or {})


def _caller_extra_headers(optional_params: Mapping[str, object]) -> Mapping[str, object]:
    return _EXTRA_BODY.validate_python(optional_params.get("extra_headers") or {})


def wants_usage_stream(model: str, optional_params: Mapping[str, object]) -> bool:
    caller_asked: Final = _STREAM_FORMAT_KEY in optional_params or _STREAM_FORMAT_KEY in _caller_extra_body(
        optional_params
    )
    return bills_audio_output_tokens(model) and not caller_asked


def with_usage_stream(model: str, optional_params: Mapping[str, object]) -> Mapping[str, object]:
    """Asks OpenAI for the SSE form of the audio, the only form that reports the audio tokens it bills."""
    if not wants_usage_stream(model, optional_params):
        return optional_params
    return {
        **optional_params,
        "extra_body": {**_caller_extra_body(optional_params), _STREAM_FORMAT_KEY: "sse"},
        "extra_headers": {**_caller_extra_headers(optional_params), "Accept-Encoding": "identity"},
    }


def _audio_headers(headers: httpx.Headers, response_format: object) -> httpx.Headers:
    media_type: Final = _AUDIO_MEDIA_TYPES.get(str(response_format), _AUDIO_MEDIA_TYPES["mp3"])
    kept: Final = tuple(
        (name, value)
        for name, value in headers.multi_items()
        if name.lower() not in ("content-type", "content-length", "content-encoding")
    )
    return httpx.Headers((*kept, ("content-type", media_type)))


def speech_response_with_usage(
    raw_response: httpx.Response, model: str, optional_params: Mapping[str, object]
) -> HttpxBinaryResponseContent:
    """The speech response the caller gets. The SSE answer we asked for is decoded back to audio with its usage."""
    unchanged: Final = HttpxBinaryResponseContent(response=raw_response)
    if not wants_usage_stream(model, optional_params) or not (unchanged.content_type or "").startswith(_SSE_MEDIA_TYPE):
        return unchanged
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
    raw_response.headers = headers  # rebind-ok: the unread provider response is what streams the audio to the caller
    speech_response: Final = HttpxBinaryResponseContent(response=raw_response)
    raw_response.stream = (  # rebind-ok: the unread provider response is what streams the audio to the caller
        _UsageAsyncByteStream(raw_response.stream, speech_response)
        if isinstance(raw_response.stream, httpx.AsyncByteStream)
        else _UsageSyncByteStream(raw_response.stream, speech_response)
    )
    return speech_response
