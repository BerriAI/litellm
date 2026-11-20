import base64
import re
from collections.abc import AsyncIterator, Iterator, Mapping
from fractions import Fraction
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
_BILLED_TOKENS_PER_SECOND: Final = 20
_MIN_SPEED: Final = 0.25
_MAX_SPEED: Final = 4.0
_PCM_BYTES_PER_SECOND: Final = 24000 * 2
_WAV_HEADER_BYTES: Final = 44
_HEAD_BYTES: Final = 64
_OGG_PAGE_HEADER_BYTES: Final = 27
_OGG_MAX_PAGE_BYTES: Final = _OGG_PAGE_HEADER_BYTES + 255 + 255 * 255
_OPUS_GRANULES_PER_SECOND: Final = 48000
_MP3_SAMPLES_PER_FRAME: Final = 576
_MP3_ENCODER_DELAY_SAMPLES: Final = 1105
_MPEG2_LAYER3_KBPS: Final = (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160)
_MPEG2_SAMPLE_RATES: Final = (22050, 24000, 16000)


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


def _pcm_seconds(total: int) -> Fraction:
    return Fraction(total, _PCM_BYTES_PER_SECOND)


def _wav_seconds(head: bytes, total: int) -> Fraction | None:
    byte_rate: Final = int.from_bytes(head[28:32], "little")
    if head[:4] != b"RIFF" or head[36:40] != b"data" or byte_rate == 0:
        return None
    return Fraction(total - _WAV_HEADER_BYTES, byte_rate)


def _is_mpeg2_layer3_frame(head: bytes) -> bool:
    return len(head) >= 4 and head[0] == 0xFF and head[1] & 0xFE == 0xF2


def _mp3_seconds(head: bytes, total: int) -> Fraction | None:
    if not _is_mpeg2_layer3_frame(head):
        return None
    bitrate_index: Final = head[2] >> 4
    sample_rate_index: Final = head[2] >> 2 & 3
    if bitrate_index == 15 or sample_rate_index == 3:
        return None
    sample_rate: Final = _MPEG2_SAMPLE_RATES[sample_rate_index]
    frame_bytes, uneven = divmod(72000 * _MPEG2_LAYER3_KBPS[bitrate_index], sample_rate)
    if frame_bytes == 0 or uneven or total % frame_bytes:
        return None
    return Fraction(total // frame_bytes * _MP3_SAMPLES_PER_FRAME - _MP3_ENCODER_DELAY_SAMPLES, sample_rate)


def _is_last_ogg_page(tail: bytes, start: int) -> bool:
    segments_at: Final = start + _OGG_PAGE_HEADER_BYTES
    if len(tail) < segments_at or tail[start + 4] != 0:
        return False
    segments: Final = tail[start + 26]
    return segments_at + segments + sum(tail[segments_at : segments_at + segments]) == len(tail)


def _opus_seconds(head: bytes, tail: bytes) -> Fraction | None:
    if head[28:36] != b"OpusHead":
        return None
    last_page: Final = next(
        (found.start() for found in re.finditer(b"OggS", tail) if _is_last_ogg_page(tail, found.start())), None
    )
    if last_page is None:
        return None
    played: Final = int.from_bytes(tail[last_page + 6 : last_page + 14], "little") - int.from_bytes(
        head[38:40], "little"
    )
    return Fraction(played, _OPUS_GRANULES_PER_SECOND)


def _accepted_speed(speed: object) -> Fraction | None:
    if isinstance(speed, bool) or not isinstance(speed, (int, float)):
        return None
    return Fraction(str(speed)) if _MIN_SPEED <= speed <= _MAX_SPEED else None


class BilledAudioClock:
    """Counts the audio tokens OpenAI bills for speech: 20 per second of audio at normal speed, not the larger
    `output_tokens` of its `speech.audio.done` event. None for a container whose length it cannot read."""

    def __init__(self, response_format: object, speed: object = None) -> None:
        self._format: Final = str(response_format or "mp3")
        self._speed: Final = Fraction(1) if speed is None else _accepted_speed(speed)
        self._head = b""
        self._tail = b""
        self._total = 0

    def feed(self, audio: bytes) -> None:
        if len(self._head) < _HEAD_BYTES:
            self._head = (self._head + audio)[:_HEAD_BYTES]
        self._tail = (self._tail + audio)[-_OGG_MAX_PAGE_BYTES:]
        self._total += len(audio)

    def _seconds(self) -> Fraction | None:
        match self._format:
            case "pcm":
                return _pcm_seconds(self._total)
            case "wav":
                return _wav_seconds(self._head, self._total)
            case "mp3":
                return _mp3_seconds(self._head, self._total)
            case "opus":
                return _opus_seconds(self._head, self._tail)
            case _:
                return None

    def tokens(self) -> int | None:
        seconds: Final = self._seconds()
        if seconds is None or seconds < 0 or self._speed is None:
            return None
        return int(seconds * self._speed * _BILLED_TOKENS_PER_SECOND)


def _parse_event(line: bytes) -> _SpeechEvent | None:
    if not line.startswith(b"data:"):
        return None
    try:
        return _SpeechEvent.model_validate_json(line.removeprefix(b"data:").strip())
    except ValidationError:
        return None


class SpeechUsageDecoder:
    """Turns OpenAI's speech SSE events back into the audio bytes and keeps the usage the last event reports."""

    def __init__(self, response_format: object = None, speed: object = None) -> None:
        self._pending = b""
        self._clock: Final = BilledAudioClock(response_format, speed)
        self.usage: Usage | None = None

    def feed(self, chunk: bytes) -> bytes:
        *lines, self._pending = (self._pending + chunk).split(b"\n")
        return b"".join(self._audio(event) for line in lines if (event := _parse_event(line)) is not None)

    def _audio(self, event: _SpeechEvent) -> bytes:
        audio: Final = b"" if event.audio is None else base64.b64decode(event.audio)
        self._clock.feed(audio)
        if event.usage is not None:
            billed: Final = self._clock.tokens()
            audio_tokens: Final = event.usage.output_tokens if billed is None else billed
            self.usage = Usage(
                prompt_tokens=event.usage.input_tokens,
                completion_tokens=audio_tokens,
                total_tokens=event.usage.input_tokens + audio_tokens,
                completion_tokens_details=CompletionTokensDetailsWrapper(audio_tokens=audio_tokens),
            )
        return audio


def _decoded_audio(decoder: SpeechUsageDecoder, speech_response: HttpxBinaryResponseContent, chunk: bytes) -> bytes:
    audio: Final = decoder.feed(chunk)
    if decoder.usage is not None and speech_response.usage is not decoder.usage:
        speech_response.usage = decoder.usage  # rebind-ok: the caller already holds this response while it streams
        speech_response.set_response_cost(None)
    return audio


class _UsageAsyncByteStream(httpx.AsyncByteStream):
    def __init__(
        self, stream: httpx.AsyncByteStream, speech_response: HttpxBinaryResponseContent, decoder: SpeechUsageDecoder
    ) -> None:
        self._stream: Final = stream
        self._speech_response: Final = speech_response
        self._decoder: Final = decoder

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._stream:
            if audio := _decoded_audio(self._decoder, self._speech_response, chunk):
                yield audio
        if last_audio := _decoded_audio(self._decoder, self._speech_response, b"\n"):
            yield last_audio

    async def aclose(self) -> None:
        await self._stream.aclose()


class _UsageSyncByteStream(httpx.SyncByteStream):
    def __init__(
        self, stream: httpx.SyncByteStream, speech_response: HttpxBinaryResponseContent, decoder: SpeechUsageDecoder
    ) -> None:
        self._stream: Final = stream
        self._speech_response: Final = speech_response
        self._decoder: Final = decoder

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
    response_format: Final = optional_params.get("response_format")
    headers: Final = _audio_headers(raw_response.headers, response_format)
    decoder: Final = SpeechUsageDecoder(response_format, optional_params.get("speed"))
    if raw_response.is_stream_consumed:
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
        _UsageAsyncByteStream(raw_response.stream, speech_response, decoder)
        if isinstance(raw_response.stream, httpx.AsyncByteStream)
        else _UsageSyncByteStream(raw_response.stream, speech_response, decoder)
    )
    return speech_response
