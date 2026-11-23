import base64
import re
from collections.abc import AsyncIterator, Iterator, Mapping
from fractions import Fraction
from functools import reduce
from itertools import accumulate
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
_ADTS_HEADER_BYTES: Final = 7
_ADTS_SAMPLE_RATES: Final = (96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350)
_AAC_SAMPLES_PER_BLOCK: Final = 1024
_AAC_ENCODER_DELAY_SAMPLES: Final = 1024
_FLAC_FIXED_BLOCK_SYNC: Final = b"\xff\xf8"
_FLAC_MIN_FRAME_BYTES: Final = 9
_FLAC_MAX_CHECKSUM_BYTES: Final = 1 << 18
_FLAC_SYNC: Final = re.compile(re.escape(_FLAC_FIXED_BLOCK_SYNC))


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


def _adts_frame_bytes(header: bytes) -> int | None:
    length: Final = (header[3] & 3) << 11 | header[4] << 3 | header[5] >> 5
    valid: Final = (
        header[0] == 0xFF
        and header[1] & 0xF6 == 0xF0
        and length >= _ADTS_HEADER_BYTES
        and header[2] >> 2 & 0xF < len(_ADTS_SAMPLE_RATES)
    )
    return length if valid else None


class _AdtsFrames:
    """Walks AAC ADTS frame headers as the audio arrives, since only they say how much audio the stream holds."""

    def __init__(self) -> None:
        self._carry = b""
        self._skip = 0
        self._blocks = 0
        self._sample_rate = 0
        self._broken = False

    def feed(self, audio: bytes) -> None:
        if self._broken:
            return
        data: Final = self._carry + audio
        at = self._skip  # rebind-ok: each frame says where the next one starts
        while at + _ADTS_HEADER_BYTES <= len(data):
            header = data[at : at + _ADTS_HEADER_BYTES]
            length = _adts_frame_bytes(header)
            if length is None:
                self._broken = True
                return
            self._sample_rate = _ADTS_SAMPLE_RATES[header[2] >> 2 & 0xF]
            self._blocks += (header[6] & 3) + 1
            at += length
        self._carry = data[at:]
        self._skip = max(at - len(data), 0)

    def seconds(self) -> Fraction | None:
        whole_frames: Final = not (self._broken or self._carry or self._skip)
        if not whole_frames or self._blocks == 0:
            return None
        return Fraction(self._blocks * _AAC_SAMPLES_PER_BLOCK - _AAC_ENCODER_DELAY_SAMPLES, self._sample_rate)


def _flac_crc16_shift(crc: int, _bit: int) -> int:
    return (crc << 1 ^ 0x8005 if crc & 0x8000 else crc << 1) & 0xFFFF


_FLAC_CRC16_TABLE: Final = tuple(reduce(_flac_crc16_shift, range(8), byte << 8) for byte in range(256))


def _flac_crc16_step(crc: int, byte: int) -> int:
    return crc << 8 & 0xFFFF ^ _FLAC_CRC16_TABLE[crc >> 8 ^ byte]


def _flac_crc16(data: bytes) -> int:
    return reduce(_flac_crc16_step, data, 0)


def _flac_block_size(code: int, extra: bytes) -> int | None:
    if code == 0:
        return None
    if code == 1:
        return 192
    if code <= 5:
        return 576 << code - 2
    if code <= 7:
        return int.from_bytes(extra[: code - 5], "big") + 1
    return 256 << code - 8


def _flac_number_step(high: int, byte: int) -> int:
    return high << 6 | byte & 0x3F


def _flac_frame_end_samples(frame: bytes, block_size: int) -> int | None:
    """Samples played once this fixed-block frame, the last of its stream, ends. None unless its CRC holds."""
    if len(frame) < _FLAC_MIN_FRAME_BYTES:
        return None
    leading_ones: Final = 8 - (frame[4] ^ 0xFF).bit_length()
    number_bytes: Final = max(leading_ones, 1)
    size_code: Final = frame[2] >> 4
    size_at: Final = 4 + number_bytes
    last_block: Final = _flac_block_size(size_code, frame[size_at : size_at + 2])
    if last_block is None or last_block > block_size or leading_ones in (1, 8):
        return None
    if _flac_crc16(frame[:-2]) != int.from_bytes(frame[-2:], "big"):
        return None
    number: Final = reduce(_flac_number_step, frame[5:size_at], frame[4] & 0xFF >> leading_ones)
    return number * block_size + last_block


def _flac_seconds(head: bytes, tail: bytes) -> Fraction | None:
    block_size: Final = int.from_bytes(head[10:12], "big")
    sample_rate: Final = int.from_bytes(head[18:21], "big") >> 4
    fixed_blocks: Final = head[:4] == b"fLaC" and head[4] & 0x7F == 0 and head[8:10] == head[10:12]
    if not fixed_blocks or block_size == 0 or sample_rate == 0:
        return None
    candidates: Final = tuple(len(tail) - found.start() for found in reversed(tuple(_FLAC_SYNC.finditer(tail))))
    samples: Final = next(
        (
            played
            for length, checksummed in zip(candidates, accumulate(candidates))
            if checksummed <= _FLAC_MAX_CHECKSUM_BYTES
            and (played := _flac_frame_end_samples(tail[-length:], block_size)) is not None
        ),
        None,
    )
    return None if samples is None else Fraction(samples, sample_rate)


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
        self._adts: Final = _AdtsFrames() if self._format == "aac" else None

    def feed(self, audio: bytes) -> None:
        if self._adts is not None:
            self._adts.feed(audio)
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
            case "flac":
                return _flac_seconds(self._head, self._tail)
            case "aac":
                return None if self._adts is None else self._adts.seconds()
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
