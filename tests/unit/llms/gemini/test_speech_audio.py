import struct
from typing import Final

from litellm.llms.gemini.speech_audio import (
    convert_pcm16_to_wav,
    gemini_tts_speech_response,
    pcm16_payload,
)

GEMINI_TTS_MODEL: Final = "gemini-3.1-flash-tts-preview"
PCM_BYTES: Final = b"\x01\x02\x03\x04" * 6


def _chunk(chunk_id: bytes, payload: bytes) -> bytes:
    pad: Final = b"\x00" if len(payload) % 2 else b""
    return chunk_id + struct.pack("<I", len(payload)) + payload + pad


def _wav(*chunks: bytes) -> bytes:
    body: Final = b"".join(chunks)
    return b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body


def _fmt_chunk() -> bytes:
    return struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, 1, 24000, 48000, 2, 16)


def _wav_with_list_before_data(pcm: bytes) -> bytes:
    list_chunk: Final = _chunk(b"LIST", b"INFO" + _chunk(b"INAM", b"database"))
    return _wav(_fmt_chunk(), list_chunk, _chunk(b"data", pcm))


def test_pcm16_payload_returns_non_wav_bytes_unchanged() -> None:
    assert pcm16_payload(PCM_BYTES) == PCM_BYTES


def test_pcm16_payload_reads_data_chunk_after_list_metadata() -> None:
    wav: Final = _wav_with_list_before_data(PCM_BYTES)
    naive_payload: Final = wav[wav.find(b"data", 12) + 8 :]

    assert naive_payload != PCM_BYTES
    assert pcm16_payload(wav) == PCM_BYTES


def test_pcm16_payload_excludes_chunks_after_audio_data() -> None:
    wav: Final = _wav(_fmt_chunk(), _chunk(b"data", PCM_BYTES), _chunk(b"JUNK", b"xxxx"))

    assert pcm16_payload(wav) == PCM_BYTES


def test_pcm16_payload_falls_back_when_wav_has_no_data_chunk() -> None:
    wav: Final = _wav(_fmt_chunk())

    assert pcm16_payload(wav) == wav


def test_pcm16_payload_falls_back_when_data_chunk_is_truncated() -> None:
    header: Final = _fmt_chunk() + b"data" + struct.pack("<I", 100) + b"\x00\x01"
    wav: Final = b"RIFF" + struct.pack("<I", 4 + len(header)) + b"WAVE" + header

    assert pcm16_payload(wav) == wav


def test_pcm16_payload_round_trips_bytes_from_convert_pcm16_to_wav() -> None:
    assert pcm16_payload(convert_pcm16_to_wav(PCM_BYTES)) == PCM_BYTES


def test_gemini_tts_pcm_response_strips_wav_with_list_chunk() -> None:
    wav: Final = _wav_with_list_before_data(PCM_BYTES)
    body, content_type = gemini_tts_speech_response(GEMINI_TTS_MODEL, wav, "pcm")

    assert body == PCM_BYTES
    assert content_type == "audio/pcm"


def test_gemini_tts_speech_response_is_none_for_non_gemini_models() -> None:
    assert gemini_tts_speech_response("gpt-4o-audio-preview", PCM_BYTES, "wav") is None
