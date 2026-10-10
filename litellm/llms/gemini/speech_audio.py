import struct
from collections.abc import Mapping
from typing import Final

GEMINI_TTS_CHAT_AUDIO_FORMAT: Final = "pcm16"
GEMINI_TTS_RAW_RESPONSE_FORMAT: Final = "pcm"
GEMINI_TTS_SUPPORTED_RESPONSE_FORMATS: Final = frozenset({"wav", GEMINI_TTS_RAW_RESPONSE_FORMAT})


def is_gemini_tts_model(model: str) -> bool:
    return "gemini" in model.lower() and ("tts" in model.lower() or "preview-tts" in model.lower())


def gemini_tts_chat_audio_format(model: str) -> str | None:
    return GEMINI_TTS_CHAT_AUDIO_FORMAT if is_gemini_tts_model(model) else None


def validate_gemini_tts_speech_format(
    model: str, custom_llm_provider: str, optional_params: Mapping[str, object]
) -> None:
    if not is_gemini_tts_model(model):
        return
    response_format: Final = optional_params.get("response_format")
    if not isinstance(response_format, str) or response_format in GEMINI_TTS_SUPPORTED_RESPONSE_FORMATS:
        return
    from litellm.exceptions import BadRequestError

    supported: Final = ", ".join(sorted(GEMINI_TTS_SUPPORTED_RESPONSE_FORMATS))
    raise BadRequestError(
        message=(
            f"Gemini TTS only produces raw PCM16 audio, so response_format='{response_format}'"
            f" is not supported. Supported response formats: {supported}."
        ),
        model=model,
        llm_provider=custom_llm_provider,
    )


def convert_pcm16_to_wav(pcm_data: bytes, sample_rate: int = 24000, channels: int = 1) -> bytes:
    byte_rate: Final = sample_rate * channels * 2
    block_align: Final = channels * 2
    data_size: Final = len(pcm_data)
    file_size: Final = 36 + data_size
    wav_header: Final = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        file_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        16,
        b"data",
        data_size,
    )
    return wav_header + pcm_data


def pcm16_payload(audio: bytes) -> bytes:
    if not _is_wav(audio):
        return audio
    payload: Final = _wav_data_payload(audio)
    return audio if payload is None else payload


def gemini_tts_speech_response(
    model: str, decoded_audio: bytes, response_format: str | None
) -> tuple[bytes, str] | None:
    if not is_gemini_tts_model(model):
        return None
    if response_format == GEMINI_TTS_RAW_RESPONSE_FORMAT:
        return pcm16_payload(decoded_audio), "audio/pcm"
    if _is_wav(decoded_audio):
        return decoded_audio, "audio/wav"
    return convert_pcm16_to_wav(decoded_audio), "audio/wav"


def _is_wav(audio: bytes) -> bool:
    return audio.startswith(b"RIFF") and audio[8:12] == b"WAVE"


def _wav_data_payload(audio: bytes) -> bytes | None:
    offset = 12  # rebind-ok: walk RIFF chunk headers
    while offset + 8 <= len(audio):
        chunk_id: Final = audio[offset : offset + 4]
        chunk_size: Final = int.from_bytes(audio[offset + 4 : offset + 8], "little")
        start: Final = offset + 8
        end: Final = start + chunk_size
        if end > len(audio):
            return None
        if chunk_id == b"data":
            return audio[start:end]
        offset = end + (chunk_size % 2)
    return None
