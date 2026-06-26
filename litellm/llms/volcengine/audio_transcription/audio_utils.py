import io
import os
import sys
import tempfile
import wave
from array import array
from dataclasses import dataclass
from typing import Any, cast

from litellm.llms.volcengine.common_utils import VolcEngineError

VOLCENGINE_STT_SAMPLE_RATE_HZ = 16000
VOLCENGINE_STT_NUM_CHANNELS = 1
VOLCENGINE_STT_CHUNK_BYTES = 6400

FloatArray = Any

_INSTALL_HINT = (
    "use uncompressed WAV or install audio decoding extras before calling "
    "Volcengine audio transcription."
)


@dataclass
class VolcEnginePcmAudio:
    pcm_bytes: bytes
    duration_seconds: float
    sample_rate_hz: int
    num_channels: int


def resample_to_volcengine_stt_pcm(file_bytes: bytes) -> VolcEnginePcmAudio:
    samples, source_rate = _decode_to_int16_mono(file_bytes)
    if source_rate != VOLCENGINE_STT_SAMPLE_RATE_HZ:
        samples = _resample_int16(samples, source_rate, VOLCENGINE_STT_SAMPLE_RATE_HZ)

    pcm_bytes = _pack_int16(samples)
    duration_seconds = float(len(samples)) / float(VOLCENGINE_STT_SAMPLE_RATE_HZ)

    return VolcEnginePcmAudio(
        pcm_bytes=pcm_bytes,
        duration_seconds=duration_seconds,
        sample_rate_hz=VOLCENGINE_STT_SAMPLE_RATE_HZ,
        num_channels=VOLCENGINE_STT_NUM_CHANNELS,
    )


def chunk_volcengine_stt_pcm(pcm_bytes: bytes) -> list[bytes]:
    if not pcm_bytes:
        return []
    return [
        pcm_bytes[i : i + VOLCENGINE_STT_CHUNK_BYTES]
        for i in range(0, len(pcm_bytes), VOLCENGINE_STT_CHUNK_BYTES)
    ]


def _decode_to_int16_mono(file_bytes: bytes) -> tuple[list[int], int]:
    try:
        return _decode_wav_to_int16_mono(file_bytes)
    except VolcEngineError:
        raise
    except (EOFError, wave.Error):
        return _decode_to_int16_mono_with_optional_deps(file_bytes)


def _decode_wav_to_int16_mono(file_bytes: bytes) -> tuple[list[int], int]:
    with wave.open(io.BytesIO(file_bytes), "rb") as wav:
        if wav.getcomptype() != "NONE":
            raise VolcEngineError(
                status_code=400,
                message="Volcengine STT requires uncompressed WAV without audio decoding extras.",
            )
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        source_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    samples = _wav_frames_to_int16(frames, sample_width)
    return _mixdown_int16(samples, channels), source_rate


def _wav_frames_to_int16(frames: bytes, sample_width: int) -> list[int]:
    if sample_width == 1:
        return [(sample - 128) << 8 for sample in frames]
    if sample_width == 2:
        samples = array("h")
        samples.frombytes(frames)
        if sys.byteorder != "little":
            samples.byteswap()
        return list(samples)
    if sample_width == 3:
        return [
            int.from_bytes(frames[i : i + 3], "little", signed=True) >> 8
            for i in range(0, len(frames) - 2, 3)
        ]
    if sample_width == 4:
        return [
            int.from_bytes(frames[i : i + 4], "little", signed=True) >> 16
            for i in range(0, len(frames) - 3, 4)
        ]
    raise VolcEngineError(
        status_code=400,
        message=f"Unsupported WAV sample width for Volcengine STT: {sample_width}",
    )


def _mixdown_int16(samples: list[int], channels: int) -> list[int]:
    if channels <= 0:
        raise VolcEngineError(
            status_code=400,
            message="WAV channel count must be positive for Volcengine STT.",
        )
    if channels == 1:
        return samples
    return [
        int(sum(samples[i : i + channels]) / channels)
        for i in range(0, len(samples) - channels + 1, channels)
    ]


def _decode_to_int16_mono_with_optional_deps(
    file_bytes: bytes,
) -> tuple[list[int], int]:
    try:
        import numpy as np  # type: ignore
        import soundfile as sf  # type: ignore
    except ImportError as e:
        raise VolcEngineError(
            status_code=400,
            message=f"Could not decode audio for Volcengine STT. {_INSTALL_HINT}",
        ) from e

    try:
        with io.BytesIO(file_bytes) as buf:
            data, source_rate = sf.read(buf, dtype="float32", always_2d=False)
        return _float_array_to_int16_mono(cast("FloatArray", data)), int(source_rate)
    except (OSError, RuntimeError, TypeError, ValueError):
        try:
            import audioread  # type: ignore
        except ImportError as e:
            raise VolcEngineError(
                status_code=400,
                message=f"Could not decode audio for Volcengine STT. {_INSTALL_HINT}",
            ) from e

        fd, tmp_path = tempfile.mkstemp(suffix=".audio")
        try:
            with os.fdopen(fd, "wb") as tmp_file:
                tmp_file.write(file_bytes)
            try:
                with audioread.audio_open(tmp_path) as src:
                    source_rate = int(src.samplerate)
                    channels = int(src.channels)
                    chunks = [np.frombuffer(buf, dtype=np.int16) for buf in src]
                    if not chunks:
                        raise VolcEngineError(
                            status_code=400,
                            message="Audio decode produced no samples.",
                        )
                    interleaved = np.concatenate(chunks).astype(np.float32) / 32768.0
                    if channels > 1:
                        interleaved = interleaved.reshape(-1, channels)
                    return (
                        _float_array_to_int16_mono(cast("FloatArray", interleaved)),
                        source_rate,
                    )
            except VolcEngineError:
                raise
            except (
                audioread.DecodeError,
                EOFError,
                OSError,
                RuntimeError,
                ValueError,
            ) as e:
                raise VolcEngineError(
                    status_code=400,
                    message=(
                        "Could not decode audio for Volcengine STT. Convert audio "
                        f"to wav/flac/ogg before calling the API. Underlying error: {e}"
                    ),
                ) from e
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _float_array_to_int16_mono(samples_float: "FloatArray") -> list[int]:
    import numpy as np  # type: ignore

    if samples_float.ndim == 2 and samples_float.shape[1] > 1:
        samples_float = samples_float.mean(axis=1)
    elif samples_float.ndim == 2:
        samples_float = samples_float[:, 0]

    samples_float = np.asarray(samples_float, dtype=np.float32).ravel()
    np.clip(samples_float, -1.0, 1.0, out=samples_float)
    return [int(sample * 32767.0) for sample in samples_float]


def _resample_int16(
    samples: list[int], source_rate: int, target_rate: int
) -> list[int]:
    if source_rate == target_rate or not samples:
        return samples
    target_size = max(1, round(len(samples) * float(target_rate) / float(source_rate)))
    if target_size == 1:
        return [samples[0]]
    source_last = len(samples) - 1
    target_last = target_size - 1
    resampled: list[int] = []
    for i in range(target_size):
        position = float(i * source_last) / float(target_last)
        left = int(position)
        right = min(left + 1, source_last)
        fraction = position - left
        value = samples[left] + (samples[right] - samples[left]) * fraction
        resampled.append(_clamp_int16(round(value)))
    return resampled


def _pack_int16(samples: list[int]) -> bytes:
    pcm = array("h", (_clamp_int16(sample) for sample in samples))
    if sys.byteorder != "little":
        pcm.byteswap()
    return pcm.tobytes()


def _clamp_int16(value: int) -> int:
    return max(-32768, min(32767, int(value)))
