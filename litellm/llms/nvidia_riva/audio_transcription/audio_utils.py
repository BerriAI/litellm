"""
Audio resampling utilities for the NVIDIA Riva STT provider.

We intentionally avoid a hard dependency on ``ffmpeg`` so this works in
slim Python environments. Format coverage:

- ``soundfile`` handles wav / flac / ogg out of the box (libsndfile).
- ``audioread`` is tried for everything ``soundfile`` cannot decode (mp3,
  m4a, mp4, webm, ...). This is a soft optional dependency.

If neither library can decode the input we raise a clear error instructing
the caller to convert the audio upstream.
"""

import io
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Tuple, cast

from litellm.llms.nvidia_riva.audio_transcription.transformation import (
    RIVA_TARGET_NUM_CHANNELS,
    RIVA_TARGET_SAMPLE_RATE_HZ,
)
from litellm.llms.nvidia_riva.common_utils import NvidiaRivaException

# Keep this as Any: the module intentionally avoids importing numpy at module
# import time (optional dependency), and project-wide mypy config evaluates this
# file in contexts where conditional type aliases can degrade to "FloatArray?".
FloatArray = Any


_INSTALL_HINT = (
    "Install Riva STT extras to enable automatic audio resampling: "
    "`pip install 'litellm[stt-nvidia-riva]'`"
)


@dataclass
class ResampledAudio:
    pcm_bytes: bytes
    duration_seconds: float
    sample_rate_hz: int
    num_channels: int


def resample_to_riva_pcm(file_bytes: bytes) -> ResampledAudio:
    """
    Decode ``file_bytes`` and produce 16 kHz mono LINEAR_PCM (int16 little
    endian) suitable for streaming to Riva, plus the audio duration in
    seconds (used for cost calculation when Riva does not return usage).
    """
    try:
        import numpy as np  # type: ignore
    except ImportError as e:
        raise NvidiaRivaException(
            status_code=500,
            message=f"numpy is required for Riva audio resampling. {_INSTALL_HINT}",
        ) from e

    samples_float, source_rate = _decode_to_float32(file_bytes)

    # Downmix to mono by averaging channels.
    if samples_float.ndim == 2 and samples_float.shape[1] > 1:
        samples_float = samples_float.mean(axis=1)
    elif samples_float.ndim == 2:
        samples_float = samples_float[:, 0]

    samples_float = np.asarray(samples_float, dtype=np.float32).ravel()

    if source_rate != RIVA_TARGET_SAMPLE_RATE_HZ:
        samples_float = _resample(
            samples_float, source_rate, RIVA_TARGET_SAMPLE_RATE_HZ
        )

    # Clip + convert float [-1, 1] to int16 little-endian PCM.
    np.clip(samples_float, -1.0, 1.0, out=samples_float)
    pcm_int16 = (samples_float * 32767.0).astype("<i2")
    pcm_bytes = pcm_int16.tobytes()

    duration_seconds = float(pcm_int16.size) / float(RIVA_TARGET_SAMPLE_RATE_HZ)

    return ResampledAudio(
        pcm_bytes=pcm_bytes,
        duration_seconds=duration_seconds,
        sample_rate_hz=RIVA_TARGET_SAMPLE_RATE_HZ,
        num_channels=RIVA_TARGET_NUM_CHANNELS,
    )


def _decode_to_float32(file_bytes: bytes) -> Tuple["FloatArray", int]:
    """
    Decode arbitrary audio bytes into a float32 array shaped either
    ``(n_samples,)`` (mono) or ``(n_samples, n_channels)`` plus the source
    sample rate.

    Tries ``soundfile`` first (wav/flac/ogg), then falls back to
    ``audioread`` for compressed formats. Raises a clear error if neither
    works.
    """
    import numpy as np  # type: ignore

    sf_error: Exception | None = None
    try:
        import soundfile as sf  # type: ignore

        with io.BytesIO(file_bytes) as buf:
            data, source_rate = sf.read(buf, dtype="float32", always_2d=False)
        return cast("FloatArray", data), int(source_rate)
    except ImportError as e:
        sf_error = e
    except Exception as e:
        # soundfile raises RuntimeError / LibsndfileError for formats it
        # cannot decode (mp3 on older libsndfile, m4a, webm, ...).
        sf_error = e

    try:
        import audioread  # type: ignore
    except ImportError as e:
        raise NvidiaRivaException(
            status_code=400,
            message=(
                "Could not decode audio for Riva STT. Install audio extras "
                f"(`pip install 'litellm[stt-nvidia-riva]'`) or convert your "
                f"audio to wav/flac/ogg before calling the API. "
                f"Underlying error: {sf_error}"
            ),
        ) from e

    # audioread backends (FFmpeg subprocess, GStreamer, Core Audio) require a
    # filesystem path, so spill the bytes to a temp file. mkstemp is portable
    # to Windows where re-opening a NamedTemporaryFile is not allowed.
    fd, tmp_path = tempfile.mkstemp(suffix=".audio")
    try:
        with os.fdopen(fd, "wb") as tmp_file:
            tmp_file.write(file_bytes)
        try:
            with audioread.audio_open(tmp_path) as src:
                source_rate = int(src.samplerate)
                channels = int(src.channels)
                chunks = []
                for buf in src:
                    chunks.append(np.frombuffer(buf, dtype=np.int16))
                if not chunks:
                    raise NvidiaRivaException(
                        status_code=400,
                        message="Audio decode produced no samples.",
                    )
                interleaved = np.concatenate(chunks).astype(np.float32) / 32768.0
                if channels > 1:
                    interleaved = interleaved.reshape(-1, channels)
                return cast("FloatArray", interleaved), source_rate
        except NvidiaRivaException:
            raise
        except Exception as e:
            raise NvidiaRivaException(
                status_code=400,
                message=(
                    "Could not decode audio for Riva STT. Convert your audio to "
                    f"wav/flac/ogg before calling the API. Underlying error: {e}"
                ),
            ) from e
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _resample(
    samples: "FloatArray", source_rate: int, target_rate: int
) -> "FloatArray":
    """
    Resample mono float32 ``samples`` from ``source_rate`` to ``target_rate``.

    Prefers high-quality polyphase resampling when ``soxr`` or ``scipy`` is
    available (anti-aliased, important for downsampling 44.1/48 kHz -> 16 kHz
    where naive interpolation folds high frequencies back into the speech
    band). Falls back to linear interpolation if neither is installed —
    acceptable for speech-only mono input but lossy for wideband content.
    """
    import numpy as np  # type: ignore

    if source_rate == target_rate or samples.size == 0:
        return samples

    try:
        import soxr  # type: ignore

        return cast(
            "FloatArray",
            np.asarray(
                soxr.resample(samples, source_rate, target_rate), dtype=np.float32
            ),
        )
    except ImportError:
        pass

    try:
        from math import gcd

        from scipy.signal import resample_poly  # type: ignore

        g = gcd(int(source_rate), int(target_rate))
        up = int(target_rate) // g
        down = int(source_rate) // g
        return cast(
            "FloatArray", np.asarray(resample_poly(samples, up, down), dtype=np.float32)
        )
    except ImportError:
        pass

    return _linear_resample(samples, source_rate, target_rate)


def _linear_resample(
    samples: "FloatArray", source_rate: int, target_rate: int
) -> "FloatArray":
    """Linear-interpolation fallback. See :func:`_resample` for caveats."""
    import numpy as np  # type: ignore

    duration = samples.size / float(source_rate)
    target_length = int(round(duration * target_rate))
    if target_length <= 1:
        return samples.astype(np.float32)

    src_indices = np.linspace(0, samples.size - 1, num=target_length, dtype=np.float64)
    left = np.floor(src_indices).astype(np.int64)
    right = np.minimum(left + 1, samples.size - 1)
    frac = (src_indices - left).astype(np.float32)

    return ((1.0 - frac) * samples[left] + frac * samples[right]).astype(np.float32)
