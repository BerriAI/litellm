"""
Tests for the NVIDIA Riva audio resampling utility.

The resampler turns arbitrary inbound audio (mp3/wav/m4a/...) into the wire
format Riva's gRPC ASR expects: 16 kHz mono LINEAR_PCM (int16 LE).
"""

import io
import os
import sys

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.nvidia_riva.audio_transcription.audio_utils import (
    resample_to_riva_pcm,
)
from litellm.llms.nvidia_riva.common_utils import NvidiaRivaException


def _wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_resample_24khz_stereo_to_16khz_mono_int16():
    sample_rate_in = 24000
    duration_seconds = 1.0
    n = int(sample_rate_in * duration_seconds)
    t = np.linspace(0, duration_seconds, n, endpoint=False)
    left = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    right = 0.5 * np.sin(2 * np.pi * 660.0 * t)
    stereo = np.stack([left, right], axis=1).astype(np.float32)

    wav_in = _wav_bytes(stereo, sample_rate_in)

    resampled = resample_to_riva_pcm(wav_in)

    assert resampled.sample_rate_hz == 16000
    assert resampled.num_channels == 1
    # int16 = 2 bytes per sample
    expected_samples = int(round(duration_seconds * 16000))
    assert len(resampled.pcm_bytes) == expected_samples * 2
    assert resampled.duration_seconds == pytest.approx(duration_seconds, abs=0.005)


def test_resample_16khz_mono_passes_through_int16_bytes_match_length():
    sample_rate = 16000
    n = sample_rate
    samples = (0.1 * np.sin(np.linspace(0, 2 * np.pi * 200, n))).astype(np.float32)
    wav_in = _wav_bytes(samples, sample_rate)

    resampled = resample_to_riva_pcm(wav_in)

    assert resampled.sample_rate_hz == 16000
    assert len(resampled.pcm_bytes) == n * 2
    assert resampled.duration_seconds == pytest.approx(1.0, abs=0.001)


def test_resample_preserves_int16_clip_range():
    sample_rate = 16000
    samples = np.array([2.0, -2.0, 0.0, 1.0], dtype=np.float32)
    wav_in = _wav_bytes(samples, sample_rate)

    resampled = resample_to_riva_pcm(wav_in)

    decoded = np.frombuffer(resampled.pcm_bytes, dtype="<i2")
    # Anything outside [-1, 1] should clip to int16 boundary.
    assert decoded.max() <= 32767
    assert decoded.min() >= -32767


def test_unknown_format_raises_clear_error():
    # 4 random bytes are not valid audio in any container we can decode.
    with pytest.raises(NvidiaRivaException) as excinfo:
        resample_to_riva_pcm(b"\x00\x01\x02\x03")
    # Message must hint at what to do next.
    assert "Riva STT" in excinfo.value.message
