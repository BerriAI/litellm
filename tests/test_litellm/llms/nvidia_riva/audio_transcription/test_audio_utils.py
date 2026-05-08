"""
Tests for the NVIDIA Riva audio resampling utility.

The resampler turns arbitrary inbound audio (mp3/wav/m4a/...) into the wire
format Riva's gRPC ASR expects: 16 kHz mono LINEAR_PCM (int16 LE).
"""

import io
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

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


def test_audioread_fallback_writes_to_tempfile_path(monkeypatch):
    """
    The audioread fallback handles compressed formats (mp3, m4a, ...). Most
    audioread backends call into a subprocess (FFmpeg, GStreamer) and
    require a real filesystem path — passing a BytesIO blows up with a
    TypeError in subprocess.Popen. This test would have caught that bug:
    we assert ``audio_open`` is called with a string path that points at a
    file containing exactly the input bytes.
    """
    payload = b"\xff\xfbfake-mp3-bytes-not-actually-decodable"
    seen_paths = []

    class FakeAudioSource:
        samplerate = 22050
        channels = 1

        def __iter__(self):
            yield np.array([0, 0, 0, 0], dtype=np.int16).tobytes()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_audio_open(path):
        assert isinstance(path, str), "audioread requires a filesystem path"
        seen_paths.append(path)
        with open(path, "rb") as fh:
            assert fh.read() == payload
        return FakeAudioSource()

    fake_audioread = SimpleNamespace(audio_open=fake_audio_open)
    monkeypatch.setitem(sys.modules, "audioread", fake_audioread)

    fake_sf = MagicMock()
    fake_sf.read.side_effect = RuntimeError("libsndfile cannot decode mp3")
    monkeypatch.setitem(sys.modules, "soundfile", fake_sf)

    resampled = resample_to_riva_pcm(payload)
    assert resampled.sample_rate_hz == 16000
    assert seen_paths and seen_paths[0].endswith(".audio")
    # Tempfile must be cleaned up after decode.
    assert not os.path.exists(seen_paths[0])
