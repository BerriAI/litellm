import gzip
import io
import json
import os
import sys
import wave
from array import array

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.volcengine.audio_transcription.audio_utils import (
    resample_to_volcengine_stt_pcm,
)
from litellm.llms.volcengine.audio_transcription.sauc_protocol import (
    COMP_GZIP,
    FLAG_NEG_SEQ,
    FLAG_POS_SEQ,
    MSG_AUDIO_CLIENT,
    MSG_FULL_CLIENT,
    SER_JSON,
    SER_RAW,
    decode_sauc_frame,
    encode_sauc_audio_chunk,
    encode_sauc_json_config,
)


def test_sauc_json_config_round_trip():
    payload = {"audio": {"rate": 16000}, "request": {"result_type": "single"}}

    frame = decode_sauc_frame(encode_sauc_json_config(payload))

    assert frame.message_type == MSG_FULL_CLIENT
    assert frame.flags == FLAG_POS_SEQ
    assert frame.serialization == SER_JSON
    assert frame.compression == COMP_GZIP
    assert frame.sequence == 1
    assert json.loads(gzip.decompress(frame.payload).decode("utf-8")) == payload


def test_sauc_audio_final_frame_round_trip():
    pcm = b"\x01\x00\x02\x00"

    frame = decode_sauc_frame(encode_sauc_audio_chunk(pcm=pcm, sequence=-3, last=True))

    assert frame.message_type == MSG_AUDIO_CLIENT
    assert frame.flags == FLAG_NEG_SEQ
    assert frame.serialization == SER_RAW
    assert frame.sequence == -3
    assert gzip.decompress(frame.payload) == pcm


def test_wav_audio_is_resampled_to_volcengine_pcm_without_optional_deps():
    wav_bytes = io.BytesIO()
    samples = array("h")
    for i in range(240):
        samples.extend([i, -i])
    with wave.open(wav_bytes, "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(samples.tobytes())

    audio = resample_to_volcengine_stt_pcm(wav_bytes.getvalue())

    assert audio.sample_rate_hz == 16000
    assert audio.num_channels == 1
    assert audio.duration_seconds == pytest.approx(0.01)
    assert len(audio.pcm_bytes) == 160 * 2
