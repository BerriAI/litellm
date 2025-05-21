import io
import os
import pathlib
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.llms.deepgram.audio_transcription.transformation import (
    DeepgramAudioTranscriptionConfig,
)


@pytest.fixture
def test_bytes():
    return b"litellm", b"litellm"


@pytest.fixture
def test_io_bytes(test_bytes):
    return io.BytesIO(test_bytes[0]), test_bytes[1]


@pytest.fixture
def test_file():
    pwd = os.path.dirname(os.path.realpath(__file__))
    pwd_path = pathlib.Path(pwd)
    test_root = pwd_path.parents[3]
    print(f"test_root: {test_root}")
    file_path = os.path.join(test_root, "gettysburg.wav")
    f = open(file_path, "rb")
    content = f.read()
    f.seek(0)
    return f, content


@pytest.mark.parametrize(
    "fixture_name",
    [
        "test_bytes",
        "test_io_bytes",
        "test_file",
    ],
)
def test_audio_file_handling(fixture_name, request):
    handler = DeepgramAudioTranscriptionConfig()
    (audio_file, expected_output) = request.getfixturevalue(fixture_name)
    assert expected_output == handler.transform_audio_transcription_request(
        model="deepseek-audio-transcription",
        audio_file=audio_file,
        optional_params={},
        litellm_params={},
    )
