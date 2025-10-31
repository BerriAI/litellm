import io
import os
import pathlib
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.llms.base_llm.audio_transcription.transformation import (
    AudioTranscriptionRequestData,
)
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
    result = handler.transform_audio_transcription_request(
        model="deepseek-audio-transcription",
        audio_file=audio_file,
        optional_params={},
        litellm_params={},
    )

    # Check that result is AudioTranscriptionRequestData
    assert isinstance(result, AudioTranscriptionRequestData)

    # Check that data matches expected output
    assert result.data == expected_output

    # Check that files is None for Deepgram (binary data)
    assert result.files is None


def test_get_complete_url_basic():
    """Test basic URL generation without optional parameters"""
    handler = DeepgramAudioTranscriptionConfig()
    url = handler.get_complete_url(
        api_base=None,
        api_key=None,
        model="nova-2",
        optional_params={},
        litellm_params={},
    )
    expected_url = "https://api.deepgram.com/v1/listen?model=nova-2"
    assert url == expected_url


def test_get_complete_url_with_punctuate():
    """Test URL generation with punctuate parameter"""
    handler = DeepgramAudioTranscriptionConfig()
    url = handler.get_complete_url(
        api_base=None,
        api_key=None,
        model="nova-2",
        optional_params={"punctuate": True},
        litellm_params={},
    )
    expected_url = "https://api.deepgram.com/v1/listen?model=nova-2&punctuate=true"
    assert url == expected_url


def test_get_complete_url_with_diarize():
    """Test URL generation with diarize parameter"""
    handler = DeepgramAudioTranscriptionConfig()
    url = handler.get_complete_url(
        api_base=None,
        api_key=None,
        model="nova-2",
        optional_params={"diarize": True},
        litellm_params={},
    )
    expected_url = "https://api.deepgram.com/v1/listen?model=nova-2&diarize=true"
    assert url == expected_url


def test_get_complete_url_with_measurements():
    """Test URL generation with measurements parameter"""
    handler = DeepgramAudioTranscriptionConfig()
    url = handler.get_complete_url(
        api_base=None,
        api_key=None,
        model="nova-2",
        optional_params={"measurements": True},
        litellm_params={},
    )
    expected_url = "https://api.deepgram.com/v1/listen?model=nova-2&measurements=true"
    assert url == expected_url


def test_get_complete_url_with_multiple_params():
    """Test URL generation with multiple query parameters"""
    handler = DeepgramAudioTranscriptionConfig()
    url = handler.get_complete_url(
        api_base=None,
        api_key=None,
        model="nova-2",
        optional_params={
            "punctuate": True,
            "diarize": False,
            "measurements": True,
            "smart_format": True,
        },
        litellm_params={},
    )
    # URL should contain all parameters
    assert "model=nova-2" in url
    assert "punctuate=true" in url
    assert "diarize=false" in url
    assert "measurements=true" in url
    assert "smart_format=true" in url
    assert url.startswith("https://api.deepgram.com/v1/listen?")


def test_get_complete_url_with_language_parameter():
    """Test that language parameter is excluded from query string (handled separately)"""
    handler = DeepgramAudioTranscriptionConfig()
    url = handler.get_complete_url(
        api_base=None,
        api_key=None,
        model="nova-2",
        optional_params={
            "language": "en",
            "punctuate": True,
        },
        litellm_params={},
    )
    expected_url = "https://api.deepgram.com/v1/listen?model=nova-2&punctuate=true"
    assert url == expected_url
    # Language should NOT appear in URL as it's handled separately
    assert "language=" not in url


def test_get_complete_url_with_custom_api_base():
    """Test URL generation with custom API base"""
    handler = DeepgramAudioTranscriptionConfig()
    url = handler.get_complete_url(
        api_base="https://custom.deepgram.com/v2",
        api_key=None,
        model="nova-2",
        optional_params={"punctuate": True},
        litellm_params={},
    )
    expected_url = "https://custom.deepgram.com/v2/listen?model=nova-2&punctuate=true"
    assert url == expected_url


def test_get_complete_url_with_string_values():
    """Test URL generation with string parameter values"""
    handler = DeepgramAudioTranscriptionConfig()
    url = handler.get_complete_url(
        api_base=None,
        api_key=None,
        model="nova-2",
        optional_params={
            "tier": "enhanced",
            "version": "latest",
            "punctuate": True,
        },
        litellm_params={},
    )
    # URL should contain all parameters
    assert "model=nova-2" in url
    assert "tier=enhanced" in url
    assert "version=latest" in url
    assert "punctuate=true" in url
    assert url.startswith("https://api.deepgram.com/v1/listen?")


def test_get_complete_url_with_detect_language():
    """Test URL generation with detect_language parameter"""
    handler = DeepgramAudioTranscriptionConfig()
    url = handler.get_complete_url(
        api_base=None,
        api_key=None,
        model="nova-2",
        optional_params={"detect_language": True},
        litellm_params={},
    )
    expected_url = "https://api.deepgram.com/v1/listen?model=nova-2&detect_language=true"
    assert url == expected_url


def test_get_complete_url_with_detect_language_and_other_params():
    """Test URL generation with detect_language and other parameters"""
    handler = DeepgramAudioTranscriptionConfig()
    url = handler.get_complete_url(
        api_base=None,
        api_key=None,
        model="nova-2",
        optional_params={
            "detect_language": True,
            "punctuate": True,
            "diarize": False,
        },
        litellm_params={},
    )
    # URL should contain all parameters
    assert "model=nova-2" in url
    assert "detect_language=true" in url
    assert "punctuate=true" in url
    assert "diarize=false" in url
    assert url.startswith("https://api.deepgram.com/v1/listen?")
