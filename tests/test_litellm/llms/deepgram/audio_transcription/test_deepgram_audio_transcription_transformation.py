import io
import os
import pathlib
import sys
from unittest.mock import MagicMock

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
from litellm.types.utils import TranscriptionResponse


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


def test_transform_response_without_diarization():
    """Test response transformation without diarization"""
    handler = DeepgramAudioTranscriptionConfig()

    # Mock response without diarization
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "metadata": {
            "duration": 10.5,
        },
        "results": {
            "channels": [
                {
                    "alternatives": [
                        {
                            "transcript": "Hello this is a test.",
                            "confidence": 0.99,
                            "words": [
                                {"word": "Hello", "start": 0.0, "end": 0.5},
                                {"word": "this", "start": 0.6, "end": 0.8},
                                {"word": "is", "start": 0.9, "end": 1.1},
                                {"word": "a", "start": 1.2, "end": 1.3},
                                {"word": "test", "start": 1.4, "end": 1.8},
                            ],
                        }
                    ]
                }
            ]
        },
    }

    result = handler.transform_audio_transcription_response(mock_response)

    assert isinstance(result, TranscriptionResponse)
    assert result.text == "Hello this is a test."
    assert result["task"] == "transcribe"
    assert result["duration"] == 10.5
    assert len(result["words"]) == 5


def test_transform_response_with_diarization_and_paragraphs():
    """Test response transformation with diarization and paragraphs property"""
    handler = DeepgramAudioTranscriptionConfig()

    # Mock response with diarization and paragraphs
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "metadata": {
            "duration": 15.0,
        },
        "results": {
            "channels": [
                {
                    "alternatives": [
                        {
                            "transcript": "Hello how are you I am fine thanks",
                            "paragraphs": {
                                "transcript": "\nSpeaker 0: Hello how are you\n\nSpeaker 1: I am fine thanks\n"
                            },
                            "words": [
                                {"word": "Hello", "start": 0.0, "end": 0.5, "speaker": 0},
                                {"word": "how", "start": 0.6, "end": 0.8, "speaker": 0},
                                {"word": "are", "start": 0.9, "end": 1.1, "speaker": 0},
                                {"word": "you", "start": 1.2, "end": 1.3, "speaker": 0},
                                {"word": "I", "start": 2.0, "end": 2.2, "speaker": 1},
                                {"word": "am", "start": 2.3, "end": 2.5, "speaker": 1},
                                {"word": "fine", "start": 2.6, "end": 2.9, "speaker": 1},
                                {"word": "thanks", "start": 3.0, "end": 3.5, "speaker": 1},
                            ],
                        }
                    ]
                }
            ]
        },
    }

    result = handler.transform_audio_transcription_response(mock_response)

    assert isinstance(result, TranscriptionResponse)
    # Should use the pre-formatted paragraphs transcript
    assert result.text == "\nSpeaker 0: Hello how are you\n\nSpeaker 1: I am fine thanks\n"
    assert result["task"] == "transcribe"
    assert result["duration"] == 15.0


def test_transform_response_with_diarization_without_paragraphs():
    """Test response transformation with diarization but no paragraphs property"""
    handler = DeepgramAudioTranscriptionConfig()

    # Mock response with diarization but without paragraphs
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "metadata": {
            "duration": 15.0,
        },
        "results": {
            "channels": [
                {
                    "alternatives": [
                        {
                            "transcript": "Hello how are you I am fine thanks",
                            "words": [
                                {"word": "hello", "punctuated_word": "Hello", "start": 0.0, "end": 0.5, "speaker": 0},
                                {"word": "how", "punctuated_word": "how", "start": 0.6, "end": 0.8, "speaker": 0},
                                {"word": "are", "punctuated_word": "are", "start": 0.9, "end": 1.1, "speaker": 0},
                                {"word": "you", "punctuated_word": "you", "start": 1.2, "end": 1.3, "speaker": 0},
                                {"word": "i", "punctuated_word": "I", "start": 2.0, "end": 2.2, "speaker": 1},
                                {"word": "am", "punctuated_word": "am", "start": 2.3, "end": 2.5, "speaker": 1},
                                {"word": "fine", "punctuated_word": "fine", "start": 2.6, "end": 2.9, "speaker": 1},
                                {"word": "thanks", "punctuated_word": "thanks.", "start": 3.0, "end": 3.5, "speaker": 1},
                            ],
                        }
                    ]
                }
            ]
        },
    }

    result = handler.transform_audio_transcription_response(mock_response)

    assert isinstance(result, TranscriptionResponse)
    # Should reconstruct from words using punctuated_word
    expected_text = "Speaker 0: Hello how are you\n\nSpeaker 1: I am fine thanks.\n"
    assert result.text == expected_text
    assert result["task"] == "transcribe"
    assert result["duration"] == 15.0


def test_reconstruct_diarized_transcript_with_punctuated_words():
    """Test reconstruction uses punctuated_word when available"""
    handler = DeepgramAudioTranscriptionConfig()

    words = [
        {"word": "hello", "punctuated_word": "Hello", "speaker": 0},
        {"word": "world", "punctuated_word": "world!", "speaker": 0},
        {"word": "how", "punctuated_word": "How", "speaker": 1},
        {"word": "are", "punctuated_word": "are", "speaker": 1},
        {"word": "you", "punctuated_word": "you?", "speaker": 1},
    ]

    result = handler._reconstruct_diarized_transcript(words)

    # Check that punctuated_word is used and speakers are properly separated
    assert "Hello world!" in result
    assert "How are you?" in result
    assert "Speaker 0:" in result
    assert "Speaker 1:" in result


def test_reconstruct_diarized_transcript_fallback_to_word():
    """Test reconstruction falls back to 'word' when punctuated_word is missing"""
    handler = DeepgramAudioTranscriptionConfig()

    words = [
        {"word": "Hello", "speaker": 0},  # No punctuated_word
        {"word": "world", "speaker": 0},
        {"word": "test", "punctuated_word": "test.", "speaker": 1},  # Has punctuated_word
    ]

    result = handler._reconstruct_diarized_transcript(words)

    # Should use 'word' when punctuated_word is not available
    assert "Hello world" in result
    assert "test." in result
    assert "Speaker 0:" in result
    assert "Speaker 1:" in result


def test_reconstruct_diarized_transcript_empty_words():
    """Test reconstruction with empty words list"""
    handler = DeepgramAudioTranscriptionConfig()

    result = handler._reconstruct_diarized_transcript([])

    assert result == ""


def test_reconstruct_diarized_transcript_single_speaker():
    """Test reconstruction with single speaker"""
    handler = DeepgramAudioTranscriptionConfig()

    words = [
        {"word": "This", "punctuated_word": "This", "speaker": 0},
        {"word": "is", "punctuated_word": "is", "speaker": 0},
        {"word": "a", "punctuated_word": "a", "speaker": 0},
        {"word": "test", "punctuated_word": "test.", "speaker": 0},
    ]

    result = handler._reconstruct_diarized_transcript(words)

    # Should have only one speaker segment
    assert result.count("Speaker 0:") == 1
    assert "This is a test." in result


def test_reconstruct_diarized_transcript_multiple_speaker_changes():
    """Test reconstruction with multiple speaker changes"""
    handler = DeepgramAudioTranscriptionConfig()

    words = [
        {"word": "Hi", "speaker": 0},
        {"word": "there", "speaker": 0},
        {"word": "Hello", "speaker": 1},
        {"word": "back", "speaker": 0},  # Speaker 0 again
        {"word": "Thanks", "speaker": 1},  # Speaker 1 again
    ]

    result = handler._reconstruct_diarized_transcript(words)

    # Should have 4 speaker segments (0, 1, 0, 1)
    assert result.count("Speaker 0:") == 2
    assert result.count("Speaker 1:") == 2
    assert "Hi there" in result
    assert "Hello" in result
    assert "back" in result
    assert "Thanks" in result
