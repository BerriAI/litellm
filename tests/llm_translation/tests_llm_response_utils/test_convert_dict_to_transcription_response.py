import sys
import os
from datetime import datetime, timedelta
import pytest

sys.path.insert(0, os.path.abspath("../../"))

from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_transcription_response import (
    convert_dict_to_transcription_response,
    _return_verbose_transcription_response,
)
from litellm.types.utils import TranscriptionResponse, VerboseTranscriptionResponse


def test_convert_dict_to_transcription_response_basic():
    """Test basic conversion with minimal fields."""
    response_object = {
        "text": "Imagine the wildest idea that you've ever had, and you're curious about how it might scale to something that's a 100, a 1,000 times bigger. This is a place where you can get to do that."
    }
    result = convert_dict_to_transcription_response(
        model_response_object=None,
        response_object=response_object,
        hidden_params=None,
        _response_headers=None,
    )

    print("result from convert_dict_to_transcription_response_basic", result)

    assert isinstance(result, TranscriptionResponse)

    """
    Expected Response:
    {
        "text": "Four score and seven years ago, our fathers brought forth on this continent a new nation, conceived in liberty and dedicated to the proposition that all men are created equal. Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure."
    }
    """
    assert result.text == response_object["text"]
    assert not hasattr(result, "language")
    assert not hasattr(result, "task")
    assert not hasattr(result, "duration")
    assert not hasattr(result, "segments")


def test_convert_dict_to_transcription_response_full():
    """Test conversion with all fields present."""
    response_object = {
        "task": "transcribe",
        "language": "english",
        "duration": 8.470000267028809,
        "text": "The beach was a popular spot on a hot summer day. People were swimming in the ocean, building sandcastles, and playing beach volleyball.",
        "segments": [
            {
                "id": 0,
                "seek": 0,
                "start": 0.0,
                "end": 3.319999933242798,
                "text": " The beach was a popular spot on a hot summer day.",
                "tokens": [
                    50364,
                    440,
                    7534,
                    390,
                    257,
                    3743,
                    4008,
                    322,
                    257,
                    2368,
                    4266,
                    786,
                    13,
                    50530,
                ],
                "temperature": 0.0,
                "avg_logprob": -0.2860786020755768,
                "compression_ratio": 1.2363636493682861,
                "no_speech_prob": 0.00985979475080967,
            }
        ],
    }
    result = convert_dict_to_transcription_response(
        model_response_object=None,
        response_object=response_object,
        hidden_params=None,
        _response_headers=None,
    )

    assert isinstance(result, TranscriptionResponse)
    assert result.text == response_object["text"]
    assert result.task == response_object["task"]
    assert result.language == response_object["language"]
    assert result.duration == response_object["duration"]
    assert result.segments == response_object["segments"]


def test_convert_dict_to_transcription_response_with_response_headers():
    """Test conversion with response headers."""
    response_object = {"text": "Sample text"}
    response_headers = {"Content-Type": "application/json"}

    result = convert_dict_to_transcription_response(
        model_response_object=None,
        response_object=response_object,
        hidden_params=None,
        _response_headers=response_headers,
    )

    assert result._response_headers == response_headers


def test_convert_dict_to_transcription_response_none_response():
    """Test error handling for None response object."""
    with pytest.raises(Exception, match="Error in response object format"):
        convert_dict_to_transcription_response(
            model_response_object=None,
            response_object=None,
            hidden_params=None,
            _response_headers=None,
        )


def test_return_verbose_transcription_response():
    """Test _return_verbose_transcription_response function."""
    # Test with all optional keys present
    response_object = {
        "language": "en",
        "task": "transcribe",
        "duration": 10.5,
        "words": [{"word": "hello", "start": 0.0, "end": 0.5}],
        "segments": [{"id": 0, "text": "Hello world", "start": 0.0, "end": 1.0}],
        "text": "Hello world",
    }
    result = _return_verbose_transcription_response(response_object)

    assert isinstance(result, VerboseTranscriptionResponse)
    assert result.language == "en"
    assert result.task == "transcribe"
    assert result.duration == 10.5
    assert result.words == [{"word": "hello", "start": 0.0, "end": 0.5}]
    assert result.segments == [
        {"id": 0, "text": "Hello world", "start": 0.0, "end": 1.0}
    ]

    # Test with some optional keys missing
    response_object = {"language": "fr", "duration": 5.2, "text": "Bonjour le monde"}
    result = _return_verbose_transcription_response(response_object)

    assert isinstance(result, VerboseTranscriptionResponse)
    assert result.language == "fr"
    assert result.duration == 5.2
    assert hasattr(result, "task")
    assert hasattr(result, "words")
    assert hasattr(result, "segments")

    # Test with no optional keys
    response_object = {"text": "Just text"}
    result = _return_verbose_transcription_response(response_object)

    assert result is None
