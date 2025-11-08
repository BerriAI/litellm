import pytest
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    convert_to_model_response_object,
)
from litellm import TranscriptionResponse, Usage


def test_convert_audio_transcription_with_usage():
    """
    Tests if the usage object is correctly parsed for an audio_transcription response.
    """
    # Mock response object from a provider like OpenAI for transcription
    mock_response_object = {
        "text": "This is a test transcription.",
        "usage": {
            "prompt_tokens": 784,
            "completion_tokens": 235,
            "total_tokens": 1019,
        },
    }

    # Call the function to convert the dictionary to a response object
    converted_response = convert_to_model_response_object(
        response_object=mock_response_object,
        response_type="audio_transcription",
        model_response_object=TranscriptionResponse(),  # Pass a base object to be populated
    )

    # Assert: Check if the usage information was correctly populated
    assert converted_response is not None
    assert isinstance(converted_response, TranscriptionResponse)
    assert hasattr(converted_response, "usage")
    assert isinstance(converted_response.usage, Usage)

    # Assert token values
    assert converted_response.usage.prompt_tokens == 784
    assert converted_response.usage.completion_tokens == 235
    assert converted_response.usage.total_tokens == 1019

    # Assert other fields
    assert converted_response.text == "This is a test transcription."


def test_convert_audio_transcription_without_usage():
    """
    Tests graceful handling when the usage object is missing from an audio_transcription response.
    """
    # Mock response object without a usage key
    mock_response_object = {
        "text": "This is another test transcription.",
    }

    converted_response = convert_to_model_response_object(
        response_object=mock_response_object,
        response_type="audio_transcription",
        model_response_object=TranscriptionResponse(),
    )

    # Assert: Check that the usage object defaults to zero values
    assert converted_response is not None
    assert isinstance(converted_response, TranscriptionResponse)
    assert hasattr(converted_response, "usage")  # The attribute should still exist
    assert isinstance(converted_response.usage, Usage)

    # Assert default token values
    assert converted_response.usage.prompt_tokens == 0
    assert converted_response.usage.completion_tokens == 0
    assert converted_response.usage.total_tokens == 0

    assert converted_response.text == "This is another test transcription."
