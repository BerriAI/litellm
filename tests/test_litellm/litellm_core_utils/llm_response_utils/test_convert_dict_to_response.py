"""
Tests for non-streaming response conversion in
``litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response``.

Regression coverage for transcription token-usage objects that omit
``input_token_details``. OpenAI-compatible transcription servers (for example
llama.cpp with ``response_format=json``) return a ``tokens`` usage block
without ``input_token_details``, which previously raised a pydantic
ValidationError because the field was required.
"""

import pytest

from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    convert_to_model_response_object,
)
from litellm.types.utils import (
    TranscriptionResponse,
    TranscriptionUsageTokensObject,
)


def test_transcription_token_usage_without_input_token_details():
    response_object = {
        "text": "the quick brown fox",
        "usage": {
            "type": "tokens",
            "input_tokens": 5,
            "output_tokens": 10,
            "total_tokens": 15,
        },
    }

    result = convert_to_model_response_object(
        response_object=response_object,
        response_type="audio_transcription",
    )

    assert isinstance(result, TranscriptionResponse)
    assert result.text == "the quick brown fox"
    assert isinstance(result.usage, TranscriptionUsageTokensObject)
    assert result.usage.input_token_details is None


def test_transcription_token_usage_with_input_token_details():
    response_object = {
        "text": "hello world",
        "usage": {
            "type": "tokens",
            "input_tokens": 5,
            "output_tokens": 10,
            "total_tokens": 15,
            "input_token_details": {"audio_tokens": 2, "text_tokens": 3},
        },
    }

    result = convert_to_model_response_object(
        response_object=response_object,
        response_type="audio_transcription",
    )

    assert isinstance(result.usage, TranscriptionUsageTokensObject)
    assert result.usage.input_token_details is not None
    assert result.usage.input_token_details.audio_tokens == 2
    assert result.usage.input_token_details.text_tokens == 3
