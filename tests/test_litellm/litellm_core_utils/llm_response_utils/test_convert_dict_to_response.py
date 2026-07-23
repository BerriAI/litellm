"""
Tests for ``convert_to_model_response_object`` transcription usage handling in
``litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response``.
"""

from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    convert_to_model_response_object,
)


def test_transcription_usage_tokens_allows_none_input_token_details():
    """Servers like llama.cpp (response_format=json) return token usage without the
    per-modality input_token_details breakdown, so it must be accepted as None.

    Regression for https://github.com/BerriAI/litellm/issues/33764
    """
    response_object = {
        "text": "the transcription text",
        "usage": {
            "type": "tokens",
            "input_tokens": 14,
            "output_tokens": 45,
            "total_tokens": 59,
            "input_token_details": None,
        },
    }

    result = convert_to_model_response_object(
        response_object=response_object,
        response_type="audio_transcription",
    )

    assert result.text == "the transcription text"
    assert result.usage.type == "tokens"
    assert result.usage.total_tokens == 59
    assert result.usage.input_token_details is None


def test_transcription_usage_tokens_preserves_input_token_details():
    """When the server does provide input_token_details, it is still parsed."""
    response_object = {
        "text": "hello",
        "usage": {
            "type": "tokens",
            "input_tokens": 14,
            "output_tokens": 45,
            "total_tokens": 59,
            "input_token_details": {"audio_tokens": 14, "text_tokens": 0},
        },
    }

    result = convert_to_model_response_object(
        response_object=response_object,
        response_type="audio_transcription",
    )

    assert result.usage.input_token_details.audio_tokens == 14
    assert result.usage.input_token_details.text_tokens == 0
