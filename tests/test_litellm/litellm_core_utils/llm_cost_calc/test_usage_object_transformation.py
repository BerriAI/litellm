"""
Tests for
``litellm.litellm_core_utils.llm_cost_calc.usage_object_transformation``.

Regression coverage for transcription token-usage objects whose
``input_token_details`` is ``None``. OpenAI-compatible transcription servers
(for example llama.cpp with ``response_format=json``) return a ``tokens`` usage
block without that field, so the transformation must preserve the token totals
that drive spend accounting instead of dereferencing the missing detail object.
"""

from litellm.litellm_core_utils.llm_cost_calc.usage_object_transformation import (
    TranscriptionUsageObjectTransformation,
)
from litellm.types.utils import (
    TranscriptionUsageDurationObject,
    TranscriptionUsageInputTokenDetailsObject,
    TranscriptionUsageTokensObject,
)


def test_transform_transcription_usage_object_without_input_token_details():
    usage_object = TranscriptionUsageTokensObject(
        type="tokens",
        input_tokens=5,
        output_tokens=10,
        total_tokens=15,
    )

    result = TranscriptionUsageObjectTransformation.transform_transcription_usage_object(
        usage_object
    )

    assert result is not None
    assert result.prompt_tokens == 5
    assert result.completion_tokens == 10
    assert result.total_tokens == 15
    assert result.prompt_tokens_details is None


def test_transform_transcription_usage_object_with_input_token_details():
    usage_object = TranscriptionUsageTokensObject(
        type="tokens",
        input_tokens=5,
        output_tokens=10,
        total_tokens=15,
        input_token_details=TranscriptionUsageInputTokenDetailsObject(
            audio_tokens=2, text_tokens=3
        ),
    )

    result = TranscriptionUsageObjectTransformation.transform_transcription_usage_object(
        usage_object
    )

    assert result is not None
    assert result.total_tokens == 15
    assert result.prompt_tokens_details is not None
    assert result.prompt_tokens_details.text_tokens == 3
    assert result.prompt_tokens_details.audio_tokens == 2


def test_transform_transcription_usage_object_duration_returns_none():
    usage_object = TranscriptionUsageDurationObject(type="duration", seconds=1.5)

    result = TranscriptionUsageObjectTransformation.transform_transcription_usage_object(
        usage_object
    )

    assert result is None
