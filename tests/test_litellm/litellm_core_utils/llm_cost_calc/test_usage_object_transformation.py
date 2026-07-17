"""
Tests for ``TranscriptionUsageObjectTransformation`` in
``litellm.litellm_core_utils.llm_cost_calc.usage_object_transformation``.
"""

from litellm.litellm_core_utils.llm_cost_calc.usage_object_transformation import (
    TranscriptionUsageObjectTransformation,
)
from litellm.types.utils import (
    TranscriptionUsageInputTokenDetailsObject,
    TranscriptionUsageTokensObject,
)


def test_transform_transcription_tokens_without_input_token_details():
    """Token usage without the per-modality input_token_details breakdown must still
    produce a Usage (token counts drive cost), not crash on None.

    Regression for https://github.com/BerriAI/litellm/issues/33764
    """
    usage_object = TranscriptionUsageTokensObject(
        type="tokens",
        input_tokens=14,
        output_tokens=45,
        total_tokens=59,
        input_token_details=None,
    )

    usage = TranscriptionUsageObjectTransformation.transform_transcription_usage_object(usage_object)

    assert usage is not None
    assert usage.prompt_tokens == 14
    assert usage.completion_tokens == 45
    assert usage.total_tokens == 59
    assert usage.prompt_tokens_details is None


def test_transform_transcription_tokens_with_input_token_details():
    """When the breakdown is present it is carried into prompt_tokens_details."""
    usage_object = TranscriptionUsageTokensObject(
        type="tokens",
        input_tokens=14,
        output_tokens=45,
        total_tokens=59,
        input_token_details=TranscriptionUsageInputTokenDetailsObject(audio_tokens=14, text_tokens=0),
    )

    usage = TranscriptionUsageObjectTransformation.transform_transcription_usage_object(usage_object)

    assert usage.prompt_tokens_details.audio_tokens == 14
    assert usage.prompt_tokens_details.text_tokens == 0
