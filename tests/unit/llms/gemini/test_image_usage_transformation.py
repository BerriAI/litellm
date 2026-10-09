from typing import Final

import pytest

from litellm.llms.gemini.image_usage_transformation import transform_gemini_image_usage

PROMPT_DETAILS: Final = [{"modality": "TEXT", "tokenCount": 30}, {"modality": "IMAGE", "tokenCount": 5}]


@pytest.mark.parametrize(
    ("candidates_details", "expected_output_details"),
    [
        ({}, {"image_tokens": 1716, "text_tokens": 0}),
        (
            {"candidatesTokensDetails": [{"modality": "IMAGE", "tokenCount": 1120}]},
            {"image_tokens": 1120, "text_tokens": 596},
        ),
        (
            {"candidates_tokens_details": [{"modality": "TEXT", "token_count": 16}]},
            {"image_tokens": 0, "text_tokens": 1716},
        ),
    ],
)
def test_transform_gemini_image_usage_reports_image_and_chat_style_counts(
    candidates_details: dict[str, object], expected_output_details: dict[str, int]
):
    usage: Final = transform_gemini_image_usage(
        {
            "promptTokenCount": 35,
            "candidatesTokenCount": 1716,
            "totalTokenCount": 1751,
            "promptTokensDetails": PROMPT_DETAILS,
            **candidates_details,
        }
    )

    assert usage.model_dump() == {
        "input_tokens": 35,
        "input_tokens_details": {"image_tokens": 5, "text_tokens": 30},
        "output_tokens": 1716,
        "total_tokens": 1751,
        "prompt_tokens": 35,
        "prompt_tokens_details": {"image_tokens": 5, "text_tokens": 30},
        "completion_tokens": 1716,
        "completion_tokens_details": expected_output_details,
        "output_tokens_details": expected_output_details,
    }


def test_transform_gemini_image_usage_adds_exclusive_thoughts_to_output_tokens() -> None:
    usage: Final = transform_gemini_image_usage(
        {
            "promptTokenCount": 12,
            "candidatesTokenCount": 1300,
            "thoughtsTokenCount": 40,
            "totalTokenCount": 1352,
            "candidatesTokensDetails": [
                {"modality": "TEXT", "tokenCount": 10},
                {"modality": "IMAGE", "tokenCount": 1290},
            ],
        }
    )

    assert usage.output_tokens == 1340
    assert usage.completion_tokens == 1340
    assert usage.completion_tokens_details == {
        "text_tokens": 10,
        "image_tokens": 1290,
        "reasoning_tokens": 40,
    }
    assert usage.output_tokens_details == usage.completion_tokens_details


def test_transform_gemini_image_usage_does_not_double_count_inclusive_thoughts() -> None:
    usage: Final = transform_gemini_image_usage(
        {
            "promptTokenCount": 12,
            "candidatesTokenCount": 1300,
            "thoughtsTokenCount": 40,
            "totalTokenCount": 1312,
            "candidatesTokensDetails": [
                {"modality": "TEXT", "tokenCount": 10},
                {"modality": "IMAGE", "tokenCount": 1250},
            ],
        }
    )

    assert usage.output_tokens == 1300
    assert usage.completion_tokens == 1300
    assert usage.completion_tokens_details == {
        "text_tokens": 10,
        "image_tokens": 1250,
        "reasoning_tokens": 40,
    }
    assert usage.output_tokens_details == usage.completion_tokens_details


def test_transform_gemini_image_usage_fallback_excludes_thoughts_from_image_tokens() -> None:
    usage: Final = transform_gemini_image_usage(
        {
            "promptTokenCount": 12,
            "candidatesTokenCount": 1300,
            "thoughtsTokenCount": 40,
            "totalTokenCount": 1352,
        }
    )

    assert usage.output_tokens == 1340
    assert usage.completion_tokens_details == {
        "text_tokens": 0,
        "image_tokens": 1300,
        "reasoning_tokens": 40,
    }
    assert usage.output_tokens_details == usage.completion_tokens_details
