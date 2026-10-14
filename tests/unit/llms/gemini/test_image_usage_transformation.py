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
