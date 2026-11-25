from typing import Final

import pytest
from pydantic import ValidationError

import litellm
from litellm.llms.gemini.image_usage_transformation import transform_gemini_image_usage
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig
from litellm.types.llms.vertex_ai import UsageMetadata
from litellm.types.utils import ImageResponse, ModelResponse

PROMPT_DETAILS: Final = [{"modality": "TEXT", "tokenCount": 30}, {"modality": "IMAGE", "tokenCount": 5}]


@pytest.mark.parametrize(
    "cached_fields",
    [{}, {"cachedContentTokenCount": None}, {"cachedContentTokenCount": 0}, {"cachedContentTokenCount": 6}],
)
def test_null_cached_content_count_matches_absent(cached_fields: dict[str, int | None]) -> None:
    usage: Final = transform_gemini_image_usage(
        {
            "promptTokenCount": 35,
            "candidatesTokenCount": 4,
            "totalTokenCount": 39,
            "promptTokensDetails": PROMPT_DETAILS,
            **cached_fields,
        }
    )
    assert usage.input_tokens_details.model_dump() == {
        "text_tokens": 30,
        "image_tokens": 5,
        "cached_tokens": cached_fields.get("cachedContentTokenCount") or 0,
        "cached_tokens_details": None,
    }
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (35, 4, 39)


@pytest.mark.parametrize("provider", ["gemini", "vertex_ai"])
@pytest.mark.parametrize("cached_tokens", [0, 600])
@pytest.mark.parametrize("cache_breakdown", [False, True])
def test_image_generation_cached_usage_cost_matches_chat(
    provider: str, cached_tokens: int, cache_breakdown: bool
) -> None:
    metadata: Final[UsageMetadata] = {
        "promptTokenCount": 1000,
        "candidatesTokenCount": 100,
        "totalTokenCount": 1100,
        "cachedContentTokenCount": cached_tokens,
        "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 800}, {"modality": "IMAGE", "tokenCount": 200}],
        "candidatesTokensDetails": [{"modality": "IMAGE", "tokenCount": 100}],
        **(
            {
                "cacheTokensDetails": [
                    {"modality": "TEXT", "tokenCount": cached_tokens * 5 // 6},
                    {"modality": "IMAGE", "tokenCount": cached_tokens // 6},
                ]
            }
            if cache_breakdown
            else {}
        ),
    }
    image_response: Final = ImageResponse(usage=transform_gemini_image_usage(metadata))
    chat_response: Final = ModelResponse(usage=VertexGeminiConfig.calculate_usage({"usageMetadata": metadata}))
    model: Final = f"{provider}/gemini-3.1-flash-lite-image"
    assert litellm.completion_cost(completion_response=image_response, model=model) == pytest.approx(
        litellm.completion_cost(completion_response=chat_response, model=model)
    )


@pytest.mark.parametrize(
    ("candidates_details", "expected_output_details"),
    [
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
        "input_tokens_details": {
            "image_tokens": 5,
            "text_tokens": 30,
            "cached_tokens": 0,
            "cached_tokens_details": None,
        },
        "output_tokens": 1716,
        "total_tokens": 1751,
        "prompt_tokens": 35,
        "prompt_tokens_details": {
            "image_tokens": 5,
            "text_tokens": 30,
            "cached_tokens": 0,
            "cached_tokens_details": None,
        },
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


@pytest.mark.parametrize("thoughts", [0, 40])
def test_transform_gemini_image_usage_preserves_unspecified_output_modality(thoughts: int) -> None:
    usage: Final = transform_gemini_image_usage(
        {
            "promptTokenCount": 12,
            "candidatesTokenCount": 1300,
            "thoughtsTokenCount": thoughts,
            "totalTokenCount": 1312 + thoughts,
        }
    )

    assert usage.output_tokens == 1300 + thoughts
    assert usage.completion_tokens_details == {
        "reasoning_tokens": thoughts,
    }
    assert usage.output_tokens_details == usage.completion_tokens_details


@pytest.mark.parametrize("count_field", ["promptTokenCount", "totalTokenCount", "cachedContentTokenCount"])
def test_transform_gemini_image_usage_rejects_a_count_that_is_not_a_number(count_field: str):
    with pytest.raises(ValidationError):
        transform_gemini_image_usage(
            {"promptTokenCount": 35, "candidatesTokenCount": 4, "totalTokenCount": 39, count_field: "many"}
        )
