"""
Tests for OpenAI gpt-image-1 cost calculator

This tests the fix for GitHub issue #13847:
https://github.com/BerriAI/litellm/issues/13847

gpt-image-1 uses token-based pricing:
- Text Input: $5.00/1M tokens
- Image Input: $10.00/1M tokens
- Image Output: $40.00/1M tokens
"""

import pytest

import litellm
from litellm.types.utils import (
    ImageObject,
    ImageResponse,
    ImageUsage,
    ImageUsageInputTokensDetails,
)


@pytest.fixture(autouse=True)
def _use_local_model_cost_map(monkeypatch):
    original_model_cost = litellm.model_cost
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    litellm.get_model_info.cache_clear()
    try:
        yield
    finally:
        litellm.model_cost = original_model_cost
        litellm.get_model_info.cache_clear()


class TestGPTImageCostCalculator:
    """Test the OpenAI gpt-image cost calculator"""

    def test_gpt_image_1_cost_no_usage(self):
        """Test that cost returns 0 when no usage data is available"""
        from litellm.llms.openai.image_generation.cost_calculator import cost_calculator

        image_response = ImageResponse(
            created=1234567890,
            data=[ImageObject(url="http://example.com/image.jpg")],
        )

        cost = cost_calculator(
            model="gpt-image-1",
            image_response=image_response,
            custom_llm_provider="openai",
        )

        assert cost == 0.0

    @pytest.mark.parametrize(
        "usage",
        [
            None,
            ImageUsage(
                input_tokens=0,
                input_tokens_details=ImageUsageInputTokensDetails(image_tokens=0, text_tokens=0),
                output_tokens=0,
                total_tokens=0,
            ),
        ],
    )
    def test_gpt_image_1_bills_deployment_output_cost_per_image_without_usage_tokens(
        self, usage: ImageUsage | None
    ) -> None:
        from litellm.llms.openai.image_generation.cost_calculator import cost_calculator

        image_response = ImageResponse(
            created=1234567890,
            data=[ImageObject(url="http://example.com/one.jpg"), ImageObject(url="http://example.com/two.jpg")],
            usage=usage,
        )

        cost = cost_calculator(
            model="gpt-image-1",
            image_response=image_response,
            custom_llm_provider="openai",
            model_info={"output_cost_per_image": 0.05},
        )

        assert cost == pytest.approx(0.10)


class TestGPTImageCostRouting:
    """Test that gpt-image models are properly routed to the token-based calculator"""



class TestGPTImage15OutputImageTokens:
    """
    Test for GitHub issue #19508:
    Image usage calculation does not include image tokens in gpt-image-1.5

    gpt-image-1.5 returns output_tokens_details with separate image_tokens and text_tokens,
    and these must be correctly included in cost calculation.
    """


class TestCompletionCostIntegration:
    """Test the full completion_cost integration for gpt-image-1"""


class TestGPTImage2OutputImageTokensNoBreakdown:
    """
    Regression test: the OpenAI Images endpoints (/v1/images/generations and
    /v1/images/edits) return usage with NO output token breakdown — litellm's
    ImageUsage has no ``output_tokens_details`` field. Before the fix, the
    generated-image OUTPUT tokens were priced at the text rate
    (``output_cost_per_token`` = $10/1M for gpt-image-2) instead of the image rate
    (``output_cost_per_image_token`` = $30/1M), a ~3x undercount on the dominant
    cost component.
    """


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
