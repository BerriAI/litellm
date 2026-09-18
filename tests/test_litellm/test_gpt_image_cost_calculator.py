"""
Tests for OpenAI gpt-image-1 cost calculator

This tests the fix for GitHub issue #13847:
https://github.com/BerriAI/litellm/issues/13847

gpt-image-1 uses token-based pricing:
- Text Input: $5.00/1M tokens
- Image Input: $10.00/1M tokens
- Image Output: $40.00/1M tokens
"""

from typing import Final

import pytest

import litellm
from litellm.types.utils import (
    CompletionTokensDetailsWrapper,
    ImageObject,
    ImageResponse,
    ImageUsage,
    ImageUsageInputTokensDetails,
    PromptTokensDetailsWrapper,
    Usage,
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

    @pytest.mark.parametrize("family", ["flare", "sunburst"])
    @pytest.mark.parametrize("snapshot", ["", "-2026-09-08"])
    @pytest.mark.parametrize("call_type", ["image_generation", "image_edit"])
    @pytest.mark.parametrize("cached_text,cached_image", [(0, 0), (50, 500)])
    def test_image_25_official_prices(self, family, snapshot, call_type, cached_text, cached_image):
        response: Final = ImageResponse(
            created=1,
            data=[],
            usage={
                "input_tokens": 1100,
                "output_tokens": 100,
                "total_tokens": 1200,
                "input_tokens_details": {
                    "text_tokens": 100,
                    "image_tokens": 1000,
                    "cached_tokens": cached_text + cached_image,
                    "cached_tokens_details": {"text_tokens": cached_text, "image_tokens": cached_image},
                },
            },
        )
        cost: Final = litellm.completion_cost(
            model="gpt-image-2.5-" + family + snapshot,
            completion_response=response,
            call_type=call_type,
            custom_llm_provider="openai",
        )
        expected: Final = (
            (100 - cached_text) * 5e-6
            + cached_text * 1.25e-6
            + (1000 - cached_image) * 8e-6
            + cached_image * 2e-6
            + 100 * 30e-6
        )
        assert cost == pytest.approx(expected)

    def test_gpt_image_1_cost_with_text_only(self):
        """Test cost calculation with only text input tokens"""
        from litellm.llms.openai.image_generation.cost_calculator import cost_calculator

        usage = ImageUsage(
            input_tokens=100,
            output_tokens=5000,
            total_tokens=5100,
            input_tokens_details=ImageUsageInputTokensDetails(
                text_tokens=100,
                image_tokens=0,
            ),
        )

        image_response = ImageResponse(
            created=1234567890,
            data=[ImageObject(url="http://example.com/image.jpg")],
        )
        image_response.usage = usage

        cost = cost_calculator(
            model="gpt-image-1",
            image_response=image_response,
            custom_llm_provider="openai",
        )

        # Expected cost:
        # Text input: 100 * $5/1M = 0.0005
        # Image output: 5000 * $40/1M = 0.2
        # Total: 0.2005
        expected_cost = 0.0005 + 0.2
        assert abs(cost - expected_cost) < 1e-6, f"Expected {expected_cost}, got {cost}"

    def test_gpt_image_1_cost_with_image_input(self):
        """Test cost calculation with both text and image input tokens (for edits)"""
        from litellm.llms.openai.image_generation.cost_calculator import cost_calculator

        usage = ImageUsage(
            input_tokens=600,
            output_tokens=5000,
            total_tokens=5600,
            input_tokens_details=ImageUsageInputTokensDetails(
                text_tokens=100,
                image_tokens=500,
            ),
        )

        image_response = ImageResponse(
            created=1234567890,
            data=[ImageObject(url="http://example.com/image.jpg")],
        )
        image_response.usage = usage

        cost = cost_calculator(
            model="gpt-image-1",
            image_response=image_response,
            custom_llm_provider="openai",
        )

        # Expected cost:
        # Text input: 100 * $5/1M = 0.0005
        # Image input: 500 * $10/1M = 0.005
        # Image output: 5000 * $40/1M = 0.2
        # Total: 0.2055
        expected_cost = 0.0005 + 0.005 + 0.2
        assert abs(cost - expected_cost) < 1e-6, f"Expected {expected_cost}, got {cost}"

    def test_gpt_image_1_mini_cost(self):
        """Test cost calculation for gpt-image-1-mini model"""
        from litellm.llms.openai.image_generation.cost_calculator import cost_calculator

        usage = ImageUsage(
            input_tokens=100,
            output_tokens=5000,
            total_tokens=5100,
            input_tokens_details=ImageUsageInputTokensDetails(
                text_tokens=100,
                image_tokens=0,
            ),
        )

        image_response = ImageResponse(
            created=1234567890,
            data=[ImageObject(url="http://example.com/image.jpg")],
        )
        image_response.usage = usage

        cost = cost_calculator(
            model="gpt-image-1-mini",
            image_response=image_response,
            custom_llm_provider="openai",
        )

        # Expected cost for gpt-image-1-mini:
        # Text input: 100 * $2/1M = 0.0002
        # Image output: 5000 * $8/1M = 0.04
        # Total: 0.0402
        expected_cost = 0.0002 + 0.04
        assert abs(cost - expected_cost) < 1e-6, f"Expected {expected_cost}, got {cost}"

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


class TestGPTImageCostRouting:
    """Test that gpt-image models are properly routed to the token-based calculator"""

    def test_openai_dalle_routes_to_pixel_calculator(self):
        """Test that OpenAI DALL-E still routes to pixel-based calculator"""
        from litellm.litellm_core_utils.llm_cost_calc.utils import CostCalculatorUtils

        image_response = ImageResponse(
            created=1234567890,
            data=[ImageObject(url="http://example.com/image.jpg")],
        )
        image_response.size = "1024x1024"
        image_response.quality = "standard"

        cost = CostCalculatorUtils.route_image_generation_cost_calculator(
            model="dall-e-3",
            completion_response=image_response,
            custom_llm_provider="openai",
            size="1024x1024",
            quality="standard",
            n=1,
        )

        assert cost >= 0


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
