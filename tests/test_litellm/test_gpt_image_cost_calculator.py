"""
Tests for OpenAI gpt-image-1 cost calculator

This tests the fix for GitHub issue #13847:
https://github.com/BerriAI/litellm/issues/13847

gpt-image-1 uses token-based pricing:
- Text Input: $5.00/1M tokens
- Image Input: $10.00/1M tokens
- Image Output: $40.00/1M tokens
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import pytest

import litellm
from litellm.types.utils import (
    CompletionTokensDetailsWrapper,
    ImageResponse,
    ImageObject,
    ImageUsage,
    ImageUsageInputTokensDetails,
    PromptTokensDetailsWrapper,
    Usage,
)


class TestGPTImageCostCalculator:
    """Test the OpenAI gpt-image-1 cost calculator"""

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

    def test_openai_gpt_image_routes_to_token_calculator(self):
        """Test that OpenAI gpt-image-1 routes to token-based calculator"""
        from litellm.litellm_core_utils.llm_cost_calc.utils import CostCalculatorUtils

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

        cost = CostCalculatorUtils.route_image_generation_cost_calculator(
            model="gpt-image-1",
            completion_response=image_response,
            custom_llm_provider="openai",
        )

        expected_cost = 0.0005 + 0.2
        assert abs(cost - expected_cost) < 1e-6, f"Expected {expected_cost}, got {cost}"

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

    def test_gpt_image_15_output_image_tokens_cost(self):
        """
        Test that output image tokens are correctly included in cost calculation.

        This tests the fix for issue #19508 where output_tokens_details.image_tokens
        were not being included in the cost calculation, causing costs to be
        underreported (e.g., $0.046 instead of $0.14).
        """
        # Simulate gpt-image-1.5 response with output_tokens_details
        # This is what the API returns and what convert_to_image_response transforms
        usage = Usage(
            prompt_tokens=169,
            completion_tokens=4599,
            total_tokens=4768,
            prompt_tokens_details=PromptTokensDetailsWrapper(
                text_tokens=169,
                image_tokens=0,
            ),
            completion_tokens_details=CompletionTokensDetailsWrapper(
                text_tokens=439,
                image_tokens=4160,
            ),
        )

        image_response = ImageResponse(
            created=1234567890,
            data=[ImageObject(b64_json="test")],
        )
        image_response.usage = usage
        image_response._hidden_params = {"custom_llm_provider": "openai"}

        cost = litellm.completion_cost(
            completion_response=image_response,
            model="gpt-image-1.5",
            call_type="image_generation",
            custom_llm_provider="openai",
        )

        # gpt-image-1.5 pricing:
        # - input_cost_per_token: 5e-06 ($5/1M for text input)
        # - output_cost_per_token: 1e-05 ($10/1M for text output)
        # - output_cost_per_image_token: 3.2e-05 ($32/1M for image output)
        #
        # Expected cost:
        # Input text: 169 * $5/1M = $0.000845
        # Output text: 439 * $10/1M = $0.00439
        # Output image: 4160 * $32/1M = $0.13312
        # Total: $0.138355
        expected_cost = 169 * 5e-06 + 439 * 1e-05 + 4160 * 3.2e-05

        assert abs(cost - expected_cost) < 1e-6, (
            f"Expected {expected_cost}, got {cost}. "
            f"Image tokens may not be included in cost calculation."
        )


class TestCompletionCostIntegration:
    """Test the full completion_cost integration for gpt-image-1"""

    def test_completion_cost_gpt_image_1(self):
        """Test completion_cost correctly calculates gpt-image-1 costs"""
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
        image_response._hidden_params = {"custom_llm_provider": "openai"}

        cost = litellm.completion_cost(
            completion_response=image_response,
            model="gpt-image-1",
            call_type="image_generation",
            custom_llm_provider="openai",
        )

        expected_cost = 0.0005 + 0.2
        assert abs(cost - expected_cost) < 1e-6, f"Expected {expected_cost}, got {cost}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
