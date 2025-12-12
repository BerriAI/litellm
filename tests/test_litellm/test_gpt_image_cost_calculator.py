"""
Tests for OpenAI gpt-image-1 cost calculator

This tests the fix for GitHub issue #13847:
https://github.com/BerriAI/litellm/issues/13847

gpt-image-1 cost calculation was incomplete - it only calculated output image
cost using pixel-based pricing and completely ignored input token costs.

The correct pricing structure for gpt-image-1 is token-based:
- Text Input: $5.00/1M tokens
- Image Input: $10.00/1M tokens
- Image Output: $40.00/1M tokens
"""

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath("../.."))

import pytest

import litellm
from litellm.types.utils import ImageResponse, ImageObject


class TestGPTImageCostCalculator:
    """Test the OpenAI gpt-image-1 cost calculator"""

    def test_gpt_image_1_cost_with_text_only(self):
        """Test cost calculation with only text input tokens"""
        from litellm.llms.openai.image_generation.cost_calculator import cost_calculator

        # Create mock ImageResponse with usage data
        usage = MagicMock()
        usage.input_tokens = 100
        usage.output_tokens = 5000
        usage.input_tokens_details = MagicMock()
        usage.input_tokens_details.text_tokens = 100
        usage.input_tokens_details.image_tokens = 0

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

        # Expected cost calculation:
        # Text input: 100 tokens * $5.00/1M = 100 * 5e-06 = 0.0005
        # Image input: 0 tokens * $10.00/1M = 0
        # Image output: 5000 tokens * $40.00/1M = 5000 * 4e-05 = 0.2
        # Total: 0.0005 + 0 + 0.2 = 0.2005
        expected_cost = 0.0005 + 0.2
        assert abs(cost - expected_cost) < 1e-6, f"Expected {expected_cost}, got {cost}"

    def test_gpt_image_1_cost_with_image_input(self):
        """Test cost calculation with both text and image input tokens"""
        from litellm.llms.openai.image_generation.cost_calculator import cost_calculator

        # Create mock ImageResponse with usage data including image input
        usage = MagicMock()
        usage.input_tokens = 600  # 100 text + 500 image
        usage.output_tokens = 5000
        usage.input_tokens_details = MagicMock()
        usage.input_tokens_details.text_tokens = 100
        usage.input_tokens_details.image_tokens = 500

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

        # Expected cost calculation:
        # Text input: 100 tokens * $5.00/1M = 100 * 5e-06 = 0.0005
        # Image input: 500 tokens * $10.00/1M = 500 * 1e-05 = 0.005
        # Image output: 5000 tokens * $40.00/1M = 5000 * 4e-05 = 0.2
        # Total: 0.0005 + 0.005 + 0.2 = 0.2055
        expected_cost = 0.0005 + 0.005 + 0.2
        assert abs(cost - expected_cost) < 1e-6, f"Expected {expected_cost}, got {cost}"

    def test_gpt_image_1_mini_cost(self):
        """Test cost calculation for gpt-image-1-mini model"""
        from litellm.llms.openai.image_generation.cost_calculator import cost_calculator

        # Create mock ImageResponse with usage data
        usage = MagicMock()
        usage.input_tokens = 100
        usage.output_tokens = 5000
        usage.input_tokens_details = MagicMock()
        usage.input_tokens_details.text_tokens = 100
        usage.input_tokens_details.image_tokens = 0

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

        # Expected cost calculation for gpt-image-1-mini:
        # Text input: 100 tokens * $2.00/1M = 100 * 2e-06 = 0.0002
        # Image input: 0 tokens * $2.50/1M = 0
        # Image output: 5000 tokens * $8.00/1M = 5000 * 8e-06 = 0.04
        # Total: 0.0002 + 0 + 0.04 = 0.0402
        expected_cost = 0.0002 + 0.04
        assert abs(cost - expected_cost) < 1e-6, f"Expected {expected_cost}, got {cost}"

    def test_gpt_image_1_cost_no_usage(self):
        """Test that cost returns 0 when no usage data is available"""
        from litellm.llms.openai.image_generation.cost_calculator import cost_calculator

        image_response = ImageResponse(
            created=1234567890,
            data=[ImageObject(url="http://example.com/image.jpg")],
        )
        # No usage attribute set

        cost = cost_calculator(
            model="gpt-image-1",
            image_response=image_response,
            custom_llm_provider="openai",
        )

        assert cost == 0.0

    def test_gpt_image_1_cost_no_input_details(self):
        """Test cost calculation when input_tokens_details is not available"""
        from litellm.llms.openai.image_generation.cost_calculator import cost_calculator

        # Create mock ImageResponse with usage data but no details
        usage = MagicMock()
        usage.input_tokens = 100
        usage.output_tokens = 5000
        usage.input_tokens_details = None  # No details

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

        # When no details, all input tokens are assumed to be text tokens
        # Text input: 100 tokens * $5.00/1M = 100 * 5e-06 = 0.0005
        # Image output: 5000 tokens * $40.00/1M = 5000 * 4e-05 = 0.2
        # Total: 0.0005 + 0.2 = 0.2005
        expected_cost = 0.0005 + 0.2
        assert abs(cost - expected_cost) < 1e-6, f"Expected {expected_cost}, got {cost}"


class TestGPTImageCostRouting:
    """Test that gpt-image models are properly routed to the token-based calculator"""

    def test_openai_gpt_image_routes_to_token_calculator(self):
        """Test that OpenAI gpt-image-1 routes to token-based calculator"""
        from litellm.litellm_core_utils.llm_cost_calc.utils import CostCalculatorUtils

        # Create mock ImageResponse with usage data
        usage = MagicMock()
        usage.input_tokens = 100
        usage.output_tokens = 5000
        usage.input_tokens_details = MagicMock()
        usage.input_tokens_details.text_tokens = 100
        usage.input_tokens_details.image_tokens = 0

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

        # Should be > 0 (token-based calculation)
        assert cost > 0, f"Expected positive cost, got {cost}"

        # Verify it's using token-based pricing, not pixel-based
        # Pixel-based would give a very different result
        expected_cost = 0.0005 + 0.2  # text + output
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

        # DALL-E should use pixel-based calculation
        cost = CostCalculatorUtils.route_image_generation_cost_calculator(
            model="dall-e-3",
            completion_response=image_response,
            custom_llm_provider="openai",
            size="1024x1024",
            quality="standard",
            n=1,
        )

        # DALL-E cost should be calculated (pixel-based)
        assert cost >= 0


class TestCompletionCostIntegration:
    """Test the full completion_cost integration for gpt-image-1"""

    def test_completion_cost_gpt_image_1(self):
        """Test completion_cost correctly calculates gpt-image-1 costs"""
        # Create mock ImageResponse with usage data
        usage = MagicMock()
        usage.input_tokens = 100
        usage.output_tokens = 5000
        usage.total_tokens = 5100
        usage.input_tokens_details = MagicMock()
        usage.input_tokens_details.text_tokens = 100
        usage.input_tokens_details.image_tokens = 0

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

        # Expected cost: text input + output image tokens
        expected_cost = 0.0005 + 0.2
        assert abs(cost - expected_cost) < 1e-6, f"Expected {expected_cost}, got {cost}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
