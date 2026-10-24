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
    ImageObject,
    ImageResponse,
    ImageUsage,
    ImageUsageInputTokensDetails,
    ModelInfo,
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


class TestGPTImageDataResidencyUplift:
    """
    Test for GitHub issue #46407:
    gpt-image cost ignored the OpenAI regional-processing (data residency) uplift.

    The chat path applies ``regional_processing_uplift_multiplier_<region>`` from
    the resolved price table inside ``generic_cost_per_token``; the image route
    never received ``data_residency``, so the uplift could never reach gpt-image
    calls. These tests pin the uplift on the image path using a price table that
    carries the multiplier key (same mechanism the chat path uses with cost-map
    entries).
    """

    _MODEL_INFO: Final[ModelInfo] = {
        "input_cost_per_token": 5e-06,
        "output_cost_per_token": 1e-05,
        "input_cost_per_image_token": 1e-05,
        "output_cost_per_image_token": 4e-05,
        "regional_processing_uplift_multiplier_eu": 1.1,
        "regional_processing_uplift_multiplier_us": 1.1,
    }

    @staticmethod
    def _image_response() -> ImageResponse:
        return ImageResponse(
            created=1234567890,
            data=[ImageObject(url="http://example.com/image.jpg")],
            usage=ImageUsage(
                input_tokens=1000,
                input_tokens_details=ImageUsageInputTokensDetails(image_tokens=600, text_tokens=400),
                output_tokens=500,
                total_tokens=1500,
            ),
        )

    def test_data_residency_uplift_applies_to_gpt_image_cost(self) -> None:
        from litellm.llms.openai.image_generation.cost_calculator import cost_calculator

        base: Final = cost_calculator(
            model="gpt-image-1",
            image_response=self._image_response(),
            custom_llm_provider="openai",
            model_info=self._MODEL_INFO,
        )
        eu: Final = cost_calculator(
            model="gpt-image-1",
            image_response=self._image_response(),
            custom_llm_provider="openai",
            model_info=self._MODEL_INFO,
            data_residency="eu",
        )
        assert base > 0
        assert abs(eu - base * 1.1) < 1e-12

    def test_data_residency_uplift_via_calculate_image_response_cost_from_usage(self) -> None:
        from litellm.litellm_core_utils.llm_cost_calc.utils import calculate_image_response_cost_from_usage

        base: Final = calculate_image_response_cost_from_usage(
            model="gpt-image-1",
            image_response=self._image_response(),
            custom_llm_provider="openai",
            model_info=self._MODEL_INFO,
        )
        eu: Final = calculate_image_response_cost_from_usage(
            model="gpt-image-1",
            image_response=self._image_response(),
            custom_llm_provider="openai",
            model_info=self._MODEL_INFO,
            data_residency="eu",
        )
        assert base is not None and eu is not None
        assert abs(eu - base * 1.1) < 1e-12

    def test_no_uplift_when_model_carries_no_multiplier(self) -> None:
        from litellm.llms.openai.image_generation.cost_calculator import cost_calculator

        model_info: Final[ModelInfo] = {
            "input_cost_per_token": 5e-06,
            "output_cost_per_token": 1e-05,
            "input_cost_per_image_token": 1e-05,
            "output_cost_per_image_token": 4e-05,
        }
        base: Final = cost_calculator(
            model="gpt-image-1",
            image_response=self._image_response(),
            custom_llm_provider="openai",
            model_info=model_info,
        )
        eu: Final = cost_calculator(
            model="gpt-image-1",
            image_response=self._image_response(),
            custom_llm_provider="openai",
            model_info=model_info,
            data_residency="eu",
        )
        assert eu == base

    def test_no_uplift_when_data_residency_absent(self) -> None:
        from litellm.llms.openai.image_generation.cost_calculator import cost_calculator

        none_region: Final = cost_calculator(
            model="gpt-image-1",
            image_response=self._image_response(),
            custom_llm_provider="openai",
            model_info=self._MODEL_INFO,
            data_residency=None,
        )
        base: Final = cost_calculator(
            model="gpt-image-1",
            image_response=self._image_response(),
            custom_llm_provider="openai",
            model_info=self._MODEL_INFO,
        )
        assert none_region == base

    def test_completion_cost_threads_data_residency_to_image_route(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """End-to-end: completion_cost's image branch must hand data_residency to the route."""
        from litellm.cost_calculator import completion_cost

        monkeypatch.setitem(litellm.model_cost["gpt-image-1"], "regional_processing_uplift_multiplier_eu", 1.1)

        base: Final = completion_cost(
            completion_response=self._image_response(),
            model="gpt-image-1",
            custom_llm_provider="openai",
            call_type="image_generation",
        )
        eu: Final = completion_cost(
            completion_response=self._image_response(),
            model="gpt-image-1",
            custom_llm_provider="openai",
            call_type="image_generation",
            data_residency="eu",
        )
        assert base > 0
        assert abs(eu - base * 1.1) < 1e-12
