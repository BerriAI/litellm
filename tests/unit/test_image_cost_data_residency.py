"""
Tests for the OpenAI regional-processing uplift on gpt-image spend

This tests the fix for GitHub issue #46407:
https://github.com/BerriAI/litellm/issues/46407

`data_residency: eu` (routed through eu.api.openai.com) must apply the
per-model regional uplift multiplier to image-generation spend exactly as it
already does for chat/responses spend.
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


def _usage() -> ImageUsage:
    return ImageUsage(
        input_tokens=100,
        input_tokens_details=ImageUsageInputTokensDetails(image_tokens=50, text_tokens=50),
        output_tokens=200,
        total_tokens=300,
    )


def _image_response() -> ImageResponse:
    return ImageResponse(
        created=1234567890,
        data=[ImageObject(url="http://example.com/image.jpg")],
        usage=_usage(),
    )


class TestImageDataResidencyUplift:
    def test_image_usage_path_applies_eu_uplift(self) -> None:
        """The ImageUsage path (calculate_image_response_cost_from_usage) must apply the EU multiplier."""
        from litellm.llms.openai.image_generation.cost_calculator import cost_calculator

        image_response = _image_response()
        base_cost = cost_calculator(
            model="gpt-image-2.5-flare",
            image_response=image_response,
            custom_llm_provider="openai",
        )
        eu_cost = cost_calculator(
            model="gpt-image-2.5-flare",
            image_response=image_response,
            custom_llm_provider="openai",
            data_residency="eu",
        )

        assert base_cost > 0
        assert eu_cost == pytest.approx(base_cost * 1.1)

    def test_route_from_completion_cost_applies_eu_uplift(self) -> None:
        """completion_cost forwards data_residency into the image branch (issue #46407 part 1)."""
        base_cost = litellm.completion_cost(
            model="gpt-image-2.5-flare",
            custom_llm_provider="openai",
            completion_response=_image_response(),
            call_type="aimage_generation",
        )
        eu_cost = litellm.completion_cost(
            model="gpt-image-2.5-flare",
            custom_llm_provider="openai",
            completion_response=_image_response(),
            call_type="aimage_generation",
            data_residency="eu",
        )

        assert base_cost > 0
        assert eu_cost == pytest.approx(base_cost * 1.1)

    def test_us_uplift_applies_and_unknown_residency_does_not(self) -> None:
        from litellm.llms.openai.image_generation.cost_calculator import cost_calculator

        image_response = _image_response()
        base_cost = cost_calculator(
            model="gpt-image-2.5-flare",
            image_response=image_response,
            custom_llm_provider="openai",
        )

        us_cost = cost_calculator(
            model="gpt-image-2.5-flare",
            image_response=image_response,
            custom_llm_provider="openai",
            data_residency="us",
        )
        unknown_cost = cost_calculator(
            model="gpt-image-2.5-flare",
            image_response=image_response,
            custom_llm_provider="openai",
            data_residency="ap-southeast-1",
        )

        assert us_cost == pytest.approx(base_cost * 1.1)
        assert unknown_cost == pytest.approx(base_cost)

    def test_pre_cutoff_gpt_image_model_has_no_uplift(self) -> None:
        """gpt-image-1.5 predates the 2026-03-05 cutoff and carries no multiplier: residency must not change spend."""
        from litellm.llms.openai.image_generation.cost_calculator import cost_calculator

        image_response = _image_response()
        base_cost = cost_calculator(
            model="gpt-image-1.5",
            image_response=image_response,
            custom_llm_provider="openai",
        )
        eu_cost = cost_calculator(
            model="gpt-image-1.5",
            image_response=image_response,
            custom_llm_provider="openai",
            data_residency="eu",
        )

        assert base_cost > 0
        assert eu_cost == pytest.approx(base_cost)
