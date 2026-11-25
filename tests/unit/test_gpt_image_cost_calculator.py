"""
Tests for OpenAI gpt-image-1 cost calculator

This tests the fix for GitHub issue #13847:
https://github.com/BerriAI/litellm/issues/13847

gpt-image-1 uses token-based pricing:
- Text Input: $5.00/1M tokens
- Image Input: $10.00/1M tokens
- Image Output: $40.00/1M tokens
"""

from collections.abc import Iterator
from datetime import datetime
from typing import Final

import pytest

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import convert_to_model_response_object
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


@pytest.mark.parametrize("provider", ["openai", "azure"])
@pytest.mark.parametrize("call_type", ["image_generation", "aimage_generation", "image_edit", "aimage_edit"])
@pytest.mark.parametrize("reasoning_tokens", [0, 60, 100])
def test_image_cost_adds_svg_authoring_without_double_billing_reasoning(
    provider: str, call_type: str, reasoning_tokens: int
) -> None:
    model: Final = f"{provider}/gpt-image-2.6-flare" if provider == "openai" else "azure/gpt-image-2.5-flare"
    body: Final = {
        "created": 1234567890,
        "data": [{"b64_json": "image"}],
        "usage": {
            "input_tokens": 20,
            "output_tokens": 40,
            "total_tokens": 60,
            "input_tokens_details": {"text_tokens": 10, "image_tokens": 10},
        },
        "svg_usage": {
            "model": "gpt-6.2-sol",
            "input_tokens": 30,
            "output_tokens": 100,
            "reasoning_tokens": reasoning_tokens,
            "total_tokens": 130,
        },
    }
    response: Final = convert_to_model_response_object(response_object=body, response_type="image_generation")
    image_only: Final = ImageResponse(**{key: value for key, value in body.items() if key != "svg_usage"})
    rendering_cost: Final = litellm.completion_cost(model=model, completion_response=image_only, call_type=call_type)
    authoring_cost: Final = sum(
        litellm.cost_per_token(
            model="gpt-6.2-sol", custom_llm_provider=provider, prompt_tokens=30, completion_tokens=100
        )
    )
    total: Final = litellm.completion_cost(model=model, completion_response=response, call_type=call_type)
    assert authoring_cost > 0
    assert total == pytest.approx(rendering_cost + authoring_cost)


@pytest.mark.parametrize("svg_model", ["gpt-6.2-sol", "unmapped-svg-authoring-model"])
@pytest.mark.parametrize("tokens", [0, 100])
def test_svg_empty_or_unmapped_usage_does_not_break_rendering_cost(svg_model: str, tokens: int) -> None:
    response: Final = ImageResponse(
        data=[ImageObject(b64_json="image")],
        usage=ImageUsage(
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            input_tokens_details=ImageUsageInputTokensDetails(text_tokens=10, image_tokens=0),
        ),
        svg_usage={
            "model": svg_model,
            "input_tokens": tokens,
            "output_tokens": tokens,
            "reasoning_tokens": tokens,
            "total_tokens": 2 * tokens,
        },
    )
    image_only: Final = response.model_copy(update={"svg_usage": None})
    rendering_cost: Final = litellm.completion_cost(
        model="gpt-image-2.6-flare", completion_response=image_only, call_type="image_edit"
    )
    expected_authoring_cost: Final = (
        sum(litellm.cost_per_token(model=svg_model, prompt_tokens=tokens, completion_tokens=tokens))
        if svg_model == "gpt-6.2-sol"
        else 0.0
    )
    assert litellm.completion_cost(
        model="gpt-image-2.6-flare", completion_response=response, call_type="image_edit"
    ) == pytest.approx(rendering_cost + expected_authoring_cost)


@pytest.mark.asyncio
@pytest.mark.parametrize("call_type", ["aimage_generation", "aimage_edit"])
async def test_svg_authoring_cost_reaches_logging_cost_calculator(call_type: str) -> None:
    response: Final = ImageResponse(
        data=[ImageObject(b64_json="image")],
        usage=ImageUsage(
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            input_tokens_details=ImageUsageInputTokensDetails(text_tokens=10, image_tokens=0),
        ),
        svg_usage={
            "model": "gpt-6.2-sol",
            "input_tokens": 30,
            "output_tokens": 100,
            "reasoning_tokens": 60,
            "total_tokens": 130,
        },
    )
    image_only: Final = response.model_copy(update={"svg_usage": None})
    expected: Final = litellm.completion_cost(
        model="gpt-image-2.6-flare", completion_response=image_only, call_type=call_type
    ) + sum(litellm.cost_per_token(model="gpt-6.2-sol", prompt_tokens=30, completion_tokens=100))
    logging_obj: Final = Logging(
        model="gpt-image-2.6-flare",
        messages=[],
        stream=False,
        call_type=call_type,
        start_time=datetime(2026, 1, 1),
        litellm_call_id="svg-authoring",
        function_id="svg-authoring",
    )
    logging_obj.update_environment_variables(
        model="gpt-image-2.6-flare", user="", optional_params={}, litellm_params={"custom_llm_provider": "openai"}
    )
    assert logging_obj.response_cost_calculator(result=response) == pytest.approx(expected)
    assert await logging_obj._response_cost_calculator_async(result=response) == pytest.approx(expected)


SERVING_IMAGE_DEPLOYMENT: Final = "image-deployment"


def _svg_logging_obj(
    call_type: str,
    metadata: dict[str, object],
    metadata_key: str = "metadata",
    served_by: str = SERVING_IMAGE_DEPLOYMENT,
    copied_from: str | None = None,
) -> Logging:
    logging_obj: Final = Logging(
        model="gpt-image-2.6-flare",
        messages=[],
        stream=False,
        call_type=call_type,
        start_time=datetime(2026, 1, 1),
        litellm_call_id="svg-author",
        function_id="svg-author",
    )
    logging_obj.update_environment_variables(
        model="gpt-image-2.6-flare",
        user="",
        optional_params={},
        litellm_params={
            "custom_llm_provider": "openai",
            metadata_key: {**metadata, "model_info": {"id": served_by, "original_model_id": copied_from}},
        },
    )
    return logging_obj


@pytest.fixture
def svg_author_deployment() -> Iterator[str]:
    from litellm import Router

    deployment_id: Final = "svg-author-deployment"
    Router(
        model_list=[
            {
                "model_name": "my-nano",
                "litellm_params": {"model": "openai/gpt-5-nano", "api_key": "sk-test"},
                "model_info": {
                    "id": deployment_id,
                    "input_cost_per_token": SVG_AUTHOR_INPUT_PRICE,
                    "output_cost_per_token": SVG_AUTHOR_OUTPUT_PRICE,
                },
            }
        ]
    )
    yield deployment_id
    litellm.model_cost.pop(deployment_id, None)


def _svg_response(svg_model: str) -> ImageResponse:
    return ImageResponse(
        data=[ImageObject(b64_json="image")],
        usage=ImageUsage(
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            input_tokens_details=ImageUsageInputTokensDetails(text_tokens=10, image_tokens=0),
        ),
        svg_usage={"model": svg_model, "input_tokens": 30, "output_tokens": 100, "total_tokens": 130},
    )


def _rendering_cost(response: ImageResponse, call_type: str) -> float:
    return litellm.completion_cost(
        model="gpt-image-2.6-flare",
        completion_response=response.model_copy(update={"svg_usage": None}),
        call_type=call_type,
    )


def _list_authoring_cost(svg_model: str) -> float:
    return sum(litellm.cost_per_token(model=svg_model, prompt_tokens=30, completion_tokens=100))


SVG_AUTHOR_INPUT_PRICE: Final = 0.00000004
SVG_AUTHOR_OUTPUT_PRICE: Final = 0.00000032


@pytest.mark.parametrize("metadata_key", ["metadata", "litellm_metadata"])
@pytest.mark.parametrize("call_type", ["aimage_generation", "aimage_edit"])
@pytest.mark.parametrize("reported_model", ["gpt-5-nano", "gpt-5-nano-2025-08-07"])
def test_svg_author_tokens_use_the_author_deployments_own_prices(
    call_type: str, reported_model: str, metadata_key: str, svg_author_deployment: str
) -> None:
    from litellm.constants import SVG_AUTHOR_DEPLOYMENT_METADATA_KEY

    response: Final = _svg_response(reported_model)
    deployment_authoring: Final = 30 * SVG_AUTHOR_INPUT_PRICE + 100 * SVG_AUTHOR_OUTPUT_PRICE
    assert deployment_authoring != pytest.approx(_list_authoring_cost(reported_model))

    stamp: Final = {
        "deployment_id": svg_author_deployment,
        "provider_model": "gpt-5-nano",
        "image_deployment_ids": [SERVING_IMAGE_DEPLOYMENT],
    }
    stamped: Final = _svg_logging_obj(call_type, {SVG_AUTHOR_DEPLOYMENT_METADATA_KEY: stamp}, metadata_key)
    unstamped: Final = _svg_logging_obj(call_type, {})
    rendering: Final = _rendering_cost(response, call_type)
    assert stamped.response_cost_calculator(result=response) == pytest.approx(rendering + deployment_authoring)
    assert unstamped.response_cost_calculator(result=response) == pytest.approx(
        rendering + _list_authoring_cost(reported_model)
    )


def test_svg_author_deployment_price_is_not_applied_to_a_different_reported_model(svg_author_deployment: str) -> None:
    from litellm.constants import SVG_AUTHOR_DEPLOYMENT_METADATA_KEY

    response: Final = _svg_response("gpt-5.5")
    stamp: Final = {
        "deployment_id": svg_author_deployment,
        "provider_model": "gpt-5-nano",
        "image_deployment_ids": [SERVING_IMAGE_DEPLOYMENT],
    }
    logging_obj: Final = _svg_logging_obj("aimage_edit", {SVG_AUTHOR_DEPLOYMENT_METADATA_KEY: stamp})
    assert logging_obj.response_cost_calculator(result=response) == pytest.approx(
        _rendering_cost(response, "aimage_edit") + _list_authoring_cost("gpt-5.5")
    )


def test_svg_author_deployment_price_is_not_applied_when_another_image_deployment_served(
    svg_author_deployment: str,
) -> None:
    from litellm.constants import SVG_AUTHOR_DEPLOYMENT_METADATA_KEY

    response: Final = _svg_response("gpt-5-nano")
    stamp: Final = {
        "deployment_id": svg_author_deployment,
        "provider_model": "gpt-5-nano",
        "image_deployment_ids": [SERVING_IMAGE_DEPLOYMENT],
    }
    logging_obj: Final = _svg_logging_obj(
        "aimage_generation", {SVG_AUTHOR_DEPLOYMENT_METADATA_KEY: stamp}, served_by="image-on-another-account"
    )
    assert logging_obj.response_cost_calculator(result=response) == pytest.approx(
        _rendering_cost(response, "aimage_generation") + _list_authoring_cost("gpt-5-nano")
    )


def test_svg_author_price_applies_to_a_request_scoped_copy_of_the_image_deployment(svg_author_deployment: str) -> None:
    from litellm.constants import SVG_AUTHOR_DEPLOYMENT_METADATA_KEY

    response: Final = _svg_response("gpt-5-nano")
    stamp: Final = {
        "deployment_id": svg_author_deployment,
        "provider_model": "gpt-5-nano",
        "image_deployment_ids": [SERVING_IMAGE_DEPLOYMENT],
    }
    logging_obj: Final = _svg_logging_obj(
        "aimage_generation",
        {SVG_AUTHOR_DEPLOYMENT_METADATA_KEY: stamp},
        served_by="request-scoped-copy",
        copied_from=SERVING_IMAGE_DEPLOYMENT,
    )
    assert logging_obj.response_cost_calculator(result=response) == pytest.approx(
        _rendering_cost(response, "aimage_generation") + 30 * SVG_AUTHOR_INPUT_PRICE + 100 * SVG_AUTHOR_OUTPUT_PRICE
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
