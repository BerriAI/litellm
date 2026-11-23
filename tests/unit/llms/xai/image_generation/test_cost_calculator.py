from datetime import datetime
from typing import Final

import httpx
import pytest
from openai import AsyncOpenAI, OpenAI

import litellm
from litellm.litellm_core_utils.llm_cost_calc.utils import CostCalculatorUtils
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import convert_to_model_response_object
from litellm.types.utils import ImageResponse


def _response(ticks: object = None) -> ImageResponse:
    return convert_to_model_response_object(
        response_object={
            "created": 123,
            "data": [{"url": "https://image.test/image.png"}],
            "usage": {"cost_in_usd_ticks": ticks},
        },
        response_type="image_generation",
    )


@pytest.mark.parametrize("ticks", (0, 1234567890))
@pytest.mark.parametrize("call_type", ("image_generation", "aimage_generation", "image_edit", "aimage_edit"))
def test_billed_cost_is_request_total_even_for_multiple_images(ticks: int, call_type: str) -> None:
    assert (
        CostCalculatorUtils.route_image_generation_cost_calculator(
            model="grok-imagine-image-2.0",
            custom_llm_provider="xai",
            completion_response=_response(ticks),
            n=4,
            call_type=call_type,
            optional_params={"resolution": "2k", "quality": "medium"},
        )
        == ticks / 10_000_000_000
    )


@pytest.mark.parametrize("ticks", (None, -1, True, "400000000", 1.5))
@pytest.mark.parametrize("resolution,unit_price", ((None, 0.17), ("1k", 0.17), ("2k", 0.29)))
def test_missing_or_invalid_ticks_use_requested_resolution(
    ticks: object, resolution: str | None, unit_price: float
) -> None:
    params: Final = {"resolution": resolution} if resolution else {}
    assert CostCalculatorUtils.route_image_generation_cost_calculator(
        model="unmapped-image-fixture",
        custom_llm_provider="xai",
        completion_response=_response(ticks),
        n=3,
        optional_params={"extra_body": params},
        model_info={"output_cost_per_image_1K": 0.17, "output_cost_per_image_2K": 0.29},
    ) == pytest.approx(unit_price * 3)


@pytest.mark.parametrize(
    "quality,call_type,resolution",
    (
        ("medium", "image_generation", "1k"),
        ("high", "image_generation", "2k"),
        (None, "image_edit", "1k"),
        ("auto", "aimage_edit", "2k"),
        ("low", "image_generation", "1.5k"),
    ),
)
def test_unpublished_rates_keep_legacy_fallback(quality: str | None, call_type: str, resolution: str) -> None:
    assert CostCalculatorUtils.route_image_generation_cost_calculator(
        model="unmapped-image-fixture",
        custom_llm_provider="xai",
        completion_response=_response(),
        n=2,
        quality=quality,
        call_type=call_type,
        optional_params={"resolution": resolution},
        model_info={"input_cost_per_image": 0.71, "output_cost_per_image_1K": 0.17, "output_cost_per_image_2K": 0.29},
    ) == pytest.approx(0.71 * 2)


@pytest.mark.parametrize("provider", ("xai", "openai"))
def test_flat_price_and_empty_response_remain_supported(provider: str) -> None:
    response: Final = ImageResponse(data=[])
    assert (
        CostCalculatorUtils.route_image_generation_cost_calculator(
            model="unmapped-image-fixture",
            custom_llm_provider=provider,
            completion_response=response,
            model_info={"output_cost_per_image": 0.37},
        )
        == 0.0
    )
    assert CostCalculatorUtils.route_image_generation_cost_calculator(
        model="unmapped-image-fixture",
        custom_llm_provider=provider,
        completion_response=response,
        n=2,
        model_info={"output_cost_per_image": 0.37},
    ) == pytest.approx(0.37 * 2)


@pytest.mark.parametrize("quality,call_type", (("low", "image_edit"), ("auto", "image_generation"), (None, None)))
def test_low_resolution_price_applies_without_flat_override(quality: str | None, call_type: str | None) -> None:
    assert CostCalculatorUtils.route_image_generation_cost_calculator(
        model="unmapped-image-fixture",
        custom_llm_provider="xai",
        completion_response=_response(),
        n=2,
        quality=quality,
        call_type=call_type,
        optional_params={"resolution": "2k"},
        model_info={"output_cost_per_image_2K": 0.29},
    ) == pytest.approx(0.29 * 2)


def test_deployment_flat_price_overrides_shared_resolution_price() -> None:
    assert CostCalculatorUtils.route_image_generation_cost_calculator(
        model="grok-imagine-image-2.0",
        custom_llm_provider="xai",
        completion_response=_response(),
        n=2,
        model_info={"output_cost_per_image": 0.83},
    ) == pytest.approx(0.83 * 2)


@pytest.mark.parametrize(
    "price_key,resolution",
    (
        ("input_cost_per_image", "1k"),
        ("output_cost_per_image", "1k"),
        ("output_cost_per_image_1K", "1k"),
        ("output_cost_per_image_2K", "2k"),
    ),
)
def test_deployment_image_price_overrides_billed_ticks(price_key: str, resolution: str) -> None:
    assert CostCalculatorUtils.route_image_generation_cost_calculator(
        model="grok-imagine-image-2.0",
        custom_llm_provider="xai",
        completion_response=_response(1234567890),
        n=3,
        optional_params={"resolution": resolution},
        model_info={price_key: 0.83},
    ) == pytest.approx(0.83 * 3)


@pytest.mark.parametrize("ticks", (None, -1, True, "600000000", 0, 600000000))
@pytest.mark.parametrize("pixel_rate", (0.0, 0.000001))
@pytest.mark.parametrize(
    "size,pixels,n", ((None, 1024 * 1024, 3), ("1024x1024", 1024 * 1024, 1), ("512x256", 512 * 256, 2))
)
def test_deployment_pixel_price_overrides_ticks_and_resolution_prices(
    ticks: object, pixel_rate: float, size: str | None, pixels: int, n: int
) -> None:
    assert CostCalculatorUtils.route_image_generation_cost_calculator(
        model="grok-imagine-image-2.0",
        custom_llm_provider="xai",
        completion_response=_response(ticks),
        size=size,
        quality="low",
        n=n,
        model_info={"input_cost_per_pixel": pixel_rate, "output_cost_per_image_2K": 0.29},
        optional_params={"resolution": "2k"},
        call_type="image_generation",
    ) == pytest.approx(pixel_rate * pixels * n)


@pytest.mark.parametrize("ticks", (None, 600000000))
@pytest.mark.parametrize("price_key", ("input_cost_per_image", "output_cost_per_image"))
def test_deployment_flat_price_still_precedes_pixel_price(ticks: object, price_key: str) -> None:
    assert CostCalculatorUtils.route_image_generation_cost_calculator(
        model="unmapped-image-fixture",
        custom_llm_provider="xai",
        completion_response=_response(ticks),
        n=2,
        quality="low",
        model_info={price_key: 0.83, "input_cost_per_pixel": 0.000001, "output_cost_per_image_2K": 0.29},
        optional_params={"resolution": "2k"},
    ) == pytest.approx(0.83 * 2)


@pytest.mark.asyncio
async def test_router_preserves_deployment_pixel_price_through_image_normalization() -> None:
    ticks: Final = 600000000
    transport: Final = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={
                "created": 123,
                "data": [{"url": "https://image.test/output.png"}],
                "usage": {"cost_in_usd_ticks": ticks},
            },
        )
    )
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "pixel-fixture",
                "litellm_params": {
                    "model": "xai/grok-imagine-image-2.0",
                    "api_key": "test",
                    "api_base": "https://image.test/v1",
                },
                "model_info": {"input_cost_per_pixel": 0.000001},
            }
        ]
    )
    logging_obj: Final = Logging(
        model="grok-imagine-image-2.0",
        messages=[],
        stream=False,
        call_type="aimage_generation",
        start_time=datetime(2026, 1, 1),
        litellm_call_id="pixel-override",
        function_id="pixel-override",
    )
    async with AsyncOpenAI(
        api_key="test", base_url="https://image.test/v1", http_client=httpx.AsyncClient(transport=transport)
    ) as client:
        response: Final = await router.aimage_generation(
            model="pixel-fixture",
            prompt="circle",
            quality="low",
            n=1,
            extra_body={"resolution": "2k"},
            client=client,
            litellm_logging_obj=logging_obj,
        )
        assert getattr(response.usage, "cost_in_usd_ticks") == ticks
        assert logging_obj.response_cost_calculator(result=response) == pytest.approx(1024 * 1024 * 0.000001)


def test_shared_resolution_prices_are_used_when_ticks_are_absent() -> None:
    pricing: Final = litellm.get_model_info("xai/grok-imagine-image-2.0")
    assert CostCalculatorUtils.route_image_generation_cost_calculator(
        model="grok-imagine-image-2.0",
        custom_llm_provider="xai",
        completion_response=_response(),
        n=2,
        optional_params={"resolution": "2k"},
    ) == pytest.approx(pricing["output_cost_per_image_2K"] * 2)


def test_missing_all_prices_keeps_missing_cost_error() -> None:
    with pytest.raises(Exception, match="Model not found in cost map"):
        CostCalculatorUtils.route_image_generation_cost_calculator(
            model="unmapped-image-fixture",
            custom_llm_provider="xai",
            completion_response=_response(),
        )


def test_non_xai_usage_ticks_do_not_override_flat_prices() -> None:
    assert CostCalculatorUtils.route_image_generation_cost_calculator(
        model="unmapped-image-fixture",
        custom_llm_provider="openai",
        completion_response=_response(1234567890),
        n=2,
        model_info={"output_cost_per_image": 0.37},
    ) == pytest.approx(0.37 * 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", (False, True))
async def test_generation_sdk_preserves_billed_ticks_through_normalization(is_async: bool) -> None:
    ticks: Final = 1234567890
    transport: Final = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={
                "created": 123,
                "data": [{"url": "https://image.test/output.png"}],
                "usage": {"cost_in_usd_ticks": ticks},
            },
        )
    )
    if is_async:
        async with AsyncOpenAI(
            api_key="test", base_url="https://image.test/v1", http_client=httpx.AsyncClient(transport=transport)
        ) as client:
            response: Final = await litellm.aimage_generation(
                model="xai/grok-imagine-image-2.0",
                prompt="circle",
                api_key="test",
                api_base="https://image.test/v1",
                client=client,
            )
            assert getattr(response.usage, "cost_in_usd_ticks") == ticks
            assert (
                litellm.completion_cost(completion_response=response, model="xai/grok-imagine-image-2.0", n=4)
                == ticks / 10_000_000_000
            )
    else:
        with OpenAI(
            api_key="test", base_url="https://image.test/v1", http_client=httpx.Client(transport=transport)
        ) as client:
            sync_response: Final = litellm.image_generation(
                model="xai/grok-imagine-image-2.0",
                prompt="circle",
                api_key="test",
                api_base="https://image.test/v1",
                client=client,
            )
            assert getattr(sync_response.usage, "cost_in_usd_ticks") == ticks
            assert (
                litellm.completion_cost(completion_response=sync_response, model="xai/grok-imagine-image-2.0", n=4)
                == ticks / 10_000_000_000
            )
