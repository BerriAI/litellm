import pytest

from litellm.caching.caching import DualCache
from litellm.router_strategy.lowest_cost import LowestCostLoggingHandler


@pytest.mark.asyncio
async def test_string_costs_are_sorted_numerically():
    handler = LowestCostLoggingHandler(router_cache=DualCache())
    deployments = [
        {
            "model_name": "test-model-group",
            "litellm_params": {
                "model": "test/unknown-model",
                "input_cost_per_token": "0.000009",
                "output_cost_per_token": "0.000009",
            },
            "model_info": {"id": "cheap"},
        },
        {
            "model_name": "test-model-group",
            "litellm_params": {
                "model": "test/unknown-model",
                "input_cost_per_token": "0.000008",
                "output_cost_per_token": "0.000012",
            },
            "model_info": {"id": "costly"},
        },
    ]

    selected = await handler.async_get_available_deployments(
        model_group="test-model-group", healthy_deployments=deployments
    )

    assert selected["model_info"]["id"] == "cheap"


@pytest.mark.asyncio
async def test_string_input_cost_with_default_output_cost_does_not_raise():
    handler = LowestCostLoggingHandler(router_cache=DualCache())
    deployments = [
        {
            "model_name": "test-model-group",
            "litellm_params": {
                "model": "test/unknown-model",
                "input_cost_per_token": "0.000009",
            },
            "model_info": {"id": "mixed"},
        }
    ]

    selected = await handler.async_get_available_deployments(
        model_group="test-model-group", healthy_deployments=deployments
    )

    assert selected is not None


@pytest.mark.asyncio
async def test_unparsable_cost_falls_back_to_the_model_cost_map():
    handler = LowestCostLoggingHandler(router_cache=DualCache())
    deployments = [
        {
            "model_name": "test-model-group",
            "litellm_params": {
                "model": "test/unknown-model",
                "input_cost_per_token": "not-a-number",
                "output_cost_per_token": "also-not-a-number",
            },
            "model_info": {"id": "unparsable"},
        },
        {
            "model_name": "test-model-group",
            "litellm_params": {
                "model": "test/unknown-model",
                "input_cost_per_token": 0.000001,
                "output_cost_per_token": 0.000001,
            },
            "model_info": {"id": "cheap"},
        },
    ]

    selected = await handler.async_get_available_deployments(
        model_group="test-model-group", healthy_deployments=deployments
    )

    assert selected["model_info"]["id"] == "cheap"
