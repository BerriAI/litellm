import pytest

from litellm.caching.caching import DualCache
from litellm.router_strategy.lowest_cost import LowestCostLoggingHandler


@pytest.mark.asyncio
async def test_provider_alias_uses_canonical_model_cost() -> None:
    deployments = [
        {
            "model_name": "cost-routing-models",
            "litellm_params": {"model": "cohere/command-r"},
            "model_info": {"id": "provider-alias-model"},
        },
        {
            "model_name": "cost-routing-models",
            "litellm_params": {
                "model": "gpt-4",
                "input_cost_per_token": 1.0,
                "output_cost_per_token": 1.0,
            },
            "model_info": {"id": "explicit-cost-model"},
        },
    ]
    handler = LowestCostLoggingHandler(router_cache=DualCache())

    selected = await handler.async_get_available_deployments(
        model_group="cost-routing-models", healthy_deployments=deployments
    )

    assert selected["model_info"]["id"] == "provider-alias-model"


@pytest.mark.asyncio
async def test_provider_prefixed_model_uses_canonical_model_cost() -> None:
    deployments = [
        {
            "model_name": "cost-routing-models",
            "litellm_params": {"model": "openai/gpt-5.6-luna"},
            "model_info": {"id": "provider-prefixed-model"},
        },
        {
            "model_name": "cost-routing-models",
            "litellm_params": {"model": "gpt-4"},
            "model_info": {"id": "unprefixed-model"},
        },
    ]
    handler = LowestCostLoggingHandler(router_cache=DualCache())

    selected = await handler.async_get_available_deployments(
        model_group="cost-routing-models", healthy_deployments=deployments
    )

    assert selected["model_info"]["id"] == "provider-prefixed-model"


@pytest.mark.asyncio
async def test_unknown_provider_prefixed_model_keeps_fallback_cost() -> None:
    deployments = [
        {
            "model_name": "cost-routing-models",
            "litellm_params": {"model": "openai/not-in-model-cost-map"},
            "model_info": {"id": "unknown-model"},
        },
        {
            "model_name": "cost-routing-models",
            "litellm_params": {
                "model": "gpt-4",
                "input_cost_per_token": 6.0,
                "output_cost_per_token": 6.0,
            },
            "model_info": {"id": "explicit-cost-model"},
        },
    ]
    handler = LowestCostLoggingHandler(router_cache=DualCache())

    selected = await handler.async_get_available_deployments(
        model_group="cost-routing-models", healthy_deployments=deployments
    )

    assert selected["model_info"]["id"] == "unknown-model"


@pytest.mark.asyncio
async def test_mismatched_provider_does_not_use_canonical_model_cost() -> None:
    deployments = [
        {
            "model_name": "cost-routing-models",
            "litellm_params": {"model": "groq/gpt-4"},
            "model_info": {"id": "mismatched-provider-model"},
        },
        {
            "model_name": "cost-routing-models",
            "litellm_params": {
                "model": "gpt-4",
                "input_cost_per_token": 1.0,
                "output_cost_per_token": 1.0,
            },
            "model_info": {"id": "explicit-cost-model"},
        },
    ]
    handler = LowestCostLoggingHandler(router_cache=DualCache())

    selected = await handler.async_get_available_deployments(
        model_group="cost-routing-models", healthy_deployments=deployments
    )

    assert selected["model_info"]["id"] == "explicit-cost-model"
