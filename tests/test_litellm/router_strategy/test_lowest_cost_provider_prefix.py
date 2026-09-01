from typing import Final

import pytest

import litellm
from litellm.caching.caching import DualCache
from litellm.router_strategy.lowest_cost import LowestCostLoggingHandler
from litellm.types.utils import ModelInfo


@pytest.fixture(autouse=True)
def _cleanup_model_cost():
    keys_before = set(litellm.model_cost.keys())
    yield
    keys_after = set(litellm.model_cost.keys())
    for key in keys_after - keys_before:
        del litellm.model_cost[key]


@pytest.mark.asyncio
async def test_lowest_cost_routing_resolves_provider_prefixed_model():
    litellm.model_cost["test-gpt-luna"] = {
        "input_cost_per_token": 2e-07,
        "output_cost_per_token": 2e-07,
        "litellm_provider": "openai",
        "mode": "chat",
    }
    litellm.model_cost["test-gpt-sol"] = {
        "input_cost_per_token": 5e-06,
        "output_cost_per_token": 5e-06,
        "litellm_provider": "openai",
        "mode": "chat",
    }

    deployments = [
        {
            "model_name": "test-group",
            "litellm_params": {"model": "openai/test-gpt-sol"},
            "model_info": {"id": "sol"},
        },
        {
            "model_name": "test-group",
            "litellm_params": {"model": "openai/test-gpt-luna"},
            "model_info": {"id": "luna"},
        },
    ]

    handler = LowestCostLoggingHandler(router_cache=DualCache())
    selected = await handler.async_get_available_deployments(
        model_group="test-group",
        healthy_deployments=deployments,
    )

    assert selected is not None
    assert selected["model_info"]["id"] == "luna"


@pytest.mark.asyncio
async def test_lowest_cost_routing_keeps_provider_pricing_isolated(monkeypatch: pytest.MonkeyPatch):
    shared_model: Final = "shared-provider-model"

    def provider_model_info(model: str, custom_llm_provider: str | None = None) -> ModelInfo:
        assert model == shared_model
        assert custom_llm_provider is not None
        cost: Final = 1.0 if custom_llm_provider == "expensive-provider" else 1e-07
        return {
            "input_cost_per_token": cost,
            "output_cost_per_token": cost,
            "litellm_provider": custom_llm_provider,
            "mode": "chat",
            "supported_openai_params": None,
        }

    monkeypatch.setattr(litellm, "get_model_info", provider_model_info)

    deployments = [
        {
            "model_name": "test-provider-group",
            "litellm_params": {
                "model": shared_model,
                "custom_llm_provider": "expensive-provider",
            },
            "model_info": {"id": "expensive"},
        },
        {
            "model_name": "test-provider-group",
            "litellm_params": {
                "model": shared_model,
                "custom_llm_provider": "cheap-provider",
            },
            "model_info": {"id": "cheap"},
        },
    ]

    handler = LowestCostLoggingHandler(router_cache=DualCache())
    selected = await handler.async_get_available_deployments(
        model_group="test-provider-group",
        healthy_deployments=deployments,
    )

    assert selected is not None
    assert selected["model_info"]["id"] == "cheap"
    assert shared_model not in litellm.model_cost


@pytest.mark.asyncio
async def test_lowest_cost_routing_uses_direct_match():
    litellm.model_cost["openai/test-direct-match"] = {
        "input_cost_per_token": 1e-07,
        "output_cost_per_token": 1e-07,
        "litellm_provider": "openai",
        "mode": "chat",
    }

    deployments = [
        {
            "model_name": "test-direct",
            "litellm_params": {"model": "openai/test-direct-match"},
            "model_info": {"id": "direct-1"},
        },
    ]

    handler = LowestCostLoggingHandler(router_cache=DualCache())
    selected = await handler.async_get_available_deployments(
        model_group="test-direct",
        healthy_deployments=deployments,
    )

    assert selected is not None
    assert selected["model_info"]["id"] == "direct-1"


@pytest.mark.asyncio
async def test_lowest_cost_routing_fallback_for_unmapped_model():
    deployments = [
        {
            "model_name": "test-group-unmapped",
            "litellm_params": {"model": "custom-unknown-provider/unknown-model-xyz"},
            "model_info": {"id": "unmapped-1"},
        },
    ]

    handler = LowestCostLoggingHandler(router_cache=DualCache())
    selected = await handler.async_get_available_deployments(
        model_group="test-group-unmapped",
        healthy_deployments=deployments,
    )

    assert selected is not None
    assert selected["model_info"]["id"] == "unmapped-1"


@pytest.mark.asyncio
async def test_lowest_cost_routing_explicit_params_override():
    litellm.model_cost["test-expensive-base"] = {
        "input_cost_per_token": 1.0,
        "output_cost_per_token": 1.0,
        "litellm_provider": "openai",
        "mode": "chat",
    }

    deployments = [
        {
            "model_name": "test-override",
            "litellm_params": {
                "model": "openai/test-expensive-base",
                "input_cost_per_token": 1e-08,
                "output_cost_per_token": 1e-08,
            },
            "model_info": {"id": "discounted-deployment"},
        },
    ]

    handler = LowestCostLoggingHandler(router_cache=DualCache())
    selected = await handler.async_get_available_deployments(
        model_group="test-override",
        healthy_deployments=deployments,
    )

    assert selected is not None
    assert selected["model_info"]["id"] == "discounted-deployment"
