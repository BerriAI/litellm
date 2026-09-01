import pytest
import litellm
from litellm.caching.caching import DualCache
from litellm.router_strategy.lowest_cost import LowestCostLoggingHandler


@pytest.fixture(autouse=True)
def _cleanup_model_cost():
    """Snapshot litellm.model_cost before each test and restore it afterwards."""
    keys_before = set(litellm.model_cost.keys())
    yield
    keys_after = set(litellm.model_cost.keys())
    for key in keys_after - keys_before:
        del litellm.model_cost[key]


@pytest.mark.asyncio
async def test_lowest_cost_routing_resolves_provider_prefixed_model():
    """
    When model_cost only has the bare model name (e.g. 'test-gpt-luna'),
    the router should still resolve 'openai/test-gpt-luna' via get_model_info
    fallback and pick the cheapest deployment.
    """
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
async def test_lowest_cost_routing_caches_resolved_model():
    """
    After the first routing call resolves a provider-prefixed model via
    get_model_info, the result should be cached in litellm.model_cost
    so subsequent calls do not re-invoke get_model_info.
    """
    litellm.model_cost["test-gpt-cached"] = {
        "input_cost_per_token": 1e-07,
        "output_cost_per_token": 1e-07,
        "litellm_provider": "openai",
        "mode": "chat",
    }

    deployments = [
        {
            "model_name": "test-cache-group",
            "litellm_params": {"model": "openai/test-gpt-cached"},
            "model_info": {"id": "cached-1"},
        },
    ]

    handler = LowestCostLoggingHandler(router_cache=DualCache())

    # Before routing, the prefixed key should not exist in model_cost
    assert "openai/test-gpt-cached" not in litellm.model_cost

    await handler.async_get_available_deployments(
        model_group="test-cache-group",
        healthy_deployments=deployments,
    )

    # After routing, the prefixed key should now be cached in model_cost
    assert "openai/test-gpt-cached" in litellm.model_cost
    cached = litellm.model_cost["openai/test-gpt-cached"]
    assert cached["input_cost_per_token"] == 1e-07


@pytest.mark.asyncio
async def test_lowest_cost_routing_direct_match_no_fallback():
    """
    When the full model name (including provider prefix) already exists
    in model_cost, routing should use it directly without needing fallback.
    """
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
    """
    When a model cannot be resolved in model_cost or get_model_info,
    it should gracefully fall back to default 5.0 cost without crashing.
    """
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
    """
    Explicit input_cost_per_token and output_cost_per_token in litellm_params
    should override whatever is in model_cost.
    """
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
