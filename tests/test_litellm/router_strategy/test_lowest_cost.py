import pytest

from litellm.caching.caching import DualCache
from litellm.router_strategy.lowest_cost import LowestCostLoggingHandler


@pytest.mark.asyncio
async def test_provider_prefixed_models_use_resolved_costs() -> None:
    deployments = [
        {
            "model_name": "test-group",
            "litellm_params": {"model": "openai/gpt-5.6-sol"},
            "model_info": {"id": "sol"},
        },
        {
            "model_name": "test-group",
            "litellm_params": {"model": "openai/gpt-5.6-luna"},
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
