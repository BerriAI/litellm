from unittest.mock import patch

import litellm
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


@pytest.mark.asyncio
async def test_unknown_provider_model_does_not_query_dynamic_metadata() -> None:
    deployments = [
        {
            "model_name": "test-group",
            "litellm_params": {"model": "ollama/caller-controlled-model"},
            "model_info": {"id": "unknown"},
        },
        {
            "model_name": "test-group",
            "litellm_params": {"model": "openai/gpt-5.6-luna"},
            "model_info": {"id": "luna"},
        },
    ]
    handler = LowestCostLoggingHandler(router_cache=DualCache())

    with patch.object(litellm, "get_model_info") as get_model_info:
        selected = await handler.async_get_available_deployments(
            model_group="test-group",
            healthy_deployments=deployments,
        )

    assert selected is not None
    assert selected["model_info"]["id"] == "luna"
    get_model_info.assert_not_called()


@pytest.mark.asyncio
async def test_invalid_static_cost_entry_is_ignored() -> None:
    deployments = [
        {
            "model_name": "test-group",
            "litellm_params": {"model": "test-provider/corrupt-cost"},
            "model_info": {"id": "invalid"},
        },
        {
            "model_name": "test-group",
            "litellm_params": {"model": "openai/gpt-5.6-luna"},
            "model_info": {"id": "luna"},
        },
    ]
    handler = LowestCostLoggingHandler(router_cache=DualCache())

    with patch.dict(
        litellm.model_cost,
        {"test-provider/corrupt-cost": {"input_cost_per_token": "invalid"}},
    ):
        selected = await handler.async_get_available_deployments(
            model_group="test-group",
            healthy_deployments=deployments,
        )

    assert selected is not None
    assert selected["model_info"]["id"] == "luna"
