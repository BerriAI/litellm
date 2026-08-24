#### What this tests ####
#    cost-based routing must break input+output price ties on cache-read price

import pytest

from litellm.caching.caching import DualCache
from litellm.router_strategy.lowest_cost import LowestCostLoggingHandler


@pytest.mark.parametrize("cheaper_cache_first", [True, False])
@pytest.mark.asyncio
async def test_cost_routing_breaks_input_output_tie_on_cache_read_cost(cheaper_cache_first):
    """
    Regression test for https://github.com/BerriAI/litellm/issues/38064

    Two deployments with identical input+output price must be separated by their
    cache-read price, not by whichever one happens to be listed first. The pricier
    deployment omits a cache-read price to exercise the input-cost fallback.
    """
    cheaper = {
        "model_name": "cache-tie-test",
        "litellm_params": {
            "model": "openai/tie-model-not-in-cost-map",
            "input_cost_per_token": 1e-06,
            "output_cost_per_token": 2e-06,
            "cache_read_input_token_cost": 1e-08,
        },
        "model_info": {"id": "cheaper-cache"},
    }
    pricier = {
        "model_name": "cache-tie-test",
        "litellm_params": {
            "model": "openai/tie-model-not-in-cost-map",
            "input_cost_per_token": 1e-06,
            "output_cost_per_token": 2e-06,
        },
        "model_info": {"id": "pricier-cache"},
    }
    model_list = [cheaper, pricier] if cheaper_cache_first else [pricier, cheaper]

    logger = LowestCostLoggingHandler(router_cache=DualCache())

    selected = await logger.async_get_available_deployments(
        model_group="cache-tie-test", healthy_deployments=model_list
    )

    assert selected["model_info"]["id"] == "cheaper-cache"
