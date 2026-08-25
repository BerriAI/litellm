#### What this tests ####
#    cost-based-routing scored deployments on input + output price alone, so two
#    deployments whose input and output prices tie were ordered only by their position
#    in model_list. cache_read_input_token_cost can differ 10x across such a tie, which
#    is the price that dominates cache-heavy traffic. Issue #38064.

import pytest

import litellm
from litellm.caching.caching import DualCache
from litellm.router_strategy.lowest_cost import LowestCostLoggingHandler

MODEL_GROUP = "test-tied-price-model"
CHEAP_CACHE = "cheap-cache-read"
PRICEY_CACHE = "pricey-cache-read"


def _deployments(order):
    return [
        {
            "model_name": MODEL_GROUP,
            "litellm_params": {"model": f"openai/{name}"},
            "model_info": {"id": name},
        }
        for name in order
    ]


@pytest.fixture
def tied_prices():
    """Two models with identical input and output prices, cache-read differing 10x."""
    for name, cache_read in ((PRICEY_CACHE, 2.8e-08), (CHEAP_CACHE, 2.8e-09)):
        litellm.register_model(
            {
                f"openai/{name}": {
                    "litellm_provider": "openai",
                    "mode": "chat",
                    "input_cost_per_token": 1.4e-07,
                    "output_cost_per_token": 2.8e-07,
                    "cache_read_input_token_cost": cache_read,
                }
            }
        )
    yield
    for name in (PRICEY_CACHE, CHEAP_CACHE):
        litellm.model_cost.pop(f"openai/{name}", None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "order",
    [[PRICEY_CACHE, CHEAP_CACHE], [CHEAP_CACHE, PRICEY_CACHE]],
    ids=["pricey-listed-first", "cheap-listed-first"],
)
async def test_tied_input_output_price_breaks_on_cache_read_not_list_order(order, tied_prices):
    handler = LowestCostLoggingHandler(router_cache=DualCache())
    healthy = _deployments(order)

    selected = await handler.async_get_available_deployments(model_group=MODEL_GROUP, healthy_deployments=healthy)

    assert selected["model_info"]["id"] == CHEAP_CACHE
