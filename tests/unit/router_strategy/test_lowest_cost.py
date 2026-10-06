import copy
from datetime import datetime

import pytest

import litellm
from litellm.caching.caching import DualCache
from litellm.router_strategy.lowest_cost import LowestCostLoggingHandler

DEPLOYMENT_ID = "9876"
COST_KEY = "cost_map:gpt-5.5-pool"
LATENCY_KEYS = ("gpt-5.5-pool_map", "gpt-5.5-pool_cost_map")
KWARGS = {
    "litellm_params": {
        "metadata": {"model_group": "gpt-5.5-pool"},
        "model_info": {"id": DEPLOYMENT_ID},
    }
}


def _chat_response_with_no_completion_tokens() -> litellm.ModelResponse:
    return litellm.ModelResponse(
        model="gpt-5.5",
        choices=[{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "length"}],
        usage=litellm.Usage(prompt_tokens=12, completion_tokens=0, total_tokens=12),
    )


def _recorded_minute_counters(cache: DualCache) -> dict[str, int]:
    cached = cache.get_cache(key=COST_KEY) or {}
    minute_buckets = cached.get(DEPLOYMENT_ID, {})
    assert len(minute_buckets) == 1, f"expected one minute bucket, got {minute_buckets}"
    return next(iter(minute_buckets.values()))


def test_log_success_event_counts_a_response_with_no_completion_tokens():
    cache = DualCache()
    handler = LowestCostLoggingHandler(router_cache=cache)

    handler.log_success_event(
        kwargs=KWARGS,
        response_obj=_chat_response_with_no_completion_tokens(),
        start_time=datetime(2026, 1, 1, 12, 0, 0),
        end_time=datetime(2026, 1, 1, 12, 0, 2),
    )

    assert _recorded_minute_counters(cache) == {"tpm": 12, "rpm": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_log_success_event_keeps_cost_bookkeeping_out_of_the_latency_routing_entry(use_async: bool):
    cache = DualCache()
    latency_entry = {DEPLOYMENT_ID: {"latency": [0.5], "time_to_first_token": [0.1]}}
    for latency_key in LATENCY_KEYS:
        cache.set_cache(key=latency_key, value=copy.deepcopy(latency_entry))
    handler = LowestCostLoggingHandler(router_cache=cache)
    call_args = {
        "kwargs": KWARGS,
        "response_obj": _chat_response_with_no_completion_tokens(),
        "start_time": datetime(2026, 1, 1, 12, 0, 0),
        "end_time": datetime(2026, 1, 1, 12, 0, 2),
    }

    if use_async:
        await handler.async_log_success_event(**call_args)
    else:
        handler.log_success_event(**call_args)

    assert [cache.get_cache(key=latency_key) for latency_key in LATENCY_KEYS] == [latency_entry, latency_entry]
    assert _recorded_minute_counters(cache) == {"tpm": 12, "rpm": 1}


@pytest.mark.asyncio
async def test_async_get_available_deployments_applies_rpm_limit_from_the_cost_entry():
    cache = DualCache()
    handler = LowestCostLoggingHandler(router_cache=cache)
    precise_minute = datetime.now().strftime("%Y-%m-%d-%H-%M")
    cache.set_cache(key=COST_KEY, value={DEPLOYMENT_ID: {precise_minute: {"tpm": 12, "rpm": 1}}})
    healthy_deployments = [{"model_info": {"id": DEPLOYMENT_ID}, "litellm_params": {"model": "gpt-5.5", "rpm": 1}}]

    picked = await handler.async_get_available_deployments(
        model_group="gpt-5.5-pool",
        healthy_deployments=healthy_deployments,
        messages=[{"role": "user", "content": "hi"}],
    )

    assert picked is None


@pytest.mark.asyncio
async def test_async_log_success_event_counts_a_response_with_no_completion_tokens():
    cache = DualCache()
    handler = LowestCostLoggingHandler(router_cache=cache)

    await handler.async_log_success_event(
        kwargs=KWARGS,
        response_obj=_chat_response_with_no_completion_tokens(),
        start_time=datetime(2026, 1, 1, 12, 0, 0),
        end_time=datetime(2026, 1, 1, 12, 0, 2),
    )

    assert _recorded_minute_counters(cache) == {"tpm": 12, "rpm": 1}


#    cost-based-routing scored deployments on input + output price alone, so two
#    deployments whose input and output prices tie were ordered only by their position
#    in model_list. cache_read_input_token_cost can differ 10x across such a tie, which
#    is the price that dominates cache-heavy traffic. Issue #38064.

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


@pytest.fixture
def tied_prices_no_cache_entry():
    """Same tie, but the cost map carries an explicit null cache-read price."""
    for name in (PRICEY_CACHE, CHEAP_CACHE):
        litellm.register_model(
            {
                f"openai/{name}": {
                    "litellm_provider": "openai",
                    "mode": "chat",
                    "input_cost_per_token": 1.4e-07,
                    "output_cost_per_token": 2.8e-07,
                }
            }
        )
        litellm.model_cost[f"openai/{name}"]["cache_read_input_token_cost"] = None
    yield
    for name in (PRICEY_CACHE, CHEAP_CACHE):
        litellm.model_cost.pop(f"openai/{name}", None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "order",
    [[PRICEY_CACHE, CHEAP_CACHE], [CHEAP_CACHE, PRICEY_CACHE]],
    ids=["pricey-listed-first", "cheap-listed-first"],
)
async def test_deployment_level_cache_read_override_breaks_the_tie(order, tied_prices):
    """A per-deployment cache-read price must win over the shared model-map entry, the same way
    input and output prices already honour litellm_params."""
    handler = LowestCostLoggingHandler(router_cache=DualCache())
    overrides = {PRICEY_CACHE: 9e-07, CHEAP_CACHE: 1e-09}
    healthy = [
        {
            "model_name": MODEL_GROUP,
            "litellm_params": {
                "model": f"openai/{PRICEY_CACHE}",
                "cache_read_input_token_cost": overrides[name],
            },
            "model_info": {"id": name},
        }
        for name in order
    ]

    selected = await handler.async_get_available_deployments(model_group=MODEL_GROUP, healthy_deployments=healthy)

    assert selected["model_info"]["id"] == CHEAP_CACHE


@pytest.mark.asyncio
async def test_null_cache_read_price_does_not_break_sorting(tied_prices_no_cache_entry):
    """An explicitly null cache-read price means unset, not zero, so it must fall back to the
    input price instead of putting None into the sort key and raising TypeError."""
    handler = LowestCostLoggingHandler(router_cache=DualCache())

    selected = await handler.async_get_available_deployments(
        model_group=MODEL_GROUP, healthy_deployments=_deployments([PRICEY_CACHE, CHEAP_CACHE])
    )

    assert selected["model_info"]["id"] in (PRICEY_CACHE, CHEAP_CACHE)
