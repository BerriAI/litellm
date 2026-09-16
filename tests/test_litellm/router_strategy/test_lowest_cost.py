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


@pytest.mark.parametrize("free_cache_first", [True, False])
@pytest.mark.asyncio
async def test_cost_routing_honors_zero_deployment_cache_read_cost(free_cache_first):
    """
    A deployment-level cache_read_input_token_cost of 0 is a price, not a missing value.

    Truthiness checks treat it as unset and fall through to the cost map / input-cost
    fallback, which ranks a deployment whose cache reads are free as the priciest one
    in a tie. Providers that do not charge for cache reads make this a real config.
    """
    free = {
        "model_name": "cache-zero-test",
        "litellm_params": {
            "model": "openai/zero-model-not-in-cost-map",
            "input_cost_per_token": 1e-06,
            "output_cost_per_token": 2e-06,
            "cache_read_input_token_cost": 0.0,
        },
        "model_info": {"id": "free-cache"},
    }
    paid = {
        "model_name": "cache-zero-test",
        "litellm_params": {
            "model": "openai/zero-model-not-in-cost-map",
            "input_cost_per_token": 1e-06,
            "output_cost_per_token": 2e-06,
            "cache_read_input_token_cost": 1e-08,
        },
        "model_info": {"id": "paid-cache"},
    }
    model_list = [free, paid] if free_cache_first else [paid, free]

    logger = LowestCostLoggingHandler(router_cache=DualCache())

    selected = await logger.async_get_available_deployments(
        model_group="cache-zero-test", healthy_deployments=model_list
    )

    assert selected["model_info"]["id"] == "free-cache"
