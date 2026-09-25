import copy
from datetime import datetime
from unittest.mock import patch

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


@pytest.mark.parametrize(
    "unknown_model",
    [None, "caller-controlled-model", "ollama/caller-controlled-model"],
)
@pytest.mark.asyncio
async def test_unknown_models_do_not_query_dynamic_metadata(
    unknown_model: str | None,
) -> None:
    deployments = [
        {
            "model_name": "test-group",
            "litellm_params": {"model": unknown_model},
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
async def test_exact_custom_cost_entry_remains_authoritative() -> None:
    deployments = [
        {
            "model_name": "test-group",
            "litellm_params": {"model": "custom/provider-entry"},
            "model_info": {"id": "custom"},
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
        {
            "custom/provider-entry": {
                "input_cost_per_token": 1e-9,
                "output_cost_per_token": 1e-9,
            }
        },
    ):
        selected = await handler.async_get_available_deployments(
            model_group="test-group",
            healthy_deployments=deployments,
        )

    assert selected is not None
    assert selected["model_info"]["id"] == "custom"


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
