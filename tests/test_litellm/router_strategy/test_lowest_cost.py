from datetime import datetime

import pytest

import litellm
from litellm.caching.caching import DualCache
from litellm.router_strategy.lowest_cost import LowestCostLoggingHandler

DEPLOYMENT_ID = "9876"
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
    cached = cache.get_cache(key="gpt-5.5-pool_map") or {}
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
async def test_string_costs_are_sorted_numerically():
    handler = LowestCostLoggingHandler(router_cache=DualCache())
    deployments = [
        {
            "model_name": "test-model-group",
            "litellm_params": {
                "model": "test/unknown-model",
                "input_cost_per_token": "0.000009",
                "output_cost_per_token": "0.000009",
            },
            "model_info": {"id": "cheap"},
        },
        {
            "model_name": "test-model-group",
            "litellm_params": {
                "model": "test/unknown-model",
                "input_cost_per_token": "0.000008",
                "output_cost_per_token": "0.000012",
            },
            "model_info": {"id": "costly"},
        },
    ]

    selected = await handler.async_get_available_deployments(
        model_group="test-model-group", healthy_deployments=deployments
    )

    assert selected["model_info"]["id"] == "cheap"


@pytest.mark.asyncio
async def test_string_input_cost_with_default_output_cost_does_not_raise():
    handler = LowestCostLoggingHandler(router_cache=DualCache())
    deployments = [
        {
            "model_name": "test-model-group",
            "litellm_params": {
                "model": "test/unknown-model",
                "input_cost_per_token": "0.000009",
            },
            "model_info": {"id": "mixed"},
        }
    ]

    selected = await handler.async_get_available_deployments(
        model_group="test-model-group", healthy_deployments=deployments
    )

    assert selected is not None
