import copy
from datetime import datetime
from typing import Final
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


@pytest.mark.parametrize(
    ("candidate_model", "candidate_price", "expected_id"),
    (
        ("openai/cheap", None, "candidate"),
        ("other/cheap", None, "reference"),
        (None, None, "reference"),
        ("unknown", None, "reference"),
        ("ollama/unknown", None, "reference"),
        ("custom/cheap", {"input_cost_per_token": 0.01, "output_cost_per_token": 0.02}, "candidate"),
        ("openai/cheap", {"input_cost_per_token": 2.0, "output_cost_per_token": 3.0}, "reference"),
        ("custom/broken", {"input_cost_per_token": "invalid"}, "reference"),
        ("custom/null", {"input_cost_per_token": None, "output_cost_per_token": None}, "reference"),
    ),
)
@pytest.mark.asyncio
@pytest.mark.parametrize("reverse_order", [False, True], ids=["candidate-first", "reference-first"])
async def test_provider_prefix_uses_matching_static_prices_and_preserves_exact_entries(
    candidate_model: str | None,
    candidate_price: dict[str, float | str | None] | None,
    expected_id: str,
    reverse_order: bool,
) -> None:
    deployments: Final = [
        {"litellm_params": {"model": candidate_model}, "model_info": {"id": "candidate"}},
        {"litellm_params": {"model": "openai/reference"}, "model_info": {"id": "reference"}},
    ]
    prices: Final = {
        "cheap": {"input_cost_per_token": 0.01, "output_cost_per_token": 0.02, "litellm_provider": "openai"},
        "reference": {"input_cost_per_token": 0.2, "output_cost_per_token": 0.3, "litellm_provider": "openai"},
        **({candidate_model: candidate_price} if candidate_model is not None and candidate_price is not None else {}),
    }
    handler: Final = LowestCostLoggingHandler(router_cache=DualCache())

    with patch.dict(litellm.model_cost, prices, clear=True), patch("litellm.get_model_info") as metadata_lookup:
        selected: Final = await handler.async_get_available_deployments(
            model_group="test-group", healthy_deployments=deployments[::-1] if reverse_order else deployments
        )

    expected: Final = next(deployment for deployment in deployments if deployment["model_info"]["id"] == expected_id)
    assert selected is expected
    metadata_lookup.assert_not_called()


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
