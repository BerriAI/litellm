from typing import Final

import pytest

from litellm.caching.caching import DualCache
from litellm.proxy.hooks.model_max_budget_limiter import (
    _PROXY_VirtualKeyModelMaxBudgetLimiter,
)
from litellm.types.utils import LiteLLMBatch, Usage

KEY_HASH: Final = "key-hash-batch"
USER_ID: Final = "user-batch"
MODEL_GROUP: Final = "batch-qa-primary"
BATCH_COST: Final = 2.925e-05
CHAT_COST: Final = 0.001
KEY_SPEND_KEY: Final = f"virtual_key_spend:{KEY_HASH}:{MODEL_GROUP}:1d"
USER_SPEND_KEY: Final = f"user_model_spend:{USER_ID}:{MODEL_GROUP}:1d"


def _batch(batch_id: str, status: str) -> LiteLLMBatch:
    return LiteLLMBatch(
        id=batch_id,
        completion_window="24h",
        created_at=1,
        endpoint="/v1/chat/completions",
        input_file_id="file-batch",
        object="batch",
        status=status,
        usage=Usage(prompt_tokens=20, completion_tokens=18, total_tokens=38),
    )


def _event(call_type: str, response_cost: float) -> dict:
    return {
        "call_type": call_type,
        "standard_logging_object": {
            "call_type": call_type,
            "response_cost": response_cost,
            "model": "openai/gpt-5.4-mini",
            "model_group": MODEL_GROUP,
            "metadata": {"user_api_key_hash": KEY_HASH, "user_api_key_user_id": USER_ID},
        },
        "litellm_params": {
            "metadata": {
                "user_api_key_model_max_budget": {MODEL_GROUP: {"budget_limit": 0.0001, "time_period": "1d"}},
                "user_api_key_user_model_max_budget": {MODEL_GROUP: {"budget_limit": 0.0001, "time_period": "1d"}},
            }
        },
    }


async def _poll(limiter: _PROXY_VirtualKeyModelMaxBudgetLimiter, batch: LiteLLMBatch, response_cost: float) -> None:
    await limiter.async_log_success_event(
        _event("aretrieve_batch", response_cost), response_obj=batch, start_time=None, end_time=None
    )


async def _chat(limiter: _PROXY_VirtualKeyModelMaxBudgetLimiter) -> None:
    await limiter.async_log_success_event(_event("acompletion", CHAT_COST), response_obj=None, start_time=None, end_time=None)


async def _spend(limiter: _PROXY_VirtualKeyModelMaxBudgetLimiter, spend_key: str) -> float:
    return await limiter.dual_cache.async_get_cache(key=spend_key) or 0.0


@pytest.mark.asyncio
async def test_polls_of_a_finished_batch_charge_each_per_model_budget_once():
    limiter: Final = _PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=DualCache())
    first: Final = _batch("batch_first", "completed")

    await _poll(limiter, _batch("batch_first", "in_progress"), response_cost=0)
    for _ in range(3):
        await _poll(limiter, first, response_cost=BATCH_COST)

    assert await _spend(limiter, KEY_SPEND_KEY) == pytest.approx(BATCH_COST)
    assert await _spend(limiter, USER_SPEND_KEY) == pytest.approx(BATCH_COST)


@pytest.mark.asyncio
async def test_a_second_batch_and_chat_requests_still_charge_the_budget():
    limiter: Final = _PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=DualCache())

    await _poll(limiter, _batch("batch_first", "completed"), response_cost=BATCH_COST)
    await _poll(limiter, _batch("batch_first", "completed"), response_cost=BATCH_COST)
    await _poll(limiter, _batch("batch_second", "completed"), response_cost=BATCH_COST)
    await _chat(limiter)
    await _chat(limiter)

    assert await _spend(limiter, KEY_SPEND_KEY) == pytest.approx(2 * BATCH_COST + 2 * CHAT_COST)
