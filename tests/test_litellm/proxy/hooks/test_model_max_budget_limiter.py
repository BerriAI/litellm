import pytest

import litellm
from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.model_max_budget_limiter import (
    _PROXY_VirtualKeyModelMaxBudgetLimiter,
)

VIRTUAL_KEY = "test-key-hash"

OPUS_GROUP_BUDGET = {
    "opus-family": {
        "models": ["anthropic-opus-4-7", "anthropic-opus-4-8"],
        "budget_limit": 10.0,
        "time_period": "30d",
    }
}


def _make_limiter() -> _PROXY_VirtualKeyModelMaxBudgetLimiter:
    return _PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=DualCache())


def _make_key(model_max_budget: dict) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        token=VIRTUAL_KEY,
        key_alias="test-alias",
        model_max_budget=model_max_budget,
    )


async def _log_spend(
    limiter: _PROXY_VirtualKeyModelMaxBudgetLimiter,
    model: str,
    response_cost: float,
    model_max_budget: dict,
) -> None:
    kwargs = {
        "standard_logging_object": {
            "response_cost": response_cost,
            "model": model,
            "metadata": {"user_api_key_hash": VIRTUAL_KEY},
        },
        "litellm_params": {"metadata": {"user_api_key_model_max_budget": model_max_budget}},
    }
    await limiter.async_log_success_event(kwargs, response_obj=None, start_time=None, end_time=None)


@pytest.mark.asyncio
async def test_group_budget_shared_across_models():
    limiter = _make_limiter()
    key = _make_key(OPUS_GROUP_BUDGET)

    await _log_spend(limiter, "anthropic-opus-4-7", 11.0, OPUS_GROUP_BUDGET)

    with pytest.raises(litellm.BudgetExceededError, match="model group=opus-family"):
        await limiter.is_key_within_model_budget(key, "anthropic-opus-4-7")
    with pytest.raises(litellm.BudgetExceededError, match="model group=opus-family"):
        await limiter.is_key_within_model_budget(key, "anthropic-opus-4-8")


@pytest.mark.asyncio
async def test_group_budget_ignores_models_outside_group():
    limiter = _make_limiter()
    key = _make_key(OPUS_GROUP_BUDGET)

    await _log_spend(limiter, "anthropic-opus-4-7", 11.0, OPUS_GROUP_BUDGET)

    assert await limiter.is_key_within_model_budget(key, "anthropic-sonnet-5") is True


@pytest.mark.asyncio
async def test_group_budget_within_budget_passes():
    limiter = _make_limiter()
    key = _make_key(OPUS_GROUP_BUDGET)

    await _log_spend(limiter, "anthropic-opus-4-7", 4.0, OPUS_GROUP_BUDGET)
    await _log_spend(limiter, "anthropic-opus-4-8", 6.0, OPUS_GROUP_BUDGET)

    assert await limiter.is_key_within_model_budget(key, "anthropic-opus-4-7") is True
    assert await limiter.is_key_within_model_budget(key, "anthropic-opus-4-8") is True


@pytest.mark.asyncio
async def test_group_spend_accumulates_across_models_in_one_pool():
    limiter = _make_limiter()
    key = _make_key(OPUS_GROUP_BUDGET)

    await _log_spend(limiter, "anthropic-opus-4-7", 6.0, OPUS_GROUP_BUDGET)
    assert await limiter.is_key_within_model_budget(key, "anthropic-opus-4-8") is True

    await _log_spend(limiter, "anthropic-opus-4-8", 6.0, OPUS_GROUP_BUDGET)
    with pytest.raises(litellm.BudgetExceededError, match="model group=opus-family"):
        await limiter.is_key_within_model_budget(key, "anthropic-opus-4-7")


@pytest.mark.asyncio
async def test_group_budget_matches_provider_prefixed_model():
    limiter = _make_limiter()
    key = _make_key(OPUS_GROUP_BUDGET)

    await _log_spend(limiter, "anthropic-opus-4-7", 11.0, OPUS_GROUP_BUDGET)

    with pytest.raises(litellm.BudgetExceededError, match="model group=opus-family"):
        await limiter.is_key_within_model_budget(key, "anthropic/anthropic-opus-4-8")


@pytest.mark.asyncio
async def test_group_entry_is_not_a_direct_model_budget():
    limiter = _make_limiter()
    key = _make_key(OPUS_GROUP_BUDGET)

    await _log_spend(limiter, "anthropic-opus-4-7", 11.0, OPUS_GROUP_BUDGET)

    assert await limiter.is_key_within_model_budget(key, "opus-family") is True


@pytest.mark.asyncio
async def test_direct_and_group_budgets_are_both_enforced():
    model_max_budget = {
        "anthropic-opus-4-7": {"budget_limit": 1.0, "time_period": "30d"},
        **OPUS_GROUP_BUDGET,
    }
    limiter = _make_limiter()
    key = _make_key(model_max_budget)

    await _log_spend(limiter, "anthropic-opus-4-7", 2.0, model_max_budget)

    with pytest.raises(litellm.BudgetExceededError, match="model=anthropic-opus-4-7"):
        await limiter.is_key_within_model_budget(key, "anthropic-opus-4-7")
    assert await limiter.is_key_within_model_budget(key, "anthropic-opus-4-8") is True


@pytest.mark.asyncio
async def test_group_budget_exceeded_even_when_direct_budget_is_fine():
    model_max_budget = {
        "anthropic-opus-4-7": {"budget_limit": 100.0, "time_period": "30d"},
        **OPUS_GROUP_BUDGET,
    }
    limiter = _make_limiter()
    key = _make_key(model_max_budget)

    await _log_spend(limiter, "anthropic-opus-4-8", 11.0, model_max_budget)

    with pytest.raises(litellm.BudgetExceededError, match="model group=opus-family"):
        await limiter.is_key_within_model_budget(key, "anthropic-opus-4-7")


@pytest.mark.asyncio
async def test_spend_at_exactly_group_budget_passes():
    limiter = _make_limiter()
    key = _make_key(OPUS_GROUP_BUDGET)

    await _log_spend(limiter, "anthropic-opus-4-7", 10.0, OPUS_GROUP_BUDGET)

    assert await limiter.is_key_within_model_budget(key, "anthropic-opus-4-8") is True


@pytest.mark.asyncio
async def test_group_budget_matches_bare_request_when_member_is_provider_qualified():
    qualified_group_budget = {
        "gpt4-family": {
            "models": ["openai/gpt-4", "openai/gpt-4o"],
            "budget_limit": 10.0,
            "time_period": "30d",
        }
    }
    limiter = _make_limiter()
    key = _make_key(qualified_group_budget)

    await _log_spend(limiter, "gpt-4", 11.0, qualified_group_budget)

    with pytest.raises(litellm.BudgetExceededError, match="model group=gpt4-family"):
        await limiter.is_key_within_model_budget(key, "gpt-4o")
    assert await limiter.is_key_within_model_budget(key, "gpt-5.5") is True
