"""
Regression tests for #26701.

`/v1/chat/completions` injects `custom_llm_provider` into `litellm_params`, but
`/v1/messages` and `/v1/embeddings` do not. Before the fix,
`RouterBudgetLimiting.async_log_success_event` raised
`ValueError("custom_llm_provider is required")` before incrementing any spend
counter, so provider budgets (and deployment-level `max_budget`, which sits
below the same guard) silently never accrued on those routes.

The standard logging payload carries `custom_llm_provider` for every route --
it is built from the top-level call details rather than from `litellm_params` --
so the limiter falls back to it.
"""

from unittest.mock import AsyncMock, patch

import pytest

from litellm.caching.caching import DualCache
from litellm.router_strategy.budget_limiter import RouterBudgetLimiting


def _make_limiter() -> RouterBudgetLimiting:
    return RouterBudgetLimiting(
        dual_cache=DualCache(),
        provider_budget_config={
            "anthropic": {"budget_limit": 5.0, "time_period": "24h"},
            "openai": {"budget_limit": 5.0, "time_period": "24h"},
        },
    )


def _payload(provider: str | None) -> dict:
    return {
        "response_cost": 0.001,
        "model_id": "deployment-abc123",
        "custom_llm_provider": provider,
    }


def _patched(limiter: RouterBudgetLimiting):
    """Isolate the provider-budget path from deployment/tag budget lookups."""
    return (
        patch.object(limiter, "_increment_spend_for_key", new_callable=AsyncMock),
        patch.object(limiter, "_get_budget_config_for_deployment", return_value=None),
        patch.object(limiter, "_get_budget_config_for_tag", return_value=None),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route", "model", "provider"),
    [
        ("/v1/messages", "anthropic/claude-haiku-4-5-20251001", "anthropic"),
        ("/v1/embeddings", "openai/text-embedding-3-small", "openai"),
    ],
)
async def test_provider_falls_back_to_logging_payload(route: str, model: str, provider: str):
    """
    Routes that omit `custom_llm_provider` from `litellm_params` must still
    accrue spend, using the provider recorded on the standard logging payload.
    """
    limiter = _make_limiter()
    kwargs = {
        "model": model,
        # `litellm_params` on these routes carries no `custom_llm_provider`.
        "litellm_params": {"model": model},
        "standard_logging_object": _payload(provider),
    }

    increment, _dep, _tag = _patched(limiter)
    with increment as mock_increment, _dep, _tag:
        # Before the fix this raised ValueError before any counter moved.
        await limiter.async_log_success_event(kwargs=kwargs, response_obj=None, start_time=None, end_time=None)

    assert mock_increment.await_count >= 1, f"{route}: provider spend was never incremented"
    spend_keys = [call.kwargs.get("spend_key", "") for call in mock_increment.await_args_list]
    assert any(f"provider_spend:{provider}:" in key for key in spend_keys), (
        f"{route}: expected a provider_spend key for {provider!r}, got {spend_keys}"
    )


@pytest.mark.asyncio
async def test_litellm_params_provider_takes_precedence():
    """
    `/v1/chat/completions` already sets `custom_llm_provider` in `litellm_params`;
    that value must keep winning over the payload so behaviour is unchanged.
    """
    limiter = _make_limiter()
    kwargs = {
        "model": "claude-haiku-direct",
        "litellm_params": {
            "model": "anthropic/claude-haiku-4-5-20251001",
            "custom_llm_provider": "anthropic",
        },
        # Deliberately disagrees, to prove precedence rather than coincidence.
        "standard_logging_object": _payload("openai"),
    }

    increment, _dep, _tag = _patched(limiter)
    with increment as mock_increment, _dep, _tag:
        await limiter.async_log_success_event(kwargs=kwargs, response_obj=None, start_time=None, end_time=None)

    spend_keys = [call.kwargs.get("spend_key", "") for call in mock_increment.await_args_list]
    assert any("provider_spend:anthropic:" in key for key in spend_keys), spend_keys
    assert not any("provider_spend:openai:" in key for key in spend_keys), spend_keys


@pytest.mark.asyncio
async def test_missing_provider_everywhere_still_raises():
    """
    With no provider in `litellm_params` and none on the payload there is
    nothing to attribute spend to; the existing contract must be preserved.
    """
    limiter = _make_limiter()
    kwargs = {
        "model": "totally-unresolvable-garbage-xyz-123",
        "litellm_params": {},
        "standard_logging_object": _payload(None),
    }

    increment, _dep, _tag = _patched(limiter)
    with increment, _dep, _tag, pytest.raises(ValueError, match="custom_llm_provider is required"):
        await limiter.async_log_success_event(kwargs=kwargs, response_obj=None, start_time=None, end_time=None)
