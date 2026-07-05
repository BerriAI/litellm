"""
Regression tests for #26701

/v1/messages and /v1/embeddings routes don't inject custom_llm_provider into
litellm_params (unlike /v1/chat/completions). async_log_success_event must
derive the provider from the model string instead of raising ValueError.
"""

import pytest
from unittest.mock import AsyncMock, patch

from litellm.caching.caching import DualCache
from litellm.router_strategy.budget_limiter import RouterBudgetLimiting


def _make_limiter():
    return RouterBudgetLimiting(
        dual_cache=DualCache(),
        provider_budget_config={
            "anthropic": {"budget_limit": 5.0, "time_period": "24h"}
        },
    )


def _kwargs_without_custom_provider(model_str: str) -> dict:
    """Simulates what /v1/messages and /v1/embeddings produce — no custom_llm_provider."""
    return {
        "model": model_str,
        "litellm_params": {
            "model": model_str,
            # custom_llm_provider deliberately absent — this is the bug scenario
        },
        "standard_logging_object": {
            "response_cost": 0.001,
            "model_id": "deployment-abc123",
        },
    }


@pytest.mark.asyncio
async def test_messages_route_no_custom_provider_does_not_raise():
    """
    /v1/messages kwargs missing custom_llm_provider must not raise ValueError.
    Provider should be derived from the model string 'anthropic/...' instead.
    """
    limiter = _make_limiter()
    kwargs = _kwargs_without_custom_provider("anthropic/claude-haiku-4-5-20251001")

    with (
        patch.object(
            limiter, "_increment_spend_for_key", new_callable=AsyncMock
        ) as mock_increment,
        patch.object(limiter, "_get_budget_config_for_deployment", return_value=None),
        patch.object(limiter, "_get_budget_config_for_tag", return_value=None),
    ):

        # Before the fix this raises: ValueError("custom_llm_provider is required")
        await limiter.async_log_success_event(
            kwargs=kwargs,
            response_obj=None,
            start_time=None,
            end_time=None,
        )

        # Provider was correctly derived → spend increment was called
        mock_increment.assert_called_once()


@pytest.mark.asyncio
async def test_embeddings_route_no_custom_provider_does_not_raise():
    """/v1/embeddings has the same gap — openai provider should be derived cleanly."""
    limiter = _make_limiter()
    kwargs = _kwargs_without_custom_provider("openai/text-embedding-3-small")

    with (
        patch.object(limiter, "_increment_spend_for_key", new_callable=AsyncMock),
        patch.object(limiter, "_get_budget_config_for_deployment", return_value=None),
        patch.object(limiter, "_get_budget_config_for_tag", return_value=None),
    ):

        await limiter.async_log_success_event(
            kwargs=kwargs,
            response_obj=None,
            start_time=None,
            end_time=None,
        )


@pytest.mark.asyncio
async def test_chat_completions_with_explicit_provider_still_works():
    """
    /v1/chat/completions path that already sets custom_llm_provider must
    continue to work exactly as before — no regression.
    """
    limiter = _make_limiter()
    kwargs = {
        "model": "claude-haiku-direct",
        "litellm_params": {
            "model": "anthropic/claude-haiku-4-5-20251001",
            "custom_llm_provider": "anthropic",  # set by chat/completions route
        },
        "standard_logging_object": {
            "response_cost": 0.001,
            "model_id": "deployment-abc123",
        },
    }

    with (
        patch.object(
            limiter, "_increment_spend_for_key", new_callable=AsyncMock
        ) as mock_increment,
        patch.object(limiter, "_get_budget_config_for_deployment", return_value=None),
        patch.object(limiter, "_get_budget_config_for_tag", return_value=None),
    ):

        await limiter.async_log_success_event(
            kwargs=kwargs,
            response_obj=None,
            start_time=None,
            end_time=None,
        )

        mock_increment.assert_called_once()


@pytest.mark.asyncio
async def test_unresolvable_model_string_raises_value_error_with_debug_log():
    """
    When both the dict lookup and litellm.get_llm_provider() fail
    (completely unrecognisable model string), ValueError must still be raised
    and the debug log must fire — covering the except branch added in #26701.
    """
    limiter = _make_limiter()
    kwargs = _kwargs_without_custom_provider("totally-unresolvable-garbage-xyz-123")

    with (
        patch.object(limiter, "_increment_spend_for_key", new_callable=AsyncMock),
        patch.object(limiter, "_get_budget_config_for_deployment", return_value=None),
        patch.object(limiter, "_get_budget_config_for_tag", return_value=None),
    ):

        with pytest.raises(ValueError, match="custom_llm_provider is required"):
            await limiter.async_log_success_event(
                kwargs=kwargs,
                response_obj=None,
                start_time=None,
                end_time=None,
            )


@pytest.mark.asyncio
async def test_provider_derived_from_top_level_model_when_litellm_params_missing_model():
    """
    Regression for veria-ai review comment on #32180:
    LoggedLiteLLMParams for /v1/messages does not include `model`, so
    litellm_params.get("model") returns "". Must fall back to kwargs["model"]
    so that provider spend is actually incremented (budget enforced).
    """
    limiter = _make_limiter()

    kwargs = {
        "model": "anthropic/claude-haiku-4-5-20251001",  # top-level, always present
        "litellm_params": {
            # model key absent — this is the LoggedLiteLLMParams gap
        },
        "standard_logging_object": {
            "response_cost": 0.001,
            "model_id": "deployment-abc123",
        },
    }

    with patch.object(limiter, "_increment_spend_for_key", new_callable=AsyncMock) as mock_increment, \
         patch.object(limiter, "_get_budget_config_for_deployment", return_value=None), \
         patch.object(limiter, "_get_budget_config_for_tag", return_value=None):

        await limiter.async_log_success_event(
            kwargs=kwargs,
            response_obj=None,
            start_time=None,
            end_time=None,
        )

        # Budget spend MUST be incremented — if it isn't, budget enforcement is broken
        mock_increment.assert_called_once()
