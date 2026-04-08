"""
Tests for the model vs model_group key mismatch fix on the write path.

The write path in async_log_success_event previously used
``standard_logging_payload["model"]`` (the deployment name, e.g.
``"deepseek/deepseek-chat"``) for cache keys.  Budget configs and the
read path use the public model name (``model_group``, e.g.
``"deepseek-chat"``), so the cache keys never matched when the
deployment name included a provider prefix.

The fix: prefer ``model_group`` over ``model`` on the write path.

These tests verify:
- Write path uses model_group when available
- Write path falls back to model when model_group is None
- Cache keys from write path match what the read path expects
- Spend from multiple deployments in the same group is aggregated
- End-to-end: spend written via model_group is visible on read
"""

import os
import sys
from unittest.mock import AsyncMock, patch

sys.path.insert(
    0, os.path.abspath("../../..")
)

import pytest

from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.model_max_budget_limiter import (
    END_USER_SPEND_CACHE_KEY_PREFIX,
    VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX,
    _PROXY_VirtualKeyModelMaxBudgetLimiter,
)


@pytest.fixture
def budget_limiter():
    dual_cache = DualCache()
    return _PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=dual_cache)


def _make_kwargs(
    *,
    model: str,
    model_group: str = None,
    response_cost: float = 0.05,
    virtual_key: str = "test-key-hash",
    end_user_id: str = None,
    key_budget: dict = None,
    end_user_budget: dict = None,
):
    """Build kwargs dict for async_log_success_event."""
    metadata = {"user_api_key_hash": virtual_key}
    if end_user_id:
        metadata["user_api_key_end_user_id"] = end_user_id

    slp = {
        "response_cost": response_cost,
        "model": model,
        "metadata": metadata,
    }
    if model_group is not None:
        slp["model_group"] = model_group
    if end_user_id:
        slp["end_user"] = end_user_id

    litellm_metadata = {}
    if key_budget is not None:
        litellm_metadata["user_api_key_model_max_budget"] = key_budget
    if end_user_budget is not None:
        litellm_metadata["user_api_key_end_user_model_max_budget"] = end_user_budget

    return {
        "standard_logging_object": slp,
        "litellm_params": {"metadata": litellm_metadata},
    }


# Real model names from model_prices_and_context_window.json:
#   deepseek/deepseek-chat:              input=$0.28/M, output=$0.42/M
#   openrouter/deepseek/deepseek-chat:   input=$0.14/M, output=$0.28/M
# Both are deployments a user might group under model_name "deepseek-chat".

_MODEL_GROUP = "deepseek-chat"
_DEPLOYMENT_A = "deepseek/deepseek-chat"  # single-slash prefix
_DEPLOYMENT_B = "openrouter/deepseek/deepseek-chat"  # multi-slash prefix
_BUDGET = {_MODEL_GROUP: {"budget_limit": 100.0, "time_period": "1d"}}


class TestWritePathUsesModelGroup:
    """The write path should use model_group (public name) for cache keys."""

    @pytest.mark.asyncio
    async def test_should_use_model_group_for_virtual_key_spend_key(
        self, budget_limiter
    ):
        """When model_group is set, the spend_key passed to
        _increment_spend_for_key should use model_group, not model."""
        kwargs = _make_kwargs(
            model=_DEPLOYMENT_A,
            model_group=_MODEL_GROUP,
            key_budget=_BUDGET,
        )

        with patch.object(
            budget_limiter,
            "_increment_spend_for_key",
            new_callable=AsyncMock,
        ) as mock_increment:
            await budget_limiter.async_log_success_event(
                kwargs, response_obj=None, start_time=None, end_time=None
            )
            mock_increment.assert_awaited_once()
            spend_key = mock_increment.call_args.kwargs["spend_key"]
            assert f":{_MODEL_GROUP}:" in spend_key
            assert _DEPLOYMENT_A not in spend_key

    @pytest.mark.asyncio
    async def test_should_use_model_group_with_multi_slash_deployment(
        self, budget_limiter
    ):
        """Deployment names with multiple slashes (e.g.
        openrouter/deepseek/deepseek-chat) should still use model_group."""
        kwargs = _make_kwargs(
            model=_DEPLOYMENT_B,
            model_group=_MODEL_GROUP,
            key_budget=_BUDGET,
        )

        with patch.object(
            budget_limiter,
            "_increment_spend_for_key",
            new_callable=AsyncMock,
        ) as mock_increment:
            await budget_limiter.async_log_success_event(
                kwargs, response_obj=None, start_time=None, end_time=None
            )
            mock_increment.assert_awaited_once()
            spend_key = mock_increment.call_args.kwargs["spend_key"]
            assert f":{_MODEL_GROUP}:" in spend_key
            assert _DEPLOYMENT_B not in spend_key

    @pytest.mark.asyncio
    async def test_should_use_model_group_for_end_user_spend_key(
        self, budget_limiter
    ):
        """End-user spend key should also use model_group."""
        kwargs = _make_kwargs(
            model=_DEPLOYMENT_A,
            model_group=_MODEL_GROUP,
            end_user_id="user-1",
            end_user_budget=_BUDGET,
        )

        with patch.object(
            budget_limiter,
            "_increment_spend_for_key",
            new_callable=AsyncMock,
        ) as mock_increment:
            await budget_limiter.async_log_success_event(
                kwargs, response_obj=None, start_time=None, end_time=None
            )
            mock_increment.assert_awaited_once()
            spend_key = mock_increment.call_args.kwargs["spend_key"]
            assert f":{_MODEL_GROUP}:" in spend_key
            assert _DEPLOYMENT_A not in spend_key

    @pytest.mark.asyncio
    async def test_should_fall_back_to_model_when_model_group_is_none(
        self, budget_limiter
    ):
        """When model_group is not in the payload, fall back to model."""
        kwargs = _make_kwargs(
            model=_MODEL_GROUP,
            model_group=None,
            key_budget=_BUDGET,
        )

        with patch.object(
            budget_limiter,
            "_increment_spend_for_key",
            new_callable=AsyncMock,
        ) as mock_increment:
            await budget_limiter.async_log_success_event(
                kwargs, response_obj=None, start_time=None, end_time=None
            )
            mock_increment.assert_awaited_once()
            spend_key = mock_increment.call_args.kwargs["spend_key"]
            assert f":{_MODEL_GROUP}:" in spend_key


class TestWriteReadCacheKeyAlignment:
    """Verify that what the write path writes can be found by the read path."""

    @pytest.mark.asyncio
    async def test_should_produce_cache_key_readable_by_virtual_key_lookup(
        self, budget_limiter
    ):
        """The cache key from the write path should exactly match
        what _get_virtual_key_spend_for_model reads."""
        virtual_key = "sk-hash-123"

        expected_cache_key = (
            f"{VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX}:{virtual_key}"
            f":{_MODEL_GROUP}:1d"
        )

        kwargs = _make_kwargs(
            model=_DEPLOYMENT_A,
            model_group=_MODEL_GROUP,
            virtual_key=virtual_key,
            key_budget=_BUDGET,
        )

        with patch.object(
            budget_limiter,
            "_increment_spend_for_key",
            new_callable=AsyncMock,
        ) as mock_increment:
            await budget_limiter.async_log_success_event(
                kwargs, response_obj=None, start_time=None, end_time=None
            )
            actual_spend_key = mock_increment.call_args.kwargs["spend_key"]
            assert actual_spend_key == expected_cache_key

    @pytest.mark.asyncio
    async def test_should_produce_cache_key_readable_by_end_user_lookup(
        self, budget_limiter
    ):
        """The end-user cache key from the write path should exactly match
        what _get_end_user_spend_for_model reads."""
        end_user_id = "user-abc"
        budget_duration = "7d"
        budget = {_MODEL_GROUP: {"budget_limit": 50.0, "time_period": budget_duration}}

        expected_cache_key = (
            f"{END_USER_SPEND_CACHE_KEY_PREFIX}:{end_user_id}"
            f":{_MODEL_GROUP}:{budget_duration}"
        )

        kwargs = _make_kwargs(
            model=_DEPLOYMENT_B,
            model_group=_MODEL_GROUP,
            end_user_id=end_user_id,
            end_user_budget=budget,
        )

        with patch.object(
            budget_limiter,
            "_increment_spend_for_key",
            new_callable=AsyncMock,
        ) as mock_increment:
            await budget_limiter.async_log_success_event(
                kwargs, response_obj=None, start_time=None, end_time=None
            )
            actual_spend_key = mock_increment.call_args.kwargs["spend_key"]
            assert actual_spend_key == expected_cache_key

    @pytest.mark.asyncio
    async def test_should_aggregate_spend_across_deployments(
        self, budget_limiter
    ):
        """Two requests to different deployments (deepseek/deepseek-chat
        and openrouter/deepseek/deepseek-chat) in the same model group
        should write to the same cache key, aggregating spend."""
        virtual_key = "sk-hash-agg"

        # Request 1: deepseek/deepseek-chat, costs $2.00
        kwargs1 = _make_kwargs(
            model=_DEPLOYMENT_A,
            model_group=_MODEL_GROUP,
            response_cost=2.0,
            virtual_key=virtual_key,
            key_budget=_BUDGET,
        )
        await budget_limiter.async_log_success_event(
            kwargs1, response_obj=None, start_time=None, end_time=None
        )

        # Request 2: openrouter/deepseek/deepseek-chat, costs $0.70
        kwargs2 = _make_kwargs(
            model=_DEPLOYMENT_B,
            model_group=_MODEL_GROUP,
            response_cost=0.7,
            virtual_key=virtual_key,
            key_budget=_BUDGET,
        )
        await budget_limiter.async_log_success_event(
            kwargs2, response_obj=None, start_time=None, end_time=None
        )

        # Read back: should see aggregated spend of $2.70
        from litellm.types.utils import BudgetConfig

        spend = await budget_limiter._get_virtual_key_spend_for_model(
            user_api_key_hash=virtual_key,
            model=_MODEL_GROUP,
            key_budget_config=BudgetConfig(
                budget_limit=100.0, time_period="1d"
            ),
        )
        assert spend == pytest.approx(2.7)

    @pytest.mark.asyncio
    async def test_end_to_end_write_then_read_with_model_group(
        self, budget_limiter
    ):
        """Write spend via async_log_success_event with model_group, then
        read it back via is_key_within_model_budget.  The values should
        be visible because the cache keys now align."""
        virtual_key = "sk-hash-e2e"

        user_api_key = UserAPIKeyAuth(
            token=virtual_key,
            key_alias="test-alias",
            model_max_budget=_BUDGET,
        )

        kwargs = _make_kwargs(
            model=_DEPLOYMENT_A,
            model_group=_MODEL_GROUP,
            response_cost=42.0,
            virtual_key=virtual_key,
            key_budget=_BUDGET,
        )
        await budget_limiter.async_log_success_event(
            kwargs, response_obj=None, start_time=None, end_time=None
        )

        # Read: check the budget — should see the 42.0 spend
        result = await budget_limiter.is_key_within_model_budget(
            user_api_key, _MODEL_GROUP
        )
        assert result is True  # 42.0 < 100.0

        # Verify the spend is actually there in cache
        from litellm.types.utils import BudgetConfig

        spend = await budget_limiter._get_virtual_key_spend_for_model(
            user_api_key_hash=virtual_key,
            model=_MODEL_GROUP,
            key_budget_config=BudgetConfig(
                budget_limit=100.0, time_period="1d"
            ),
        )
        assert spend == 42.0
