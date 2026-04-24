import os
import sys
from unittest.mock import AsyncMock, patch

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path

import pytest

import litellm
from litellm.caching.caching import DualCache
from litellm.proxy.hooks.model_max_budget_limiter import (
    _PROXY_VirtualKeyModelMaxBudgetLimiter,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import BudgetConfig as GenericBudgetInfo


# Test class setup
@pytest.fixture
def budget_limiter():
    dual_cache = DualCache()
    return _PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=dual_cache)


# Test _get_model_without_custom_llm_provider
def test_get_model_without_custom_llm_provider(budget_limiter):
    # Test with custom provider
    assert (
        budget_limiter._get_model_without_custom_llm_provider("openai/gpt-4") == "gpt-4"
    )

    # Test without custom provider
    assert budget_limiter._get_model_without_custom_llm_provider("gpt-4") == "gpt-4"


# Test _get_request_model_budget_config
def test_get_request_model_budget_config(budget_limiter):
    internal_budget = {
        "gpt-4": GenericBudgetInfo(budget_limit=100.0, time_period="1d"),
        "claude-3": GenericBudgetInfo(budget_limit=50.0, time_period="1d"),
    }

    # Test direct model match
    config = budget_limiter._get_request_model_budget_config(
        model="gpt-4", internal_model_max_budget=internal_budget
    )
    assert config.max_budget == 100.0

    # Test model with provider
    config = budget_limiter._get_request_model_budget_config(
        model="openai/gpt-4", internal_model_max_budget=internal_budget
    )
    assert config.max_budget == 100.0

    # Test non-existent model
    config = budget_limiter._get_request_model_budget_config(
        model="non-existent", internal_model_max_budget=internal_budget
    )
    assert config is None


# Test is_key_within_model_budget
@pytest.mark.asyncio
async def test_is_key_within_model_budget(budget_limiter):
    # Mock user API key dict
    user_api_key = UserAPIKeyAuth(
        token="test-key",
        key_alias="test-alias",
        model_max_budget={"gpt-4": {"budget_limit": 100.0, "time_period": "1d"}},
    )

    # Test when model is within budget
    with patch.object(
        budget_limiter, "_get_virtual_key_spend_for_model", return_value=50.0
    ):
        assert (
            await budget_limiter.is_key_within_model_budget(user_api_key, "gpt-4")
            is True
        )

    # Test when model exceeds budget
    with patch.object(
        budget_limiter, "_get_virtual_key_spend_for_model", return_value=150.0
    ):
        with pytest.raises(litellm.BudgetExceededError):
            await budget_limiter.is_key_within_model_budget(user_api_key, "gpt-4")

    # Test model not in budget config
    assert (
        await budget_limiter.is_key_within_model_budget(user_api_key, "non-existent")
        is True
    )


# Test _get_virtual_key_spend_for_model
@pytest.mark.asyncio
async def test_get_virtual_key_spend_for_model(budget_limiter):
    budget_config = GenericBudgetInfo(budget_limit=100.0, time_period="1d")

    # Mock cache get
    with patch.object(budget_limiter.dual_cache, "async_get_cache", return_value=50.0):
        spend = await budget_limiter._get_virtual_key_spend_for_model(
            user_api_key_hash="test-key", model="gpt-4", key_budget_config=budget_config
        )
        assert spend == 50.0

        # Test with provider prefix
        spend = await budget_limiter._get_virtual_key_spend_for_model(
            user_api_key_hash="test-key",
            model="openai/gpt-4",
            key_budget_config=budget_config,
        )
        assert spend == 50.0


@pytest.mark.asyncio
async def test_async_log_success_event_uses_per_model_budget_duration(budget_limiter):
    """
    async_log_success_event must use the per-model budget_duration for the cache key
    so spend is tracked per model correctly. Regression test for per-model budget implementation.
    """
    from litellm.proxy.hooks.model_max_budget_limiter import (
        VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX,
    )

    virtual_key = "test-key-hash"
    model = "gpt-4"
    budget_duration = "1d"
    user_api_key_model_max_budget = {
        model: {"budget_limit": 100.0, "time_period": budget_duration},
    }
    kwargs = {
        "standard_logging_object": {
            "response_cost": 0.05,
            "model": model,
            "metadata": {"user_api_key_hash": virtual_key},
        },
        "litellm_params": {
            "metadata": {
                "user_api_key_model_max_budget": user_api_key_model_max_budget
            },
        },
    }
    with patch.object(
        budget_limiter,
        "_increment_spend_for_key",
        new_callable=AsyncMock,
    ) as mock_increment:
        await budget_limiter.async_log_success_event(
            kwargs, response_obj=None, start_time=None, end_time=None
        )
        mock_increment.assert_awaited_once()
        call_kwargs = mock_increment.call_args.kwargs
        spend_key = call_kwargs["spend_key"]
        assert spend_key == (
            f"{VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX}:{virtual_key}:{model}:{budget_duration}"
        )
        assert call_kwargs["response_cost"] == 0.05


# Test is_end_user_within_model_budget
@pytest.mark.asyncio
async def test_is_end_user_within_model_budget(budget_limiter):
    # Test when model is within budget
    with patch.object(
        budget_limiter, "_get_end_user_spend_for_model", return_value=50.0
    ):
        assert (
            await budget_limiter.is_end_user_within_model_budget(
                "test-user",
                {"gpt-4": {"budget_limit": 100.0, "time_period": "1d"}},
                "gpt-4",
            )
            is True
        )

    # Test when model exceeds budget
    with patch.object(
        budget_limiter, "_get_end_user_spend_for_model", return_value=150.0
    ):
        with pytest.raises(litellm.BudgetExceededError):
            await budget_limiter.is_end_user_within_model_budget(
                "test-user",
                {"gpt-4": {"budget_limit": 100.0, "time_period": "1d"}},
                "gpt-4",
            )

    # Test model not in budget config
    assert (
        await budget_limiter.is_end_user_within_model_budget(
            "test-user",
            {"gpt-4": {"budget_limit": 100.0, "time_period": "1d"}},
            "non-existent",
        )
        is True
    )


# Test _get_end_user_spend_for_model
@pytest.mark.asyncio
async def test_get_end_user_spend_for_model(budget_limiter):
    budget_config = GenericBudgetInfo(budget_limit=100.0, time_period="1d")

    # Mock cache get
    with patch.object(budget_limiter.dual_cache, "async_get_cache", return_value=50.0):
        spend = await budget_limiter._get_end_user_spend_for_model(
            end_user_id="test-user", model="gpt-4", key_budget_config=budget_config
        )
        assert spend == 50.0

        # Test with provider prefix
        spend = await budget_limiter._get_end_user_spend_for_model(
            end_user_id="test-user",
            model="openai/gpt-4",
            key_budget_config=budget_config,
        )
        assert spend == 50.0


@pytest.mark.asyncio
async def test_async_log_success_event_uses_model_group_for_cache_key(budget_limiter):
    """
    When model_group is present in StandardLoggingPayload (proxy/router
    deployments), spend must be tracked under the model_group name — not the
    deployment-level model name — so the cache key matches the one used by
    is_key_within_model_budget (which receives request_data["model"], the
    model group alias).

    Without this, providers that decorate model names (e.g. Vertex AI
    "vertex_ai/claude-opus-4-6@default") track spend under a different cache
    key than enforcement reads, silently disabling budget limits.
    """
    from litellm.proxy.hooks.model_max_budget_limiter import (
        VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX,
    )

    virtual_key = "test-key-hash"
    model_group = "claude-opus-4-6"
    deployment_model = "vertex_ai/claude-opus-4-6@default"
    budget_duration = "1d"
    user_api_key_model_max_budget = {
        model_group: {"budget_limit": 50.0, "time_period": budget_duration},
    }
    kwargs = {
        "standard_logging_object": {
            "response_cost": 0.10,
            "model": deployment_model,
            "model_group": model_group,
            "metadata": {"user_api_key_hash": virtual_key},
        },
        "litellm_params": {
            "metadata": {
                "user_api_key_model_max_budget": user_api_key_model_max_budget,
            },
        },
    }
    with patch.object(
        budget_limiter,
        "_increment_spend_for_key",
        new_callable=AsyncMock,
    ) as mock_increment:
        await budget_limiter.async_log_success_event(
            kwargs, response_obj=None, start_time=None, end_time=None
        )
        mock_increment.assert_awaited_once()
        call_kwargs = mock_increment.call_args.kwargs
        spend_key = call_kwargs["spend_key"]
        # The cache key must use the model_group name, NOT the deployment name
        assert spend_key == (
            f"{VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX}:{virtual_key}:{model_group}:{budget_duration}"
        )
        assert call_kwargs["response_cost"] == 0.10


@pytest.mark.asyncio
async def test_async_log_success_event_falls_back_to_model_when_no_model_group(
    budget_limiter,
):
    """
    When model_group is None (non-proxy / non-router usage), spend tracking
    must fall back to using the model field so existing behaviour is preserved.
    """
    from litellm.proxy.hooks.model_max_budget_limiter import (
        VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX,
    )

    virtual_key = "test-key-hash"
    model = "gpt-4"
    budget_duration = "1d"
    user_api_key_model_max_budget = {
        model: {"budget_limit": 100.0, "time_period": budget_duration},
    }
    kwargs = {
        "standard_logging_object": {
            "response_cost": 0.05,
            "model": model,
            "model_group": None,
            "metadata": {"user_api_key_hash": virtual_key},
        },
        "litellm_params": {
            "metadata": {
                "user_api_key_model_max_budget": user_api_key_model_max_budget,
            },
        },
    }
    with patch.object(
        budget_limiter,
        "_increment_spend_for_key",
        new_callable=AsyncMock,
    ) as mock_increment:
        await budget_limiter.async_log_success_event(
            kwargs, response_obj=None, start_time=None, end_time=None
        )
        mock_increment.assert_awaited_once()
        call_kwargs = mock_increment.call_args.kwargs
        spend_key = call_kwargs["spend_key"]
        assert spend_key == (
            f"{VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX}:{virtual_key}:{model}:{budget_duration}"
        )


@pytest.mark.asyncio
async def test_async_log_success_event_end_user_uses_model_group(budget_limiter):
    """
    End-user model budget tracking must also use model_group when available,
    matching the enforcement path in is_end_user_within_model_budget.
    """
    from litellm.proxy.hooks.model_max_budget_limiter import (
        END_USER_SPEND_CACHE_KEY_PREFIX,
    )

    end_user_id = "test-user"
    model_group = "claude-sonnet-4-6"
    deployment_model = "vertex_ai/claude-sonnet-4-6@default"
    budget_duration = "1d"
    user_api_key_end_user_model_max_budget = {
        model_group: {"budget_limit": 25.0, "time_period": budget_duration},
    }
    kwargs = {
        "standard_logging_object": {
            "response_cost": 0.03,
            "model": deployment_model,
            "model_group": model_group,
            "end_user": end_user_id,
            "metadata": {"user_api_key_end_user_id": end_user_id},
        },
        "litellm_params": {
            "metadata": {
                "user_api_key_end_user_model_max_budget": user_api_key_end_user_model_max_budget,
            },
        },
    }
    with patch.object(
        budget_limiter,
        "_increment_spend_for_key",
        new_callable=AsyncMock,
    ) as mock_increment:
        await budget_limiter.async_log_success_event(
            kwargs, response_obj=None, start_time=None, end_time=None
        )
        mock_increment.assert_awaited_once()
        call_kwargs = mock_increment.call_args.kwargs
        spend_key = call_kwargs["spend_key"]
        assert spend_key == (
            f"{END_USER_SPEND_CACHE_KEY_PREFIX}:{end_user_id}:{model_group}:{budget_duration}"
        )


@pytest.mark.asyncio
async def test_async_log_success_event_uses_end_user_model_budget_duration(
    budget_limiter,
):
    """
    async_log_success_event must use the per-model budget_duration for the end user cache key
    """
    from litellm.proxy.hooks.model_max_budget_limiter import (
        END_USER_SPEND_CACHE_KEY_PREFIX,
    )

    end_user_id = "test-user"
    model = "gpt-4"
    budget_duration = "1d"
    user_api_key_end_user_model_max_budget = {
        model: {"budget_limit": 100.0, "time_period": budget_duration},
    }
    kwargs = {
        "standard_logging_object": {
            "response_cost": 0.05,
            "model": model,
            "end_user": end_user_id,
            "metadata": {"user_api_key_end_user_id": end_user_id},
        },
        "litellm_params": {
            "metadata": {
                "user_api_key_end_user_model_max_budget": user_api_key_end_user_model_max_budget
            },
        },
    }
    with patch.object(
        budget_limiter,
        "_increment_spend_for_key",
        new_callable=AsyncMock,
    ) as mock_increment:
        await budget_limiter.async_log_success_event(
            kwargs, response_obj=None, start_time=None, end_time=None
        )
        mock_increment.assert_awaited_once()
        call_kwargs = mock_increment.call_args.kwargs
        spend_key = call_kwargs["spend_key"]
        assert spend_key == (
            f"{END_USER_SPEND_CACHE_KEY_PREFIX}:{end_user_id}:{model}:{budget_duration}"
        )
        assert call_kwargs["response_cost"] == 0.05


# ---------------------------------------------------------------------------
# DB fallback tests (Bug 2 — cold cache = no enforcement)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_virtual_key_spend_db_fallback_enforces_budget(budget_limiter):
    """
    When both cache lookups return None (cold cache), _get_virtual_key_spend_for_model
    must fall back to _get_spend_from_db and return that value so the caller can
    raise BudgetExceededError rather than silently passing the request through.
    """
    budget_config = GenericBudgetInfo(budget_limit=100.0, time_period="1d")

    # Both cache tiers cold
    with patch.object(
        budget_limiter.dual_cache, "async_get_cache", return_value=None
    ):
        with patch.object(
            budget_limiter,
            "_get_spend_from_db",
            new_callable=AsyncMock,
            return_value=120.0,  # over budget
        ) as mock_db:
            spend = await budget_limiter._get_virtual_key_spend_for_model(
                user_api_key_hash="hash-abc",
                model="gpt-4o",
                key_budget_config=budget_config,
            )

    assert spend == 120.0
    mock_db.assert_awaited_once()
    call_kwargs = mock_db.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4o"
    assert call_kwargs["budget_duration"] == "1d"
    assert "hash-abc" in call_kwargs["budget_start_time_key"]
    assert call_kwargs["entity_filter"] == {"api_key": "hash-abc"}


@pytest.mark.asyncio
async def test_get_end_user_spend_db_fallback_enforces_budget(budget_limiter):
    """
    When both cache lookups return None, _get_end_user_spend_for_model must
    fall back to _get_spend_from_db and return the DB value.
    """
    budget_config = GenericBudgetInfo(budget_limit=50.0, time_period="7d")

    with patch.object(
        budget_limiter.dual_cache, "async_get_cache", return_value=None
    ):
        with patch.object(
            budget_limiter,
            "_get_spend_from_db",
            new_callable=AsyncMock,
            return_value=55.0,
        ) as mock_db:
            spend = await budget_limiter._get_end_user_spend_for_model(
                end_user_id="user-xyz",
                model="claude-3",
                key_budget_config=budget_config,
            )

    assert spend == 55.0
    mock_db.assert_awaited_once()
    call_kwargs = mock_db.call_args.kwargs
    assert call_kwargs["entity_filter"] == {"end_user": "user-xyz"}
    assert "user-xyz" in call_kwargs["budget_start_time_key"]


@pytest.mark.asyncio
async def test_get_virtual_key_spend_cache_hit_skips_db(budget_limiter):
    """
    When cache returns a value, _get_spend_from_db must NOT be called.
    The DB fallback is only for cold-cache scenarios.
    """
    budget_config = GenericBudgetInfo(budget_limit=100.0, time_period="1d")

    with patch.object(
        budget_limiter.dual_cache, "async_get_cache", return_value=30.0
    ):
        with patch.object(
            budget_limiter, "_get_spend_from_db", new_callable=AsyncMock
        ) as mock_db:
            spend = await budget_limiter._get_virtual_key_spend_for_model(
                user_api_key_hash="hash-abc",
                model="gpt-4o",
                key_budget_config=budget_config,
            )

    assert spend == 30.0
    mock_db.assert_not_awaited()


@pytest.mark.asyncio
async def test_db_fallback_seeds_cache(budget_limiter):
    """
    After a successful DB fallback, the result must be written to cache so
    subsequent requests don't hit the DB on every call.
    """
    budget_config = GenericBudgetInfo(budget_limit=100.0, time_period="1d")

    set_cache_calls = []

    async def fake_get_cache(key):
        return None  # always cold

    async def fake_set_cache(key, value, ttl=None):
        set_cache_calls.append((key, value, ttl))

    with patch.object(
        budget_limiter.dual_cache, "async_get_cache", side_effect=fake_get_cache
    ):
        with patch.object(
            budget_limiter.dual_cache, "async_set_cache", side_effect=fake_set_cache
        ):
            with patch.object(
                budget_limiter,
                "_get_spend_from_db",
                new_callable=AsyncMock,
                return_value=75.0,
            ):
                await budget_limiter._get_virtual_key_spend_for_model(
                    user_api_key_hash="hash-abc",
                    model="gpt-4o",
                    key_budget_config=budget_config,
                )

    assert len(set_cache_calls) == 1, "Expected exactly one cache write to seed the result"
    cached_key, cached_value, cached_ttl = set_cache_calls[0]
    assert cached_value == 75.0
    assert "gpt-4o" in cached_key
    assert "1d" in cached_key
