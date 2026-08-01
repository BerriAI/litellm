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


@pytest.mark.asyncio
async def test_async_log_success_event_pushes_redis_increments_when_redis_configured():
    """
    Virtual-key model max budget limiter does not run RouterBudgetLimiting.__init__,
    so the periodic Redis flush task never starts. After logging spend we must call
    _push_in_memory_increments_to_redis when Redis is wired so other workers see spend.
    """
    dual_cache = DualCache()
    dual_cache.redis_cache = object()  # truthy placeholder; push only checks is not None
    limiter = _PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=dual_cache)
    model = "gpt-4"
    kwargs = {
        "standard_logging_object": {
            "response_cost": 0.01,
            "model": model,
            "metadata": {"user_api_key_hash": "vk-hash"},
        },
        "litellm_params": {
            "metadata": {
                "user_api_key_model_max_budget": {
                    model: {"budget_limit": 10.0, "time_period": "1d"},
                },
            },
        },
    }
    with patch.object(limiter, "_increment_spend_for_key", new_callable=AsyncMock):
        with patch.object(
            limiter,
            "_push_in_memory_increments_to_redis",
            new_callable=AsyncMock,
        ) as mock_push:
            await limiter.async_log_success_event(
                kwargs, response_obj=None, start_time=None, end_time=None
            )
            mock_push.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_fallback_model_within_budget_returns_none_without_fallbacks(
    budget_limiter,
):
    user_api_key = UserAPIKeyAuth(token="test-key", budget_fallbacks={})
    assert (
        await budget_limiter.get_fallback_model_within_budget(user_api_key, "gpt-4")
        is None
    )


@pytest.mark.asyncio
async def test_get_fallback_model_within_budget_returns_first_within_budget(
    budget_limiter,
):
    user_api_key = UserAPIKeyAuth(
        token="test-key",
        model_max_budget={"gpt-4o-mini": {"budget_limit": 100.0, "time_period": "1d"}},
        budget_fallbacks={"gpt-4": ["gpt-4o-mini", "claude-haiku"]},
    )
    with patch.object(
        budget_limiter, "_get_virtual_key_spend_for_model", return_value=1.0
    ):
        result = await budget_limiter.get_fallback_model_within_budget(
            user_api_key, "gpt-4"
        )
    assert result == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_get_fallback_model_within_budget_skips_exhausted_fallback(
    budget_limiter,
):
    user_api_key = UserAPIKeyAuth(
        token="test-key",
        model_max_budget={
            "gpt-4o-mini": {"budget_limit": 100.0, "time_period": "1d"},
            "claude-haiku": {"budget_limit": 100.0, "time_period": "1d"},
        },
        budget_fallbacks={"gpt-4": ["gpt-4o-mini", "claude-haiku"]},
    )

    async def _spend_for_model(user_api_key_hash, model, key_budget_config):
        return 150.0 if model == "gpt-4o-mini" else 1.0

    with patch.object(
        budget_limiter,
        "_get_virtual_key_spend_for_model",
        side_effect=_spend_for_model,
    ):
        result = await budget_limiter.get_fallback_model_within_budget(
            user_api_key, "gpt-4"
        )
    assert result == "claude-haiku"


@pytest.mark.asyncio
async def test_get_fallback_model_within_budget_returns_none_when_chain_exhausted(
    budget_limiter,
):
    user_api_key = UserAPIKeyAuth(
        token="test-key",
        model_max_budget={
            "gpt-4o-mini": {"budget_limit": 100.0, "time_period": "1d"},
            "claude-haiku": {"budget_limit": 100.0, "time_period": "1d"},
        },
        budget_fallbacks={"gpt-4": ["gpt-4o-mini", "claude-haiku"]},
    )
    with patch.object(
        budget_limiter, "_get_virtual_key_spend_for_model", return_value=150.0
    ):
        result = await budget_limiter.get_fallback_model_within_budget(
            user_api_key, "gpt-4"
        )
    assert result is None


@pytest.mark.asyncio
async def test_async_log_success_event_skips_redis_push_without_redis(budget_limiter):
    """When dual_cache has no Redis backend, do not await _push_in_memory_increments_to_redis."""
    assert budget_limiter.dual_cache.redis_cache is None
    model = "gpt-4"
    kwargs = {
        "standard_logging_object": {
            "response_cost": 0.01,
            "model": model,
            "metadata": {"user_api_key_hash": "vk-hash"},
        },
        "litellm_params": {
            "metadata": {
                "user_api_key_model_max_budget": {
                    model: {"budget_limit": 10.0, "time_period": "1d"},
                },
            },
        },
    }
    with patch.object(budget_limiter, "_increment_spend_for_key", new_callable=AsyncMock):
        with patch.object(
            budget_limiter,
            "_push_in_memory_increments_to_redis",
            new_callable=AsyncMock,
        ) as mock_push:
            await budget_limiter.async_log_success_event(
                kwargs, response_obj=None, start_time=None, end_time=None
            )
            mock_push.assert_not_awaited()


# ---------------------------------------------------------------------------
# Model-group (shared) budgets
# ---------------------------------------------------------------------------


def test_get_request_model_budget_key_and_config_group_match(budget_limiter):
    """
    A budget entry that carries a `models` list defines a model-group budget.
    Any request model in that list must resolve to the group name (the dict key),
    so every model in the group shares one spend counter.
    """
    internal_budget = {
        "opus-family": GenericBudgetInfo(
            budget_limit=50.0,
            time_period="30d",
            models=["claude-opus-4", "claude-opus-4-1"],
        ),
        "gpt-4": GenericBudgetInfo(budget_limit=100.0, time_period="1d"),
    }

    # both group members resolve to the SAME budget key (the group name)
    key_a, config_a = budget_limiter._get_request_model_budget_key_and_config(
        model="claude-opus-4", internal_model_max_budget=internal_budget
    )
    key_b, config_b = budget_limiter._get_request_model_budget_key_and_config(
        model="claude-opus-4-1", internal_model_max_budget=internal_budget
    )
    assert key_a == key_b == "opus-family"
    assert config_a.max_budget == config_b.max_budget == 50.0

    # provider-prefixed group member still resolves to the group
    key_c, _ = budget_limiter._get_request_model_budget_key_and_config(
        model="anthropic/claude-opus-4", internal_model_max_budget=internal_budget
    )
    assert key_c == "opus-family"

    # a plain per-model entry resolves to the model name, not a group
    key_d, config_d = budget_limiter._get_request_model_budget_key_and_config(
        model="gpt-4", internal_model_max_budget=internal_budget
    )
    assert key_d == "gpt-4"
    assert config_d.max_budget == 100.0

    # a model in no group and no per-model entry resolves to nothing
    assert (
        budget_limiter._get_request_model_budget_key_and_config(
            model="gemini-2.5-pro", internal_model_max_budget=internal_budget
        )
        is None
    )


@pytest.mark.asyncio
async def test_is_key_within_model_budget_group_reads_group_counter(budget_limiter):
    """
    Enforcement for any group member must read spend from the group counter
    (budget_key == group name), so combined spend across the group is enforced.
    """
    user_api_key = UserAPIKeyAuth(
        token="test-key",
        key_alias="test-alias",
        model_max_budget={
            "opus-family": {
                "budget_limit": 50.0,
                "time_period": "30d",
                "models": ["claude-opus-4", "claude-opus-4-1"],
            }
        },
    )

    seen_models = []

    async def _spend(user_api_key_hash, model, key_budget_config):
        seen_models.append(model)
        return 60.0

    with patch.object(
        budget_limiter, "_get_virtual_key_spend_for_model", side_effect=_spend
    ):
        # spend already over the shared 50.0 budget -> every member is blocked
        for member in ("claude-opus-4", "claude-opus-4-1", "anthropic/claude-opus-4-1"):
            with pytest.raises(litellm.BudgetExceededError):
                await budget_limiter.is_key_within_model_budget(user_api_key, member)

    # all members were looked up under the single group counter
    assert seen_models == ["opus-family", "opus-family", "opus-family"]


@pytest.mark.asyncio
async def test_async_log_success_event_group_members_share_one_counter(budget_limiter):
    """
    Core regression for model-group budgets: spend from two DIFFERENT models in a
    group must increment the SAME cache key (the group name) so the budget is
    combined. Before this feature each model incremented its own key, letting a
    developer exceed the intended total by spreading usage across the family.
    """
    from litellm.proxy.hooks.model_max_budget_limiter import (
        VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX,
    )

    virtual_key = "test-key-hash"
    budget_duration = "30d"
    user_api_key_model_max_budget = {
        "opus-family": {
            "budget_limit": 50.0,
            "time_period": budget_duration,
            "models": ["claude-opus-4", "claude-opus-4-1"],
        },
    }

    def _kwargs_for(model_group):
        return {
            "standard_logging_object": {
                "response_cost": 0.10,
                "model": model_group,
                "model_group": model_group,
                "metadata": {"user_api_key_hash": virtual_key},
            },
            "litellm_params": {
                "metadata": {
                    "user_api_key_model_max_budget": user_api_key_model_max_budget,
                },
            },
        }

    expected_key = (
        f"{VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX}:{virtual_key}:opus-family:{budget_duration}"
    )

    with patch.object(
        budget_limiter, "_increment_spend_for_key", new_callable=AsyncMock
    ) as mock_increment:
        await budget_limiter.async_log_success_event(
            _kwargs_for("claude-opus-4"), None, None, None
        )
        await budget_limiter.async_log_success_event(
            _kwargs_for("claude-opus-4-1"), None, None, None
        )

    spend_keys = [c.kwargs["spend_key"] for c in mock_increment.call_args_list]
    assert spend_keys == [expected_key, expected_key]
