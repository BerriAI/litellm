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
    dual_cache.redis_cache = (
        object()
    )  # truthy placeholder; push only checks is not None
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
    with patch.object(
        budget_limiter, "_increment_spend_for_key", new_callable=AsyncMock
    ):
        with patch.object(
            budget_limiter,
            "_push_in_memory_increments_to_redis",
            new_callable=AsyncMock,
        ) as mock_push:
            await budget_limiter.async_log_success_event(
                kwargs, response_obj=None, start_time=None, end_time=None
            )
            mock_push.assert_not_awaited()


# === Team-level model_max_budget tests (LIT-2768) ===


@pytest.mark.asyncio
async def test_is_team_within_model_budget_under(budget_limiter):
    """Team budget enforcement returns True when current spend is below budget."""
    with patch.object(
        budget_limiter,
        "_get_team_spend_for_model",
        AsyncMock(return_value=10.0),
    ):
        ok = await budget_limiter.is_team_within_model_budget(
            team_id="team-1",
            team_model_max_budget={
                "gpt-4": {"budget_limit": 100.0, "time_period": "1d"}
            },
            model="gpt-4",
        )
    assert ok is True


@pytest.mark.asyncio
async def test_is_team_within_model_budget_exceeded_raises(budget_limiter):
    """Team budget exceeded raises BudgetExceededError naming the team_id."""
    with patch.object(
        budget_limiter,
        "_get_team_spend_for_model",
        AsyncMock(return_value=150.0),
    ):
        with pytest.raises(litellm.BudgetExceededError) as exc_info:
            await budget_limiter.is_team_within_model_budget(
                team_id="team-finance",
                team_model_max_budget={
                    "gpt-4": {"budget_limit": 100.0, "time_period": "1d"}
                },
                model="gpt-4",
            )
    assert "team-finance" in str(exc_info.value)
    assert "gpt-4" in str(exc_info.value)


@pytest.mark.asyncio
async def test_is_team_within_model_budget_model_not_configured(budget_limiter):
    """Models absent from team_model_max_budget are allowed without spend lookup."""
    with patch.object(
        budget_limiter,
        "_get_team_spend_for_model",
        AsyncMock(return_value=999.0),
    ) as mock_get_spend:
        ok = await budget_limiter.is_team_within_model_budget(
            team_id="team-1",
            team_model_max_budget={
                "gpt-4": {"budget_limit": 100.0, "time_period": "1d"}
            },
            model="claude-3",
        )
    assert ok is True
    mock_get_spend.assert_not_called()


@pytest.mark.asyncio
async def test_team_spend_cache_key_uses_team_prefix(budget_limiter):
    """Team spend cache key must include team_id prefix and time period, not the key hash."""
    from litellm.proxy.hooks.model_max_budget_limiter import (
        TEAM_MODEL_SPEND_CACHE_KEY_PREFIX,
    )
    from litellm.types.utils import BudgetConfig

    budget_cfg = BudgetConfig(budget_limit=10.0, budget_duration="1d")
    captured = {}

    async def fake_get(key):
        captured["key"] = key
        return 0.0

    with patch.object(
        budget_limiter.dual_cache, "async_get_cache", side_effect=fake_get
    ):
        await budget_limiter._get_team_spend_for_model(
            team_id="team-xyz",
            model="gpt-4",
            key_budget_config=budget_cfg,
        )

    assert captured["key"].startswith(
        TEAM_MODEL_SPEND_CACHE_KEY_PREFIX + ":team-xyz:gpt-4:"
    )


@pytest.mark.asyncio
async def test_async_log_success_event_increments_team_spend(budget_limiter):
    """Spend tracking writes to the team-level cache key when team_model_max_budget is present."""
    from litellm.proxy.hooks.model_max_budget_limiter import (
        TEAM_MODEL_SPEND_CACHE_KEY_PREFIX,
    )

    incremented = {}

    async def fake_increment(budget_config, spend_key, start_time_key, response_cost):
        incremented[spend_key] = response_cost

    kwargs = {
        "standard_logging_object": {
            "response_cost": 7.5,
            "model_group": "gpt-4",
            "model": "openai/gpt-4",
            "metadata": {"user_api_key_hash": "hash-1"},
        },
        "litellm_params": {
            "metadata": {
                "user_api_key_team_id": "team-tracking",
                "user_api_key_team_model_max_budget": {
                    "gpt-4": {"budget_limit": 100.0, "time_period": "1d"}
                },
            }
        },
    }

    with patch.object(
        budget_limiter, "_increment_spend_for_key", side_effect=fake_increment
    ):
        await budget_limiter.async_log_success_event(kwargs, None, 0, 0)

    matching_keys = [
        k
        for k in incremented
        if k.startswith(TEAM_MODEL_SPEND_CACHE_KEY_PREFIX + ":team-tracking:")
    ]
    assert (
        matching_keys
    ), f"Expected team-tracked spend key, got {list(incremented.keys())}"
    assert incremented[matching_keys[0]] == 7.5


@pytest.mark.asyncio
async def test_async_log_success_event_skips_team_when_team_id_missing(budget_limiter):
    """No team_id → team-level spend tracking is a no-op even if team_model_max_budget is set."""
    from litellm.proxy.hooks.model_max_budget_limiter import (
        TEAM_MODEL_SPEND_CACHE_KEY_PREFIX,
    )

    incremented = {}

    async def fake_increment(budget_config, spend_key, start_time_key, response_cost):
        incremented[spend_key] = response_cost

    kwargs = {
        "standard_logging_object": {
            "response_cost": 3.0,
            "model_group": "gpt-4",
            "model": "openai/gpt-4",
            "metadata": {"user_api_key_hash": "hash-1"},
        },
        "litellm_params": {
            "metadata": {
                # team_id intentionally absent
                "user_api_key_team_model_max_budget": {
                    "gpt-4": {"budget_limit": 100.0, "time_period": "1d"}
                },
            }
        },
    }

    with patch.object(
        budget_limiter, "_increment_spend_for_key", side_effect=fake_increment
    ):
        await budget_limiter.async_log_success_event(kwargs, None, 0, 0)

    assert not any(k.startswith(TEAM_MODEL_SPEND_CACHE_KEY_PREFIX) for k in incremented)


# === Key precedence tests (LIT-2768 — addresses Greptile review) ===


@pytest.mark.asyncio
async def test_team_check_skipped_when_key_covers_model(budget_limiter):
    """When the requesting key has its own model_max_budget entry for the
    model, the team check must short-circuit before touching the team
    counter — otherwise a sibling key on the same team can be denied
    because of an unrelated key's spend."""
    with patch.object(
        budget_limiter,
        "_get_team_spend_for_model",
        AsyncMock(return_value=10_000.0),  # team well over budget
    ) as mock_team_spend:
        ok = await budget_limiter.is_team_within_model_budget(
            team_id="team-1",
            team_model_max_budget={
                "gpt-4": {"budget_limit": 100.0, "time_period": "1d"}
            },
            model="gpt-4",
            key_model_max_budget={"gpt-4": {"budget_limit": 50.0, "time_period": "1d"}},
        )
    assert ok is True
    mock_team_spend.assert_not_called()


@pytest.mark.asyncio
async def test_team_check_runs_when_key_has_different_model(budget_limiter):
    """If the key has its own budget for a DIFFERENT model, the team
    check still applies to the model being requested."""
    with patch.object(
        budget_limiter,
        "_get_team_spend_for_model",
        AsyncMock(return_value=10.0),  # team well under budget
    ) as mock_team_spend:
        ok = await budget_limiter.is_team_within_model_budget(
            team_id="team-1",
            team_model_max_budget={
                "gpt-4": {"budget_limit": 100.0, "time_period": "1d"}
            },
            model="gpt-4",
            key_model_max_budget={
                "claude-3": {"budget_limit": 50.0, "time_period": "1d"}
            },
        )
    assert ok is True
    mock_team_spend.assert_called_once()


@pytest.mark.asyncio
async def test_team_check_skipped_for_provider_prefixed_match(budget_limiter):
    """Key precedence holds when the key's entry matches via the
    `_get_model_without_custom_llm_provider` normalization, mirroring
    is_key_within_model_budget."""
    with patch.object(
        budget_limiter,
        "_get_team_spend_for_model",
        AsyncMock(return_value=10_000.0),
    ) as mock_team_spend:
        ok = await budget_limiter.is_team_within_model_budget(
            team_id="team-1",
            team_model_max_budget={
                "gpt-4": {"budget_limit": 100.0, "time_period": "1d"}
            },
            model="openai/gpt-4",
            key_model_max_budget={"gpt-4": {"budget_limit": 50.0, "time_period": "1d"}},
        )
    assert ok is True
    mock_team_spend.assert_not_called()


@pytest.mark.asyncio
async def test_async_log_success_event_skips_team_increment_when_key_covers_model(
    budget_limiter,
):
    """Team counter must NOT be incremented for models where the key has
    its own model_max_budget entry — otherwise the team counter is driven
    by keys with private caps, blocking sibling keys."""
    from litellm.proxy.hooks.model_max_budget_limiter import (
        TEAM_MODEL_SPEND_CACHE_KEY_PREFIX,
        VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX,
    )

    incremented = {}

    async def fake_increment(budget_config, spend_key, start_time_key, response_cost):
        incremented[spend_key] = response_cost

    kwargs = {
        "standard_logging_object": {
            "response_cost": 5.0,
            "model_group": "gpt-4",
            "model": "openai/gpt-4",
            "metadata": {"user_api_key_hash": "hash-1"},
        },
        "litellm_params": {
            "metadata": {
                "user_api_key_team_id": "team-1",
                "user_api_key_team_model_max_budget": {
                    "gpt-4": {"budget_limit": 100.0, "time_period": "1d"}
                },
                "user_api_key_model_max_budget": {
                    "gpt-4": {"budget_limit": 50.0, "time_period": "1d"}
                },
            }
        },
    }
    with patch.object(
        budget_limiter, "_increment_spend_for_key", side_effect=fake_increment
    ):
        await budget_limiter.async_log_success_event(kwargs, None, 0, 0)

    team_keys = [
        k for k in incremented if k.startswith(TEAM_MODEL_SPEND_CACHE_KEY_PREFIX)
    ]
    key_keys = [
        k for k in incremented if k.startswith(VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX)
    ]
    assert not team_keys, f"Unexpected team-spend writes: {team_keys}"
    assert key_keys, "Key-spend should still be tracked"
