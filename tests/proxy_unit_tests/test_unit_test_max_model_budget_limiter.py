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


class _SharedRedisCache:
    """Minimal shared Redis stand-in backing multiple DualCache instances."""

    def __init__(self, store):
        self.store = store
        self.get_calls = 0

    async def async_get_cache(self, key, parent_otel_span=None, **kwargs):
        self.get_calls += 1
        return self.store.get(key)


@pytest.mark.asyncio
async def test_admission_uses_shared_redis_spend_across_replicas():
    """
    Regression for #33325: with multiple proxy replicas, each pod holds a
    pod-local in-memory spend below the cap while the shared Redis total is
    already above it. Admission must read the shared Redis value (Redis-first),
    so every pod rejects once the combined spend exceeds the cap instead of
    each admitting off its stale local value.
    """
    from litellm.proxy.hooks.model_max_budget_limiter import (
        VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX,
    )

    token = "test-key"
    model = "gpt-4"
    budget_duration = "1d"
    max_budget = 100.0
    spend_key = f"{VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX}:{token}:{model}:{budget_duration}"

    shared_store = {spend_key: 120.0}

    user_api_key = UserAPIKeyAuth(
        token=token,
        key_alias="test-alias",
        model_max_budget={model: {"budget_limit": max_budget, "time_period": budget_duration}},
    )

    limiters = []
    redis_caches = []
    for pod_local_spend in (60.0, 60.0):
        dual_cache = DualCache()
        redis_cache = _SharedRedisCache(shared_store)
        dual_cache.redis_cache = redis_cache
        # Seed a stale pod-local value that is below the cap on its own.
        await dual_cache.in_memory_cache.async_set_cache(key=spend_key, value=pod_local_spend)
        limiters.append(_PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=dual_cache))
        redis_caches.append(redis_cache)

    for limiter in limiters:
        with pytest.raises(litellm.BudgetExceededError):
            await limiter.is_key_within_model_budget(user_api_key, model)

    # Admission must consult shared Redis, not only the local in-memory value.
    assert all(rc.get_calls > 0 for rc in redis_caches)


@pytest.mark.asyncio
async def test_get_shared_model_spend_falls_back_to_in_memory_without_redis(
    budget_limiter,
):
    """When Redis is not configured, admission reads the local in-memory value."""
    from litellm.proxy.hooks.model_max_budget_limiter import (
        VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX,
    )

    assert budget_limiter.dual_cache.redis_cache is None
    cache_key = f"{VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX}:test-key:gpt-4:1d"
    await budget_limiter.dual_cache.in_memory_cache.async_set_cache(key=cache_key, value=42.0)

    spend = await budget_limiter._get_shared_model_spend(cache_key=cache_key)
    assert spend == 42.0


@pytest.mark.asyncio
async def test_get_shared_model_spend_falls_back_to_local_when_redis_key_missing():
    """
    Redis is attached but has not seen this key yet (e.g. this replica served
    the only requests so far and its flush is still in flight). Admission must
    fall back to the pod-local value instead of treating a Redis miss as zero
    spend, otherwise a single replica stops enforcing its own budget the moment
    Redis is wired in.
    """
    cache_key = "virtual_key_spend:test-key:gpt-4:1d"
    dual_cache = DualCache()
    dual_cache.redis_cache = _SharedRedisCache(store={})  # shared store has no entry
    await dual_cache.in_memory_cache.async_set_cache(key=cache_key, value=7.0)
    limiter = _PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=dual_cache)

    assert await limiter._get_shared_model_spend(cache_key=cache_key) == 7.0
    assert dual_cache.redis_cache.get_calls > 0


@pytest.mark.asyncio
async def test_get_shared_model_spend_returns_max_of_local_and_redis():
    """
    Local (this pod) and Redis (cross-replica total) can disagree. Admission
    must enforce against the larger value so the cap holds whether the local
    increment or another replica's flush is ahead.
    """
    cache_key = "virtual_key_spend:test-key:gpt-4:1d"

    # Redis ahead of local (other replicas already flushed a higher total).
    dual_cache = DualCache()
    dual_cache.redis_cache = _SharedRedisCache(store={cache_key: 90.0})
    await dual_cache.in_memory_cache.async_set_cache(key=cache_key, value=10.0)
    limiter = _PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=dual_cache)
    assert await limiter._get_shared_model_spend(cache_key=cache_key) == 90.0

    # Local ahead of Redis (this pod incremented but has not flushed yet).
    dual_cache = DualCache()
    dual_cache.redis_cache = _SharedRedisCache(store={cache_key: 10.0})
    await dual_cache.in_memory_cache.async_set_cache(key=cache_key, value=90.0)
    limiter = _PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=dual_cache)
    assert await limiter._get_shared_model_spend(cache_key=cache_key) == 90.0


@pytest.mark.asyncio
async def test_get_shared_model_spend_falls_back_to_local_when_redis_raises():
    """A Redis failure (including an open circuit breaker) must degrade to the
    local value rather than failing the admission check."""
    cache_key = "virtual_key_spend:test-key:gpt-4:1d"

    class _RaisingRedisCache:
        async def async_get_cache(self, key, parent_otel_span=None, **kwargs):
            raise ConnectionError("redis down")

    dual_cache = DualCache()
    dual_cache.redis_cache = _RaisingRedisCache()
    await dual_cache.in_memory_cache.async_set_cache(key=cache_key, value=5.0)
    limiter = _PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=dual_cache)

    assert await limiter._get_shared_model_spend(cache_key=cache_key) == 5.0


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
