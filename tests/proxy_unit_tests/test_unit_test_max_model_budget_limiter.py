from unittest.mock import AsyncMock, patch


import pytest

import litellm
from litellm.caching.caching import DualCache
from datetime import datetime, timezone

from litellm.litellm_core_utils.duration_parser import duration_in_seconds
from litellm.proxy._types import Litellm_EntityType
from litellm.proxy.hooks.model_max_budget_limiter import (
    _budget_model_candidates,
    _PROXY_VirtualKeyModelMaxBudgetLimiter,
    build_model_max_budget_usage,
    resolve_model_budget,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import BudgetConfig as GenericBudgetInfo


# Test class setup
@pytest.fixture
def budget_limiter():
    dual_cache = DualCache()
    return _PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=dual_cache)


# Test _budget_model_candidates
def test_budget_model_candidates():
    # Test with custom provider
    assert _budget_model_candidates("openai/gpt-4") == ("openai/gpt-4", "gpt-4")

    # Test without custom provider: no duplicate candidate
    assert _budget_model_candidates("gpt-4") == ("gpt-4",)


@pytest.mark.parametrize(
    "model,expected",
    [
        (
            "bedrock/anthropic.claude-opus-4-8",
            (
                "bedrock/anthropic.claude-opus-4-8",
                "anthropic.claude-opus-4-8",
                "claude-opus-4-8",
            ),
        ),
        (
            "us.anthropic.claude-opus-4-8",
            (
                "us.anthropic.claude-opus-4-8",
                "anthropic.claude-opus-4-8",
                "claude-opus-4-8",
            ),
        ),
        (
            "bedrock/converse/us.amazon.nova-pro-v1:0",
            (
                "bedrock/converse/us.amazon.nova-pro-v1:0",
                "us.amazon.nova-pro-v1:0",
                "amazon.nova-pro-v1:0",
                "nova-pro-v1:0",
            ),
        ),
    ],
)
def test_budget_model_candidates_reach_the_bedrock_family_name(model, expected):
    """
    Bedrock ids carry a dotted vendor segment ("anthropic.", "amazon.") on top of
    the optional cross-region prefix, so a budget configured under the bare
    family name would otherwise never match Bedrock traffic: no enforcement and
    no spend tracking at all.
    """
    assert _budget_model_candidates(model) == expected


@pytest.mark.parametrize(
    "model",
    [
        "azure/gpt-4.1",
        "gpt-image-1.5",
        "not-a-real-model.with.dots",
        "ft:gpt-4o:acme::abc",
    ],
)
def test_budget_model_candidates_never_split_a_non_bedrock_dotted_name(model):
    """
    Most dotted model ids are versions, not Bedrock vendor prefixes. Splitting one
    would offer a garbage candidate ("gpt-4.1" -> "1") that could collide with an
    unrelated budget entry, so the split is gated on litellm pricing the model as
    a Bedrock model.
    """
    for candidate in _budget_model_candidates(model):
        assert candidate in (model, model.split("/")[-1])


# Test resolve_model_budget
def test_resolve_model_budget():
    model_max_budget = {
        "gpt-4": {"budget_limit": 100.0, "time_period": "1d"},
        "claude-3": {"budget_limit": 50.0, "time_period": "1d"},
    }

    # Test direct model match
    resolved = resolve_model_budget(model="gpt-4", model_max_budget=model_max_budget)
    assert resolved.budget_model == "gpt-4"
    assert resolved.budget_config.max_budget == 100.0

    # Test model with provider: the counter is keyed on the CONFIGURED name,
    # not the request name, so every reader looks it up the same way.
    resolved = resolve_model_budget(model="openai/gpt-4", model_max_budget=model_max_budget)
    assert resolved.budget_model == "gpt-4"
    assert resolved.budget_config.max_budget == 100.0

    # Test non-existent model
    assert resolve_model_budget(model="non-existent", model_max_budget=model_max_budget) is None


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
    with patch.object(budget_limiter, "_get_spend_for_model_budget", return_value=50.0):
        assert await budget_limiter.is_key_within_model_budget(user_api_key, "gpt-4") is True

    # Test when model exceeds budget
    with patch.object(budget_limiter, "_get_spend_for_model_budget", return_value=150.0):
        with pytest.raises(litellm.BudgetExceededError):
            await budget_limiter.is_key_within_model_budget(user_api_key, "gpt-4")

    # Test model not in budget config
    assert await budget_limiter.is_key_within_model_budget(user_api_key, "non-existent") is True


# Test _get_spend_for_model_budget
@pytest.mark.asyncio
async def test_get_spend_for_model_budget_reads_the_configured_model_key(
    budget_limiter,
):
    from litellm.proxy.hooks.model_max_budget_limiter import (
        VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX,
    )

    model_max_budget = {"gpt-4": {"budget_limit": 100.0, "time_period": "1d"}}
    # openai/gpt-4 resolves to the configured "gpt-4" entry, so the lookup must
    # hit the same key async_log_success_event writes.
    resolved = resolve_model_budget(model="openai/gpt-4", model_max_budget=model_max_budget)

    async def _spend(key):
        return 50.0 if key == f"{VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX}:test-key:gpt-4:1d" else None

    with patch.object(budget_limiter.dual_cache, "async_get_cache", side_effect=_spend) as mock_get:
        spend = await budget_limiter._get_spend_for_model_budget(
            entity_type=Litellm_EntityType.KEY,
            entity_id="test-key",
            model="openai/gpt-4",
            resolved=resolved,
        )
        assert spend == 50.0
        assert [call.kwargs["key"] for call in mock_get.call_args_list] == [
            f"{VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX}:test-key:gpt-4:1d",
            f"{VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX}:test-key:openai/gpt-4:1d",
        ]


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
            "metadata": {"user_api_key_model_max_budget": user_api_key_model_max_budget},
        },
    }
    with patch.object(
        budget_limiter,
        "_increment_spend_for_key",
        new_callable=AsyncMock,
    ) as mock_increment:
        await budget_limiter.async_log_success_event(kwargs, response_obj=None, start_time=None, end_time=None)
        mock_increment.assert_awaited_once()
        call_kwargs = mock_increment.call_args.kwargs
        spend_key = call_kwargs["spend_key"]
        assert spend_key == (f"{VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX}:{virtual_key}:{model}:{budget_duration}")
        assert call_kwargs["response_cost"] == 0.05


# Test is_end_user_within_model_budget
@pytest.mark.asyncio
async def test_is_end_user_within_model_budget(budget_limiter):
    # Test when model is within budget
    with patch.object(budget_limiter, "_get_spend_for_model_budget", return_value=50.0):
        assert (
            await budget_limiter.is_end_user_within_model_budget(
                "test-user",
                {"gpt-4": {"budget_limit": 100.0, "time_period": "1d"}},
                "gpt-4",
            )
            is True
        )

    # Test when model exceeds budget
    with patch.object(budget_limiter, "_get_spend_for_model_budget", return_value=150.0):
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


# Test _get_spend_for_model_budget for the end-user scope
@pytest.mark.asyncio
async def test_get_spend_for_end_user_model_budget(budget_limiter):
    from litellm.proxy.hooks.model_max_budget_limiter import (
        END_USER_SPEND_CACHE_KEY_PREFIX,
    )

    model_max_budget = {"gpt-4": {"budget_limit": 100.0, "time_period": "1d"}}
    resolved = resolve_model_budget(model="openai/gpt-4", model_max_budget=model_max_budget)

    async def _spend(key):
        return 50.0 if key == f"{END_USER_SPEND_CACHE_KEY_PREFIX}:test-user:gpt-4:1d" else None

    with patch.object(budget_limiter.dual_cache, "async_get_cache", side_effect=_spend) as mock_get:
        spend = await budget_limiter._get_spend_for_model_budget(
            entity_type=Litellm_EntityType.END_USER,
            entity_id="test-user",
            model="openai/gpt-4",
            resolved=resolved,
        )
        assert spend == 50.0
        assert [call.kwargs["key"] for call in mock_get.call_args_list] == [
            f"{END_USER_SPEND_CACHE_KEY_PREFIX}:test-user:gpt-4:1d",
            f"{END_USER_SPEND_CACHE_KEY_PREFIX}:test-user:openai/gpt-4:1d",
        ]


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
        await budget_limiter.async_log_success_event(kwargs, response_obj=None, start_time=None, end_time=None)
        mock_increment.assert_awaited_once()
        call_kwargs = mock_increment.call_args.kwargs
        spend_key = call_kwargs["spend_key"]
        # The cache key must use the model_group name, NOT the deployment name
        assert spend_key == (f"{VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX}:{virtual_key}:{model_group}:{budget_duration}")
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
        await budget_limiter.async_log_success_event(kwargs, response_obj=None, start_time=None, end_time=None)
        mock_increment.assert_awaited_once()
        call_kwargs = mock_increment.call_args.kwargs
        spend_key = call_kwargs["spend_key"]
        assert spend_key == (f"{VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX}:{virtual_key}:{model}:{budget_duration}")


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
        await budget_limiter.async_log_success_event(kwargs, response_obj=None, start_time=None, end_time=None)
        mock_increment.assert_awaited_once()
        call_kwargs = mock_increment.call_args.kwargs
        spend_key = call_kwargs["spend_key"]
        assert spend_key == (f"{END_USER_SPEND_CACHE_KEY_PREFIX}:{end_user_id}:{model_group}:{budget_duration}")


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
            "metadata": {"user_api_key_end_user_model_max_budget": user_api_key_end_user_model_max_budget},
        },
    }
    with patch.object(
        budget_limiter,
        "_increment_spend_for_key",
        new_callable=AsyncMock,
    ) as mock_increment:
        await budget_limiter.async_log_success_event(kwargs, response_obj=None, start_time=None, end_time=None)
        mock_increment.assert_awaited_once()
        call_kwargs = mock_increment.call_args.kwargs
        spend_key = call_kwargs["spend_key"]
        assert spend_key == (f"{END_USER_SPEND_CACHE_KEY_PREFIX}:{end_user_id}:{model}:{budget_duration}")
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
            await limiter.async_log_success_event(kwargs, response_obj=None, start_time=None, end_time=None)
            mock_push.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_fallback_model_within_budget_returns_none_without_fallbacks(
    budget_limiter,
):
    user_api_key = UserAPIKeyAuth(token="test-key", budget_fallbacks={})
    assert await budget_limiter.get_fallback_model_within_budget(user_api_key, "gpt-4") is None


@pytest.mark.asyncio
async def test_get_fallback_model_within_budget_returns_first_within_budget(
    budget_limiter,
):
    user_api_key = UserAPIKeyAuth(
        token="test-key",
        model_max_budget={"gpt-4o-mini": {"budget_limit": 100.0, "time_period": "1d"}},
        budget_fallbacks={"gpt-4": ["gpt-4o-mini", "claude-haiku"]},
    )
    with patch.object(budget_limiter, "_get_spend_for_model_budget", return_value=1.0):
        result = await budget_limiter.get_fallback_model_within_budget(user_api_key, "gpt-4")
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

    async def _spend_for_model(entity_type, entity_id, model, resolved):
        return 150.0 if resolved.budget_model == "gpt-4o-mini" else 1.0

    with patch.object(
        budget_limiter,
        "_get_spend_for_model_budget",
        side_effect=_spend_for_model,
    ):
        result = await budget_limiter.get_fallback_model_within_budget(user_api_key, "gpt-4")
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
    with patch.object(budget_limiter, "_get_spend_for_model_budget", return_value=150.0):
        result = await budget_limiter.get_fallback_model_within_budget(user_api_key, "gpt-4")
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
            await budget_limiter.async_log_success_event(kwargs, response_obj=None, start_time=None, end_time=None)
            mock_push.assert_not_awaited()


def _success_kwargs(
    *,
    model_group,
    deployment_model=None,
    response_cost=0.5,
    key_hash=None,
    key_model_max_budget=None,
    user_id=None,
    user_model_max_budget=None,
    end_user_id=None,
    end_user_model_max_budget=None,
):
    return {
        "standard_logging_object": {
            "response_cost": response_cost,
            "model": deployment_model or model_group,
            "model_group": model_group,
            "end_user": end_user_id,
            "metadata": {
                "user_api_key_hash": key_hash,
                "user_api_key_user_id": user_id,
                "user_api_key_end_user_id": end_user_id,
            },
        },
        "litellm_params": {
            "metadata": {
                "user_api_key_model_max_budget": key_model_max_budget,
                "user_api_key_user_model_max_budget": user_model_max_budget,
                "user_api_key_end_user_model_max_budget": end_user_model_max_budget,
            },
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_model",
    ["gpt-4", "openai/gpt-4"],
    ids=["request_model_matches_budget_key", "request_model_carries_provider_prefix"],
)
async def test_logged_spend_is_visible_to_key_info_usage_and_enforcement(request_model):
    """
    The counter written post-call, the counter enforcement reads and the counter
    /key/info reports must be one and the same, including when the request model
    is not byte-identical to the configured budget key.

    Regression: the increment used to be keyed on the REQUEST model while
    /key/info only ever looked up the CONFIGURED model, so a key could be
    actively blocked at 429 while reporting current_spend 0.
    """
    dual_cache = DualCache()
    limiter = _PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=dual_cache)
    key_hash = "vk-hash"
    model_max_budget = {"gpt-4": {"budget_limit": 1.0, "time_period": "1d"}}

    await limiter.async_log_success_event(
        _success_kwargs(
            model_group=request_model,
            response_cost=0.75,
            key_hash=key_hash,
            key_model_max_budget=model_max_budget,
        ),
        response_obj=None,
        start_time=None,
        end_time=None,
    )

    usage = await build_model_max_budget_usage(
        entity_type=Litellm_EntityType.KEY,
        entity_id=key_hash,
        model_max_budget=model_max_budget,
        cache=dual_cache,
    )
    assert usage == {
        "gpt-4": {
            "current_spend": 0.75,
            "budget_limit": 1.0,
            "time_period": "1d",
        }
    }

    user_api_key = UserAPIKeyAuth(token=key_hash, model_max_budget=model_max_budget)
    # Still under the 1.0 limit.
    assert await limiter.is_key_within_model_budget(user_api_key, request_model) is True

    await limiter.async_log_success_event(
        _success_kwargs(
            model_group=request_model,
            response_cost=0.75,
            key_hash=key_hash,
            key_model_max_budget=model_max_budget,
        ),
        response_obj=None,
        start_time=None,
        end_time=None,
    )
    with pytest.raises(litellm.BudgetExceededError):
        await limiter.is_key_within_model_budget(user_api_key, request_model)

    usage_after = await build_model_max_budget_usage(
        entity_type=Litellm_EntityType.KEY,
        entity_id=key_hash,
        model_max_budget=model_max_budget,
        cache=dual_cache,
    )
    assert usage_after["gpt-4"]["current_spend"] == 1.5


@pytest.mark.asyncio
async def test_user_model_budget_is_tracked_and_enforced():
    """
    An internal user's own model_max_budget must be incremented post-call and
    enforced, independently of any key-level budget.
    """
    dual_cache = DualCache()
    limiter = _PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=dual_cache)
    user_id = "user-1"
    user_model_max_budget = {"gpt-4": {"budget_limit": 1.0, "time_period": "1mo"}}

    assert (
        await limiter.is_user_within_model_budget(
            user_id=user_id,
            user_model_max_budget=user_model_max_budget,
            model="openai/gpt-4",
        )
        is True
    )

    await limiter.async_log_success_event(
        _success_kwargs(
            model_group="openai/gpt-4",
            response_cost=1.5,
            user_id=user_id,
            user_model_max_budget=user_model_max_budget,
        ),
        response_obj=None,
        start_time=None,
        end_time=None,
    )

    assert await build_model_max_budget_usage(
        entity_type=Litellm_EntityType.USER,
        entity_id=user_id,
        model_max_budget=user_model_max_budget,
        cache=dual_cache,
    ) == {"gpt-4": {"current_spend": 1.5, "budget_limit": 1.0, "time_period": "1mo"}}

    with pytest.raises(litellm.BudgetExceededError) as exc:
        await limiter.is_user_within_model_budget(
            user_id=user_id,
            user_model_max_budget=user_model_max_budget,
            model="openai/gpt-4",
        )
    assert exc.value.entity_type == Litellm_EntityType.USER.value


@pytest.mark.asyncio
async def test_user_model_budget_counter_is_separate_from_the_key_counter():
    """
    A key budget and a user budget over the same model are two independent
    counters, so one request must charge each exactly once.
    """
    dual_cache = DualCache()
    limiter = _PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=dual_cache)
    model_max_budget = {"gpt-4": {"budget_limit": 10.0, "time_period": "1d"}}

    await limiter.async_log_success_event(
        _success_kwargs(
            model_group="gpt-4",
            response_cost=2.0,
            key_hash="vk-hash",
            key_model_max_budget=model_max_budget,
            user_id="user-1",
            user_model_max_budget=model_max_budget,
        ),
        response_obj=None,
        start_time=None,
        end_time=None,
    )

    assert await dual_cache.async_get_cache(key="virtual_key_spend:vk-hash:gpt-4:1d") == 2.0
    assert await dual_cache.async_get_cache(key="user_model_spend:user-1:gpt-4:1d") == 2.0


@pytest.mark.asyncio
async def test_two_models_on_one_key_do_not_share_a_budget_window():
    """
    A key budgeting two models over different periods must own one window start
    per model: a shared start lets the shorter period restart the longer one.
    """
    dual_cache = DualCache()
    limiter = _PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=dual_cache)
    model_max_budget = {
        "gpt-4": {"budget_limit": 10.0, "time_period": "1d"},
        "claude-3": {"budget_limit": 10.0, "time_period": "30d"},
    }

    start_time_keys = []
    with patch.object(limiter, "_increment_spend_for_key", new_callable=AsyncMock) as mock_increment:
        for model in ("gpt-4", "claude-3"):
            await limiter.async_log_success_event(
                _success_kwargs(
                    model_group=model,
                    key_hash="vk-hash",
                    key_model_max_budget=model_max_budget,
                ),
                response_obj=None,
                start_time=None,
                end_time=None,
            )
        start_time_keys = [call.kwargs["start_time_key"] for call in mock_increment.call_args_list]

    assert start_time_keys == [
        "virtual_key_budget_start_time:vk-hash:gpt-4:1d",
        "virtual_key_budget_start_time:vk-hash:claude-3:30d",
    ]
    assert len(set(start_time_keys)) == 2


@pytest.mark.asyncio
async def test_no_increment_when_no_scope_budgets_the_model():
    dual_cache = DualCache()
    limiter = _PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=dual_cache)
    with patch.object(limiter, "_increment_spend_for_key", new_callable=AsyncMock) as mock_increment:
        await limiter.async_log_success_event(
            _success_kwargs(
                model_group="gpt-4",
                key_hash="vk-hash",
                key_model_max_budget={"claude-3": {"budget_limit": 1.0, "time_period": "1d"}},
                user_id="user-1",
                user_model_max_budget={},
            ),
            response_obj=None,
            start_time=None,
            end_time=None,
        )
    mock_increment.assert_not_awaited()


@pytest.mark.asyncio
async def test_build_model_max_budget_usage_skips_unusable_entries():
    """A malformed or period-less entry must be omitted, not crash the report."""
    dual_cache = DualCache()
    await dual_cache.async_set_cache(key="virtual_key_spend:vk:gpt-4:1d", value=3.0)

    usage = await build_model_max_budget_usage(
        entity_type=Litellm_EntityType.KEY,
        entity_id="vk",
        model_max_budget={
            "gpt-4": {"budget_limit": 10.0, "time_period": "1d"},
            "no-period": {"budget_limit": 10.0},
            "bad-period": {"budget_limit": 10.0, "time_period": "not-a-duration"},
        },
        cache=dual_cache,
    )
    assert usage == {"gpt-4": {"current_spend": 3.0, "budget_limit": 10.0, "time_period": "1d"}}


@pytest.mark.asyncio
async def test_bedrock_traffic_charges_the_bare_family_name_budget():
    """
    The reported case: a budget configured as "claude-opus-4-8" with traffic on
    "bedrock/anthropic.claude-opus-4-8". Before the fix nothing matched, so spend
    was never tracked and the budget was never enforced no matter how far over it
    the key went.
    """
    dual_cache = DualCache()
    limiter = _PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=dual_cache)
    key_hash = "vk-hash"
    model_max_budget = {"claude-opus-4-8": {"budget_limit": 1.0, "time_period": "18h"}}
    user_api_key = UserAPIKeyAuth(token=key_hash, model_max_budget=model_max_budget)

    await limiter.async_log_success_event(
        _success_kwargs(
            model_group="bedrock/anthropic.claude-opus-4-8",
            response_cost=1.5,
            key_hash=key_hash,
            key_model_max_budget=model_max_budget,
        ),
        response_obj=None,
        start_time=None,
        end_time=None,
    )

    assert await build_model_max_budget_usage(
        entity_type=Litellm_EntityType.KEY,
        entity_id=key_hash,
        model_max_budget=model_max_budget,
        cache=dual_cache,
    ) == {
        "claude-opus-4-8": {
            "current_spend": 1.5,
            "budget_limit": 1.0,
            "time_period": "18h",
        }
    }

    with pytest.raises(litellm.BudgetExceededError):
        await limiter.is_key_within_model_budget(user_api_key, "bedrock/anthropic.claude-opus-4-8")


@pytest.mark.asyncio
async def test_user_model_budget_window_resets_when_the_period_elapses():
    """
    A monthly user budget must start a fresh window once the period elapses,
    and the window start must be scoped to that one budget model so a second
    model on a shorter period cannot drag it forward.
    """
    from litellm.proxy.hooks.model_max_budget_limiter import (
        model_budget_spend_cache_key,
        model_budget_start_time_cache_key,
    )

    dual_cache = DualCache()
    limiter = _PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=dual_cache)
    user_id = "user-1"
    user_model_max_budget = {"gpt-4": {"budget_limit": 1.0, "time_period": "1mo"}}
    spend_key = model_budget_spend_cache_key(
        entity_type=Litellm_EntityType.USER,
        entity_id=user_id,
        budget_model="gpt-4",
        budget_duration="1mo",
    )
    start_time_key = model_budget_start_time_cache_key(
        entity_type=Litellm_EntityType.USER,
        entity_id=user_id,
        budget_model="gpt-4",
        budget_duration="1mo",
    )

    kwargs = _success_kwargs(
        model_group="gpt-4",
        response_cost=1.5,
        user_id=user_id,
        user_model_max_budget=user_model_max_budget,
    )
    await limiter.async_log_success_event(kwargs, response_obj=None, start_time=None, end_time=None)
    assert await dual_cache.async_get_cache(key=spend_key) == 1.5
    with pytest.raises(litellm.BudgetExceededError):
        await limiter.is_user_within_model_budget(
            user_id=user_id,
            user_model_max_budget=user_model_max_budget,
            model="gpt-4",
        )

    # Age the window past its period. The next charge opens a new window rather
    # than adding to the exhausted one.
    elapsed = duration_in_seconds("1mo") + 60
    await dual_cache.async_set_cache(
        key=start_time_key,
        value=datetime.now(timezone.utc).timestamp() - elapsed,
        ttl=elapsed,
    )

    await limiter.async_log_success_event(kwargs, response_obj=None, start_time=None, end_time=None)
    assert await dual_cache.async_get_cache(key=spend_key) == 1.5
    assert await build_model_max_budget_usage(
        entity_type=Litellm_EntityType.USER,
        entity_id=user_id,
        model_max_budget=user_model_max_budget,
        cache=dual_cache,
    ) == {"gpt-4": {"current_spend": 1.5, "budget_limit": 1.0, "time_period": "1mo"}}


@pytest.mark.asyncio
async def test_a_zero_dollar_cap_blocks_the_model():
    """
    0 is the operator saying "nobody may spend anything on this model", which is
    the strictest cap expressible, not the absence of one. Skipping it on
    falsiness turned the strictest setting into no setting at all, so the model
    stayed wide open. The dashboard editor can produce this value, so it has to
    mean something.
    """
    dual_cache = DualCache()
    limiter = _PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=dual_cache)
    key = UserAPIKeyAuth(
        token="hash-zero",
        model_max_budget={"gpt-4": {"budget_limit": 0, "time_period": "1d"}},
    )

    with pytest.raises(litellm.BudgetExceededError) as exc:
        await limiter.is_key_within_model_budget(user_api_key_dict=key, model="gpt-4")
    assert exc.value.max_budget == 0


@pytest.mark.asyncio
async def test_a_zero_dollar_cap_is_reported_as_a_cap_not_as_absent():
    """The usage endpoints must show the 0 too, or an operator cannot see the block they configured."""
    assert await build_model_max_budget_usage(
        entity_type=Litellm_EntityType.KEY,
        entity_id="hash-zero",
        model_max_budget={"gpt-4": {"budget_limit": 0, "time_period": "1d"}},
        cache=DualCache(),
    ) == {"gpt-4": {"current_spend": 0.0, "budget_limit": 0.0, "time_period": "1d"}}


@pytest.mark.asyncio
async def test_spend_exactly_at_the_cap_is_refused():
    """
    Spending the whole budget exhausts it. `>` let a caller sit exactly on the
    limit and keep going, and every sibling budget check in the codebase
    (RouterBudgetLimiting, the key and team budget checks) uses `>=`.
    """
    dual_cache = DualCache()
    limiter = _PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=dual_cache)
    budget = {"gpt-4": {"budget_limit": 2.0, "time_period": "1d"}}
    key = UserAPIKeyAuth(token="hash-exact", model_max_budget=budget)

    await limiter.async_log_success_event(
        _success_kwargs(model_group="gpt-4", response_cost=2.0, key_hash="hash-exact", key_model_max_budget=budget),
        response_obj=None,
        start_time=None,
        end_time=None,
    )

    with pytest.raises(litellm.BudgetExceededError):
        await limiter.is_key_within_model_budget(user_api_key_dict=key, model="gpt-4")


@pytest.mark.asyncio
async def test_usage_report_reads_every_counter_in_one_batched_lookup():
    """
    model_max_budget is caller-supplied and unbounded in size, so one cache
    coroutine per configured model let a large map fan out into an unbounded
    number of concurrent lookups on an endpoint anyone holding the key can call.
    One batched read keeps it to a single round trip whatever the map's size.
    """
    dual_cache = DualCache()
    budget = {f"model-{i}": {"budget_limit": 1.0, "time_period": "1d"} for i in range(50)}

    with (
        patch.object(dual_cache, "async_batch_get_cache", new=AsyncMock(return_value=[None] * 50)) as batched,
        patch.object(dual_cache, "async_get_cache", new=AsyncMock()) as single,
    ):
        usage = await build_model_max_budget_usage(
            entity_type=Litellm_EntityType.KEY,
            entity_id="hash-many",
            model_max_budget=budget,
            cache=dual_cache,
        )

    assert batched.await_count == 1
    assert len(batched.await_args.kwargs["keys"]) == 50
    assert single.await_count == 0
    assert len(usage) == 50


@pytest.mark.asyncio
async def test_usage_report_survives_a_batch_lookup_that_returns_nothing():
    """
    async_batch_get_cache swallows its own failures and returns None. Zipping
    that against the budgets would raise and take the whole /key/info response
    with it, so an unusable result has to read as a miss instead.
    """
    dual_cache = DualCache()
    with patch.object(dual_cache, "async_batch_get_cache", new=AsyncMock(return_value=None)):
        assert await build_model_max_budget_usage(
            entity_type=Litellm_EntityType.KEY,
            entity_id="hash-none",
            model_max_budget={"gpt-4": {"budget_limit": 1.0, "time_period": "1d"}},
            cache=dual_cache,
        ) == {"gpt-4": {"current_spend": 0.0, "budget_limit": 1.0, "time_period": "1d"}}


@pytest.mark.asyncio
async def test_one_malformed_scope_does_not_abort_the_other_scopes():
    """
    Every scope is resolved before any of them is incremented, so a single
    unusable entry used to raise out of resolution and leave the key counter
    unwritten too. The key's budget is well formed here and must still be
    charged despite the user's entry being garbage.
    """
    dual_cache = DualCache()
    limiter = _PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=dual_cache)
    key_budget = {"gpt-4": {"budget_limit": 10.0, "time_period": "1d"}}

    await limiter.async_log_success_event(
        _success_kwargs(
            model_group="gpt-4",
            response_cost=0.25,
            key_hash="hash-mixed",
            key_model_max_budget=key_budget,
            user_id="user-mixed",
            user_model_max_budget={"gpt-4": {"budget_limit": "not-a-number", "time_period": "1d"}},
        ),
        response_obj=None,
        start_time=None,
        end_time=None,
    )

    assert await build_model_max_budget_usage(
        entity_type=Litellm_EntityType.KEY,
        entity_id="hash-mixed",
        model_max_budget=key_budget,
        cache=dual_cache,
    ) == {"gpt-4": {"current_spend": 0.25, "budget_limit": 10.0, "time_period": "1d"}}


@pytest.mark.asyncio
async def test_an_unusable_budget_entry_is_not_enforced_instead_of_raising():
    """
    A config typo must not turn every request for that model into a 500. It
    cannot be keyed, so it cannot be enforced; the write path rejects these, so
    reaching here means config.yaml or a direct DB edit.
    """
    limiter = _PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=DualCache())
    key = UserAPIKeyAuth(
        token="hash-malformed",
        model_max_budget={"gpt-4": {"budget_limit": "not-a-number", "time_period": "1d"}},
    )

    assert await limiter.is_key_within_model_budget(user_api_key_dict=key, model="gpt-4") is True


def test_resolve_model_budget_returns_none_for_an_unusable_entry():
    assert (
        resolve_model_budget(
            model="gpt-4",
            model_max_budget={"gpt-4": {"budget_limit": "not-a-number", "time_period": "1d"}},
        )
        is None
    )


def test_a_malformed_specific_entry_does_not_hide_a_usable_family_budget():
    """
    The candidate chain is most-specific-first and already falls through an entry
    that is ABSENT. An entry that will not parse is indistinguishable from absent
    as far as enforcement goes, so it has to fall through too: otherwise one bad
    provider-prefixed entry silently disables the valid bare-family budget sitting
    next to it, and the model goes uncapped.
    """
    resolved = resolve_model_budget(
        model="openai/gpt-4",
        model_max_budget={
            "openai/gpt-4": {"budget_limit": "not-a-number", "time_period": "1d"},
            "gpt-4": {"budget_limit": 7.0, "time_period": "1d"},
        },
    )

    assert resolved is not None
    assert resolved.budget_model == "gpt-4"
    assert resolved.budget_config.max_budget == 7.0


@pytest.mark.asyncio
async def test_a_malformed_specific_entry_still_enforces_the_family_budget():
    """The fall-through has to reach enforcement, not just resolution."""
    dual_cache = DualCache()
    limiter = _PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=dual_cache)
    budget = {
        "openai/gpt-4": {"budget_limit": "not-a-number", "time_period": "1d"},
        "gpt-4": {"budget_limit": 1.0, "time_period": "1d"},
    }
    key = UserAPIKeyAuth(token="hash-fallthrough", model_max_budget=budget)

    await limiter.async_log_success_event(
        _success_kwargs(
            model_group="openai/gpt-4",
            response_cost=2.0,
            key_hash="hash-fallthrough",
            key_model_max_budget=budget,
        ),
        response_obj=None,
        start_time=None,
        end_time=None,
    )

    with pytest.raises(litellm.BudgetExceededError):
        await limiter.is_key_within_model_budget(user_api_key_dict=key, model="openai/gpt-4")


def test_documented_budget_spelling_survives_model_validate():
    """
    `budget_limit` / `time_period` are the spelling the docs, the CRUD endpoints
    and the dashboard editor all use, and BudgetConfig maps them onto
    `max_budget` / `budget_duration` inside its `__init__`.

    Pydantic v2 normally bypasses a custom `__init__` in `model_validate`, and
    this code path validates rather than constructing. It works today, but that
    is a property of the installed Pydantic rather than of anything in this
    repository, so an upgrade could silently stop applying the mapping and
    quietly disable every budget written in the documented spelling. Pinned here
    so that becomes a red test instead of an outage.
    """
    from litellm.types.utils import BudgetConfig

    validated = BudgetConfig.model_validate({"budget_limit": 5, "time_period": "1d"})
    assert validated.max_budget == 5.0
    assert validated.budget_duration == "1d"

    # Control: an unrecognised key must NOT populate max_budget, or the assertion
    # above would also pass against a model that accepted anything at all.
    ignored = BudgetConfig.model_validate({"bogus_limit": 5, "time_period": "1d"})
    assert ignored.max_budget is None


def test_resolution_accepts_both_documented_spellings():
    """The resolver is what enforcement, tracking and reporting all go through."""
    for budget in (
        {"gpt-4": {"budget_limit": 5, "time_period": "1d"}},
        {"gpt-4": {"max_budget": 5, "budget_duration": "1d"}},
    ):
        resolved = resolve_model_budget(model="gpt-4", model_max_budget=budget)
        assert resolved is not None, f"{budget} resolved to nothing"
        assert resolved.budget_config.max_budget == 5.0
        assert resolved.budget_config.budget_duration == "1d"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "entity_type, prefix",
    [
        (Litellm_EntityType.KEY, "virtual_key_spend"),
        (Litellm_EntityType.END_USER, "end_user_model_spend"),
    ],
)
async def test_a_pre_upgrade_counter_keyed_on_the_request_model_still_enforces(entity_type, prefix):
    """An upgrading proxy must not hand out a second allowance for the window it is already in.

    Before the counter key moved to the configured budget model, spend for a
    request on `openai/gpt-4` against a budget configured as `gpt-4` was both
    written to and enforced on `{prefix}:{id}:openai/gpt-4:1d`. Reading only the
    configured-model key finds that counter empty and admits another full budget
    until the window expires.
    """
    limiter = _PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=DualCache())
    model_max_budget = {"gpt-4": {"budget_limit": 10.0, "time_period": "1d"}}
    await limiter.dual_cache.async_set_cache(key=f"{prefix}:entity-1:openai/gpt-4:1d", value=25.0, ttl=86400)

    if entity_type == Litellm_EntityType.KEY:
        budget_check = limiter.is_key_within_model_budget(
            user_api_key_dict=UserAPIKeyAuth(token="entity-1", model_max_budget=model_max_budget),
            model="openai/gpt-4",
        )
    else:
        budget_check = limiter.is_end_user_within_model_budget(
            end_user_id="entity-1",
            end_user_model_max_budget=model_max_budget,
            model="openai/gpt-4",
        )
    with pytest.raises(litellm.BudgetExceededError) as exc_info:
        await budget_check
    assert exc_info.value.current_cost == 25.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "legacy_spend, current_spend, expect_blocked",
    [(6.0, 5.0, True), (2.0, 3.0, False)],
)
async def test_the_pre_upgrade_and_post_upgrade_counters_add_up_over_one_window(
    legacy_spend, current_spend, expect_blocked
):
    """The two counters hold disjoint halves of one window, so the window's spend is their sum.

    Nothing writes the request-model spelling once this version is running, so
    the legacy counter is frozen at whatever the previous version charged and
    the configured-model counter carries everything since. Either one alone
    under-reports the window: 6 + 5 is over a cap of 10 that neither half
    reaches on its own.
    """
    limiter = _PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=DualCache())
    model_max_budget = {"gpt-4": {"budget_limit": 10.0, "time_period": "1d"}}
    await limiter.dual_cache.async_set_cache(
        key="virtual_key_spend:entity-1:openai/gpt-4:1d", value=legacy_spend, ttl=86400
    )
    await limiter.dual_cache.async_set_cache(key="virtual_key_spend:entity-1:gpt-4:1d", value=current_spend, ttl=86400)

    async def enforce():
        return await limiter.is_key_within_model_budget(
            user_api_key_dict=UserAPIKeyAuth(token="entity-1", model_max_budget=model_max_budget),
            model="openai/gpt-4",
        )

    if expect_blocked:
        with pytest.raises(litellm.BudgetExceededError) as exc_info:
            await enforce()
        assert exc_info.value.current_cost == legacy_spend + current_spend
    else:
        assert await enforce() is True


@pytest.mark.asyncio
async def test_the_configured_model_counter_is_never_counted_twice():
    """When the request names the budget exactly there is no legacy counter, only the one key.

    Both keys are `virtual_key_spend:entity-1:gpt-4:1d` here, so a lookup that
    added them without noticing would charge 12 against a cap of 10 and refuse a
    key that has spent 6.
    """
    limiter = _PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=DualCache())
    await limiter.dual_cache.async_set_cache(key="virtual_key_spend:entity-1:gpt-4:1d", value=6.0, ttl=86400)

    assert (
        await limiter.is_key_within_model_budget(
            user_api_key_dict=UserAPIKeyAuth(
                token="entity-1",
                model_max_budget={"gpt-4": {"budget_limit": 10.0, "time_period": "1d"}},
            ),
            model="gpt-4",
        )
        is True
    )


@pytest.mark.asyncio
async def test_the_pre_upgrade_counter_is_no_longer_read_a_window_after_start_up(monkeypatch):
    """The carry is bounded, so it cannot become a permanent second lookup on every request.

    A counter written by the previous version belongs to a window that was
    already open when this process replaced it, so once a full window has passed
    since start-up there is nothing left for the lookup to find.
    """
    import litellm.proxy.hooks.model_max_budget_limiter as limiter_module

    limiter = _PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=DualCache())
    model_max_budget = {"gpt-4": {"budget_limit": 10.0, "time_period": "1d"}}
    await limiter.dual_cache.async_set_cache(key="virtual_key_spend:entity-1:openai/gpt-4:1d", value=25.0, ttl=86400)
    user_api_key = UserAPIKeyAuth(token="entity-1", model_max_budget=model_max_budget)

    # Control: within the first window since start-up the same counter blocks,
    # so the assertion below cannot pass against a lookup that never worked.
    with pytest.raises(litellm.BudgetExceededError):
        await limiter.is_key_within_model_budget(user_api_key_dict=user_api_key, model="openai/gpt-4")

    monkeypatch.setattr(limiter_module, "_PROCESS_STARTED_AT", limiter_module.time.monotonic() - 86401)
    assert await limiter.is_key_within_model_budget(user_api_key_dict=user_api_key, model="openai/gpt-4") is True


@pytest.mark.asyncio
async def test_the_user_scope_has_no_pre_upgrade_counter_to_carry():
    """The user scope is introduced by this change, so a request-model key under it is not one of ours.

    Reading one would invent a counter no previous version ever wrote, which is
    the opposite of preserving one.
    """
    limiter = _PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=DualCache())
    model_max_budget = {"gpt-4": {"budget_limit": 10.0, "time_period": "1d"}}
    await limiter.dual_cache.async_set_cache(key="user_model_spend:u1:openai/gpt-4:1d", value=25.0, ttl=86400)

    assert (
        await limiter.is_user_within_model_budget(
            user_id="u1", user_model_max_budget=model_max_budget, model="openai/gpt-4"
        )
        is True
    )

    # Control: the same overspend under the key this scope does own must block,
    # or the assertion above would pass against a scope that enforces nothing.
    await limiter.dual_cache.async_set_cache(key="user_model_spend:u1:gpt-4:1d", value=25.0, ttl=86400)
    with pytest.raises(litellm.BudgetExceededError):
        await limiter.is_user_within_model_budget(
            user_id="u1", user_model_max_budget=model_max_budget, model="openai/gpt-4"
        )
