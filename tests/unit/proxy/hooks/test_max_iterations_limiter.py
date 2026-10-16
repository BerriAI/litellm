"""
Unit Tests for the max iterations limiter for the proxy.

Tests that session-scoped iteration counting works correctly:
- Enforces max_iterations per session_id (read from agent litellm_params)
- Different sessions have independent counters
"""

import asyncio
from typing import Final
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.agent_endpoints.agent_registry import AgentRegistry
from litellm.proxy.hooks.max_iterations_limiter import PROXY_MaxIterationsHandler
from litellm.proxy.utils import InternalUsageCache
from litellm.types.agents import AgentResponse


def _make_mock_agent(max_iterations: int, agent_id: str = "agent-test-123") -> AgentResponse:
    return AgentResponse(
        agent_id=agent_id,
        agent_name="test-agent",
        litellm_params={"max_iterations": max_iterations},
        agent_card_params={"name": "test-agent", "version": "1.0.0"},
    )


@pytest.mark.asyncio
async def test_max_iterations_basic_enforcement():
    """
    Test that max_iterations is enforced per session_id.

    - 3 requests with the same session_id should succeed when max_iterations=3
    - 4th request should raise 429
    """
    local_cache = DualCache()
    handler = PROXY_MaxIterationsHandler(
        internal_usage_cache=InternalUsageCache(local_cache),
    )
    user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-test-key-1234",
        agent_id="agent-test-123",
    )

    mock_agent = _make_mock_agent(max_iterations=3)

    registry: Final = AgentRegistry()
    registry.register_agent(mock_agent)
    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", registry):
        # First 3 requests should succeed
        for i in range(3):
            await handler.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=local_cache,
                data={"metadata": {"session_id": "session-abc"}},
                call_type="",
            )

        # 4th request should fail with 429
        with pytest.raises(HTTPException) as exc_info:
            await handler.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=local_cache,
                data={"metadata": {"session_id": "session-abc"}},
                call_type="",
            )
        assert exc_info.value.status_code == 429
        assert "max_iterations" in str(exc_info.value.detail).lower()


@pytest.mark.asyncio
async def test_max_iterations_different_sessions_independent():
    """
    Test that different session_ids have independent iteration counters.

    - Session A and Session B each get their own max_iterations budget
    - Exhausting Session A does not affect Session B
    """
    local_cache = DualCache()
    handler = PROXY_MaxIterationsHandler(
        internal_usage_cache=InternalUsageCache(local_cache),
    )
    user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-test-key-5678",
        agent_id="agent-test-123",
    )

    mock_agent = _make_mock_agent(max_iterations=2)

    registry: Final = AgentRegistry()
    registry.register_agent(mock_agent)
    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", registry):
        # Session A: 2 calls succeed
        for _ in range(2):
            await handler.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=local_cache,
                data={"metadata": {"session_id": "session-A"}},
                call_type="",
            )

        # Session B: 2 calls succeed (independent counter)
        for _ in range(2):
            await handler.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=local_cache,
                data={"metadata": {"session_id": "session-B"}},
                call_type="",
            )

        # Session A: 3rd call fails
        with pytest.raises(HTTPException) as exc_info:
            await handler.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=local_cache,
                data={"metadata": {"session_id": "session-A"}},
                call_type="",
            )
        assert exc_info.value.status_code == 429

        # Session B: 3rd call also fails
        with pytest.raises(HTTPException):
            await handler.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=local_cache,
                data={"metadata": {"session_id": "session-B"}},
                call_type="",
            )


@pytest.mark.asyncio
async def test_max_iterations_no_agent_id_passes():
    """
    When no agent_id is set on the key, all requests pass through.
    """
    local_cache = DualCache()
    handler = PROXY_MaxIterationsHandler(
        internal_usage_cache=InternalUsageCache(local_cache),
    )
    user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-test-key-no-agent",
    )

    result = await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=local_cache,
        data={"metadata": {"session_id": "session-any"}},
        call_type="",
    )
    assert result is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "researcher_id,orchestrator_id,researcher_session,orchestrator_session",
    [
        ("researcher", "orchestrator", "shared-trace", "shared-trace"),
        ("parent:child", "parent", "trace", "child:trace"),
    ],
)
async def test_agent_session_iteration_counters_do_not_mix_usage(
    researcher_id: str, orchestrator_id: str, researcher_session: str, orchestrator_session: str
) -> None:
    cache: Final = DualCache()
    handler: Final = PROXY_MaxIterationsHandler(InternalUsageCache(cache))
    registry: Final = AgentRegistry()
    registry.register_agent(_make_mock_agent(4, researcher_id))
    registry.register_agent(_make_mock_agent(2, orchestrator_id))
    researcher: Final = UserAPIKeyAuth(agent_id=researcher_id)
    orchestrator: Final = UserAPIKeyAuth(agent_id=orchestrator_id)
    researcher_data: Final = {"metadata": {"session_id": researcher_session}}
    orchestrator_data: Final = {"metadata": {"session_id": orchestrator_session}}

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", registry):
        for _ in range(3):
            assert await handler.async_pre_call_hook(researcher, cache, researcher_data, "") is None
        for _ in range(2):
            assert await handler.async_pre_call_hook(orchestrator, cache, orchestrator_data, "") is None
        with pytest.raises(HTTPException) as rejected:
            await handler.async_pre_call_hook(orchestrator, cache, orchestrator_data, "")
        assert rejected.value.status_code == 429
        assert "Current count: 3" in str(rejected.value.detail)
        assert await handler.async_pre_call_hook(researcher, cache, researcher_data, "") is None
        with pytest.raises(HTTPException) as researcher_rejected:
            await handler.async_pre_call_hook(researcher, cache, researcher_data, "")
        assert researcher_rejected.value.status_code == 429
        assert "Current count: 5" in str(researcher_rejected.value.detail)


@pytest.mark.asyncio
async def test_key_metadata_iteration_limit_keeps_existing_session_count() -> None:
    cache: Final = DualCache()
    handler: Final = PROXY_MaxIterationsHandler(InternalUsageCache(cache))
    key: Final = UserAPIKeyAuth(metadata={"max_iterations": 2})
    await cache.async_set_cache(key="{session_iterations:existing}:count", value=2)

    with pytest.raises(HTTPException) as rejected:
        await handler.async_pre_call_hook(key, cache, {"metadata": {"session_id": "existing"}}, "")

    assert rejected.value.status_code == 429
    assert "Current count: 3" in str(rejected.value.detail)


@pytest.mark.asyncio
async def test_legacy_agent_id_shares_the_registered_agents_iteration_limit() -> None:
    cache: Final = DualCache()
    handler: Final = PROXY_MaxIterationsHandler(InternalUsageCache(cache))
    registry: Final = AgentRegistry()
    registry.load_agents_from_config(
        (
            {
                "agent_name": "configured-agent",
                "agent_card_params": {"name": "configured-agent", "version": "1"},
                "litellm_params": {"max_iterations": 2},
            },
        )
    )
    legacy_id, agent_id = next(iter(registry.config_agent_legacy_ids.items()))
    data: Final = {"metadata": {"session_id": "trace"}}

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", registry):
        for identity in (legacy_id, agent_id):
            assert await handler.async_pre_call_hook(UserAPIKeyAuth(agent_id=identity), cache, data, "") is None
        with pytest.raises(HTTPException) as rejected:
            await handler.async_pre_call_hook(UserAPIKeyAuth(agent_id=legacy_id), cache, data, "")
        assert rejected.value.status_code == 429
        assert "Current count: 3" in str(rejected.value.detail)


@pytest.mark.asyncio
async def test_agent_iteration_limit_keeps_count_from_an_existing_session() -> None:
    cache: Final = DualCache()
    handler: Final = PROXY_MaxIterationsHandler(InternalUsageCache(cache))
    session_id: Final = "existing-session"
    legacy_key: Final = handler._make_legacy_cache_key(session_id)
    await cache.async_set_cache(key=legacy_key, value=2)
    registry: Final = AgentRegistry()
    registry.register_agent(_make_mock_agent(2))

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", registry):
        with pytest.raises(HTTPException) as rejected:
            await handler.async_pre_call_hook(
                UserAPIKeyAuth(agent_id="agent-test-123"),
                cache,
                {"metadata": {"session_id": session_id}},
                "",
            )

    assert rejected.value.status_code == 429
    assert "Current count: 3" in str(rejected.value.detail)
    legacy_count: Final = await cache.async_get_cache(key=handler._make_legacy_cache_key(session_id))
    scope: Final = await cache.async_get_cache(key=handler._make_agent_scope_cache_key(session_id))
    agent_field: Final = handler._make_agent_scope_field("agent-test-123")
    assert isinstance(scope, dict)
    assert legacy_count - scope["__total_new"] + scope[agent_field] == 3


@pytest.mark.asyncio
async def test_agent_iteration_recovers_conservatively_after_scope_eviction() -> None:
    cache: Final = DualCache()
    handler: Final = PROXY_MaxIterationsHandler(InternalUsageCache(cache))
    session_id: Final = "scope-eviction-session"
    registry: Final = AgentRegistry()
    registry.register_agent(_make_mock_agent(20, "agent-a"))
    registry.register_agent(_make_mock_agent(20, "agent-b"))

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", registry):
        assert await handler._increment_agent_and_get(session_id, "agent-a") == 1
        await cache.async_delete_cache(key=handler._make_agent_scope_cache_key(session_id))
        # With the migration scope gone, prior usage becomes a conservative shared baseline.
        assert await handler._increment_agent_and_get(session_id, "agent-b") == 2
        assert await handler._increment_agent_and_get(session_id, "agent-a") == 2


@pytest.mark.asyncio
async def test_agent_iteration_fails_closed_when_scope_loses_migration_total() -> None:
    cache: Final = DualCache()
    handler: Final = PROXY_MaxIterationsHandler(InternalUsageCache(cache))
    session_id: Final = "inconsistent-scope-session"
    scope_key: Final = handler._make_agent_scope_cache_key(session_id)
    await cache.async_set_cache(key=handler._make_legacy_cache_key(session_id), value=1)
    await cache.async_set_cache(key=scope_key, value={'agent:"agent-a"': 1})

    with pytest.raises(RuntimeError, match="missing its migration total"):
        await handler._increment_agent_and_get(session_id, "agent-a")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "counter,invalid_value",
    tuple(
        (counter, invalid_value)
        for counter in ("legacy", "total_new", "agent")
        for invalid_value in ({}, "bad", True, -1, 0.9, float("nan"), float("inf"))
    )
    + (("total_new", None), ("agent", None)),
)
async def test_malformed_iteration_ledger_is_rejected(counter: str, invalid_value: object) -> None:
    cache: Final = DualCache()
    handler: Final = PROXY_MaxIterationsHandler(InternalUsageCache(cache))
    session_id: Final = "malformed-iteration-ledger"
    agent_field: Final = handler._make_agent_scope_field("agent-a")
    await cache.async_set_cache(
        key=handler._make_legacy_cache_key(session_id), value=invalid_value if counter == "legacy" else 2
    )
    if counter != "legacy":
        await cache.async_set_cache(
            key=handler._make_agent_scope_cache_key(session_id),
            value={
                "__total_new": invalid_value if counter == "total_new" else 0,
                agent_field: invalid_value if counter == "agent" else 0,
            },
        )

    with pytest.raises((RuntimeError, TypeError, ValueError)):
        await handler._increment_agent_and_get(session_id, "agent-a")


@pytest.mark.asyncio
async def test_integral_json_float_iteration_counters_are_preserved() -> None:
    cache: Final = DualCache()
    handler: Final = PROXY_MaxIterationsHandler(InternalUsageCache(cache))
    session_id: Final = "integral-json-float"
    agent_field: Final = handler._make_agent_scope_field("agent-a")
    await cache.async_set_cache(key=handler._make_legacy_cache_key(session_id), value=2.0)
    await cache.async_set_cache(
        key=handler._make_agent_scope_cache_key(session_id), value={"__total_new": 1.0, agent_field: 1.0}
    )

    assert await handler._increment_agent_and_get(session_id, "agent-a") == 3


@pytest.mark.asyncio
async def test_agent_iteration_registers_redis_scripts_when_redis_is_attached_late() -> None:
    class FakeRedisCache:
        def __init__(self) -> None:
            self.scripts: list[str] = []
            self.calls: list[dict[str, object]] = []

        def async_register_script(self, source: str):
            self.scripts.append(source)

            async def call(**kwargs: object) -> object:
                self.calls.append(kwargs)
                return 1

            return call

    cache: Final = DualCache()
    handler: Final = PROXY_MaxIterationsHandler(InternalUsageCache(cache))
    redis_cache: Final = FakeRedisCache()
    cache.redis_cache = redis_cache
    registry: Final = AgentRegistry()
    registry.register_agent(_make_mock_agent(10, "agent-a"))

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", registry):
        assert await handler._increment_agent_and_get("late-redis", "agent-a") == 1

    assert len(redis_cache.scripts) == 1
    assert len(redis_cache.calls) == 1
    assert len(redis_cache.calls[0]["keys"]) == 2


@pytest.mark.asyncio
async def test_agent_iteration_counter_tracks_old_and_new_pods_independently() -> None:
    cache: Final = DualCache()
    handler: Final = PROXY_MaxIterationsHandler(InternalUsageCache(cache))
    session_id: Final = "mixed-rollout-session"
    legacy_key: Final = handler._make_legacy_cache_key(session_id)
    await cache.async_set_cache(key=legacy_key, value=3)
    registry: Final = AgentRegistry()
    registry.register_agent(_make_mock_agent(20, "agent-a"))
    registry.register_agent(_make_mock_agent(20, "agent-b"))

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", registry):
        assert await handler._increment_agent_and_get(session_id, "agent-a") == 4
        assert await handler._increment_agent_and_get(session_id, "agent-b") == 4

        # An old pod still writes only the legacy aggregate.
        await cache.async_set_cache(key=legacy_key, value=6)
        assert await handler._increment_agent_and_get(session_id, "agent-a") == 6
        assert await handler._increment_agent_and_get(session_id, "agent-b") == 6


@pytest.mark.asyncio
async def test_agent_iteration_redis_error_is_not_retried_or_counted_locally() -> None:
    cache: Final = DualCache()
    handler: Final = PROXY_MaxIterationsHandler(InternalUsageCache(cache))
    redis_cache: Final = object()
    cache.redis_cache = redis_cache
    handler._registered_redis_cache = redis_cache
    calls = 0

    async def fail_after_attempt(**_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise TimeoutError("reply timed out after Redis may have applied the script")

    handler.increment_script = fail_after_attempt
    with patch.object(handler, "_get_local_count", new=AsyncMock(side_effect=AssertionError("must fail closed"))):
        with pytest.raises(TimeoutError):
            await handler._increment_agent_and_get("uncertain-session", "agent-a")

    assert calls == 1
