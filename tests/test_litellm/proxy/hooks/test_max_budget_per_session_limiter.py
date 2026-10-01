"""
Unit Tests for the per-session budget limiter for the proxy.

Tests that session-scoped budget tracking works correctly:
- Enforces max_budget_per_session per session_id (read from agent litellm_params)
- Different sessions have independent budgets
- Requests under budget pass through
- Requests without agent_id pass through
"""

import asyncio
import os
import socket
import uuid
from typing import Final
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.agent_endpoints.agent_registry import AgentRegistry
from litellm.proxy.hooks.max_budget_per_session_limiter import (
    _PROXY_MaxBudgetPerSessionHandler,
)
from litellm.proxy.utils import InternalUsageCache
from litellm.types.agents import AgentResponse


def _make_mock_agent(max_budget_per_session: float, agent_id: str = "agent-budget-123") -> AgentResponse:
    return AgentResponse(
        agent_id=agent_id,
        agent_name="budget-agent",
        litellm_params={"max_budget_per_session": max_budget_per_session},
        agent_card_params={"name": "budget-agent", "version": "1.0.0"},
    )


def _redis_port_for_migration_test() -> int | None:
    port = int(os.getenv("LITELLM_TEST_REDIS_PORT", "6379"))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return port if sock.connect_ex(("127.0.0.1", port)) == 0 else None


@pytest.mark.asyncio
async def test_budget_per_session_under_budget_passes():
    """
    Requests under budget should pass through without error.
    """
    local_cache = DualCache()
    handler = _PROXY_MaxBudgetPerSessionHandler(
        internal_usage_cache=InternalUsageCache(local_cache),
    )
    user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-test-key-budget",
        agent_id="agent-budget-123",
    )

    mock_agent = _make_mock_agent(max_budget_per_session=5.0)

    registry: Final = AgentRegistry()
    registry.register_agent(mock_agent)
    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", registry):
        result = await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data={"metadata": {"session_id": "session-budget-1"}},
            call_type="",
        )
        assert result is None


@pytest.mark.asyncio
async def test_budget_per_session_exceeds_budget():
    """
    After accumulating spend beyond max_budget_per_session, the next
    pre-call check should raise 429.
    """
    local_cache = DualCache()
    handler = _PROXY_MaxBudgetPerSessionHandler(
        internal_usage_cache=InternalUsageCache(local_cache),
    )
    user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-test-key-budget",
        agent_id="agent-budget-123",
    )

    session_id = "session-over-budget"
    await handler._increment_agent_spend(session_id, "agent-budget-123", 1.50)

    mock_agent = _make_mock_agent(max_budget_per_session=1.0)

    registry: Final = AgentRegistry()
    registry.register_agent(mock_agent)
    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", registry):
        with pytest.raises(HTTPException) as exc_info:
            await handler.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=local_cache,
                data={"metadata": {"session_id": session_id}},
                call_type="",
            )
        assert exc_info.value.status_code == 429
        assert "budget exceeded" in str(exc_info.value.detail).lower()


@pytest.mark.asyncio
async def test_budget_per_session_independent_sessions():
    """
    Different session_ids have independent budget counters.
    Exhausting session A does not affect session B.
    """
    local_cache = DualCache()
    handler = _PROXY_MaxBudgetPerSessionHandler(
        internal_usage_cache=InternalUsageCache(local_cache),
    )
    user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-test-key-budget",
        agent_id="agent-budget-123",
    )

    await handler._increment_agent_spend("session-A", "agent-budget-123", 3.0)

    mock_agent = _make_mock_agent(max_budget_per_session=2.0)

    registry: Final = AgentRegistry()
    registry.register_agent(mock_agent)
    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", registry):
        # Session A should be blocked
        with pytest.raises(HTTPException) as exc_info:
            await handler.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=local_cache,
                data={"metadata": {"session_id": "session-A"}},
                call_type="",
            )
        assert exc_info.value.status_code == 429

        # Session B should still pass
        result = await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data={"metadata": {"session_id": "session-B"}},
            call_type="",
        )
        assert result is None


@pytest.mark.asyncio
async def test_no_agent_id_passes():
    """
    When no agent_id is set on the key, all requests pass through.
    """
    local_cache = DualCache()
    handler = _PROXY_MaxBudgetPerSessionHandler(
        internal_usage_cache=InternalUsageCache(local_cache),
    )
    user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-test-key-no-agent",
    )

    result = await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=local_cache,
        data={"metadata": {"session_id": "any-session"}},
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
async def test_agent_session_budget_counters_do_not_mix_usage(
    researcher_id: str, orchestrator_id: str, researcher_session: str, orchestrator_session: str
) -> None:
    cache: Final = DualCache()
    handler: Final = _PROXY_MaxBudgetPerSessionHandler(InternalUsageCache(cache))
    registry: Final = AgentRegistry()
    registry.register_agent(_make_mock_agent(3.0, researcher_id))
    registry.register_agent(_make_mock_agent(1.0, orchestrator_id))
    researcher: Final = UserAPIKeyAuth(agent_id=researcher_id)
    orchestrator: Final = UserAPIKeyAuth(agent_id=orchestrator_id)
    researcher_data: Final = {"metadata": {"session_id": researcher_session}}
    orchestrator_data: Final = {"metadata": {"session_id": orchestrator_session}}

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", registry):
        await handler.async_log_success_event(
            {
                "litellm_params": {"metadata": {"session_id": researcher_session, "agent_id": researcher_id}},
                "response_cost": 2.0,
            },
            None,
            None,
            None,
        )
        assert await handler.async_pre_call_hook(researcher, cache, researcher_data, "") is None
        assert await handler.async_pre_call_hook(orchestrator, cache, orchestrator_data, "") is None

        await handler.async_log_success_event(
            {
                "litellm_params": {"metadata": {"session_id": orchestrator_session, "agent_id": orchestrator_id}},
                "response_cost": 1.0,
            },
            None,
            None,
            None,
        )
        with pytest.raises(HTTPException) as rejected:
            await handler.async_pre_call_hook(orchestrator, cache, orchestrator_data, "")
        assert rejected.value.status_code == 429
        assert "Current spend: $1.0000" in str(rejected.value.detail)
        assert await handler.async_pre_call_hook(researcher, cache, researcher_data, "") is None


@pytest.mark.asyncio
async def test_legacy_agent_id_shares_the_registered_agents_budget() -> None:
    cache: Final = DualCache()
    handler: Final = _PROXY_MaxBudgetPerSessionHandler(InternalUsageCache(cache))
    registry: Final = AgentRegistry()
    registry.load_agents_from_config(
        (
            {
                "agent_name": "configured-agent",
                "agent_card_params": {"name": "configured-agent", "version": "1"},
                "litellm_params": {"max_budget_per_session": 1.0},
            },
        )
    )
    legacy_id, agent_id = next(iter(registry.config_agent_legacy_ids.items()))

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", registry):
        await handler.async_log_success_event(
            {"litellm_params": {"metadata": {"session_id": "trace", "agent_id": legacy_id}}, "response_cost": 1.0},
            None,
            None,
            None,
        )
        for identity in (agent_id, legacy_id):
            with pytest.raises(HTTPException) as rejected:
                await handler.async_pre_call_hook(
                    UserAPIKeyAuth(agent_id=identity), cache, {"metadata": {"session_id": "trace"}}, ""
                )
            assert rejected.value.status_code == 429
            assert "Current spend: $1.0000" in str(rejected.value.detail)


@pytest.mark.asyncio
async def test_agent_budget_keeps_spend_from_an_existing_session() -> None:
    cache: Final = DualCache()
    handler: Final = _PROXY_MaxBudgetPerSessionHandler(InternalUsageCache(cache))
    session_id: Final = "existing-session"
    legacy_key: Final = handler._make_legacy_cache_key(session_id)
    await cache.async_set_cache(key=legacy_key, value=1.0)
    registry: Final = AgentRegistry()
    registry.register_agent(_make_mock_agent(1.0))

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", registry):
        with pytest.raises(HTTPException) as rejected:
            await handler.async_pre_call_hook(
                UserAPIKeyAuth(agent_id="agent-budget-123"),
                cache,
                {"metadata": {"session_id": session_id}},
                "",
            )

    assert rejected.value.status_code == 429
    assert "Current spend: $1.0000" in str(rejected.value.detail)


@pytest.mark.asyncio
async def test_agent_budget_carries_existing_spend_into_its_counter() -> None:
    cache: Final = DualCache()
    handler: Final = _PROXY_MaxBudgetPerSessionHandler(InternalUsageCache(cache))
    session_id: Final = "existing-session"
    await cache.async_set_cache(key=handler._make_legacy_cache_key(session_id), value=0.75)
    registry: Final = AgentRegistry()
    registry.register_agent(_make_mock_agent(1.0))
    kwargs: Final = {
        "litellm_params": {"metadata": {"session_id": session_id, "agent_id": "agent-budget-123"}},
        "response_cost": 0.1,
    }

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", registry):
        await handler.async_log_success_event(kwargs, None, None, None)

    spend: Final = await handler._get_agent_spend(session_id, "agent-budget-123")
    assert spend == pytest.approx(0.85)


@pytest.mark.asyncio
async def test_agent_budget_recovers_conservatively_after_scope_eviction() -> None:
    cache: Final = DualCache()
    handler: Final = _PROXY_MaxBudgetPerSessionHandler(InternalUsageCache(cache))
    session_id: Final = "scope-eviction-session"
    registry: Final = AgentRegistry()
    registry.register_agent(_make_mock_agent(10.0, "agent-a"))
    registry.register_agent(_make_mock_agent(10.0, "agent-b"))

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", registry):
        await handler._increment_agent_spend(session_id, "agent-a", 0.4)
        await cache.async_delete_cache(key=handler._make_agent_scope_cache_key(session_id))
        # Lost per-agent detail falls back to the aggregate, rather than resetting spend.
        assert await handler._get_agent_spend(session_id, "agent-b") == pytest.approx(0.4)
        assert await handler._increment_agent_spend(session_id, "agent-a", 0.1) == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_agent_budget_fails_closed_when_scope_loses_migration_total() -> None:
    cache: Final = DualCache()
    handler: Final = _PROXY_MaxBudgetPerSessionHandler(InternalUsageCache(cache))
    session_id: Final = "inconsistent-scope-session"
    registry: Final = AgentRegistry()
    registry.register_agent(_make_mock_agent(10.0, "agent-a"))
    scope_key: Final = handler._make_agent_scope_cache_key(session_id)
    await cache.async_set_cache(key=handler._make_legacy_cache_key(session_id), value=0.4)
    await cache.async_set_cache(key=scope_key, value={})

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", registry):
        with pytest.raises(RuntimeError, match="missing its migration total"):
            await handler._increment_agent_spend(session_id, "agent-a", 0.1)


@pytest.mark.asyncio
async def test_agent_budget_registers_redis_scripts_when_redis_is_attached_late() -> None:
    class FakeRedisCache:
        def __init__(self) -> None:
            self.scripts: list[str] = []
            self.calls: list[dict[str, object]] = []

        def async_register_script(self, source: str):
            self.scripts.append(source)

            async def call(**kwargs: object) -> object:
                self.calls.append(kwargs)
                return "0.1"

            return call

    cache: Final = DualCache()
    handler: Final = _PROXY_MaxBudgetPerSessionHandler(InternalUsageCache(cache))
    redis_cache: Final = FakeRedisCache()
    cache.redis_cache = redis_cache
    registry: Final = AgentRegistry()
    registry.register_agent(_make_mock_agent(10.0, "agent-a"))

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", registry):
        await handler._increment_agent_spend("late-redis", "agent-a", 0.1)
        assert await handler._get_agent_spend("late-redis", "agent-a") == pytest.approx(0.1)

    assert len(redis_cache.scripts) == 2
    assert len(redis_cache.calls) == 2
    assert all(len(call["keys"]) == 2 for call in redis_cache.calls)


@pytest.mark.asyncio
async def test_agent_budget_tracks_old_and_new_pods_without_mixing_new_agent_spend() -> None:
    cache: Final = DualCache()
    handler: Final = _PROXY_MaxBudgetPerSessionHandler(InternalUsageCache(cache))
    session_id: Final = "mixed-rollout-session"
    legacy_key: Final = handler._make_legacy_cache_key(session_id)
    await cache.async_set_cache(key=legacy_key, value=0.5)
    registry: Final = AgentRegistry()
    registry.register_agent(_make_mock_agent(3.0, "agent-a"))
    registry.register_agent(_make_mock_agent(3.0, "agent-b"))

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", registry):
        await handler._increment_agent_spend(session_id, "agent-a", 0.4)
        assert await handler._get_agent_spend(session_id, "agent-a") == pytest.approx(0.9)
        assert await handler._get_agent_spend(session_id, "agent-b") == pytest.approx(0.5)

        # An old pod still writes only the legacy aggregate.
        await cache.async_set_cache(key=legacy_key, value=1.1)
        await handler._increment_agent_spend(session_id, "agent-a", 0.1)
        await handler._increment_agent_spend(session_id, "agent-b", 0.1)

        assert await handler._get_agent_spend(session_id, "agent-a") == pytest.approx(1.2)
        assert await handler._get_agent_spend(session_id, "agent-b") == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_agent_spend_is_recorded_before_a_budget_is_configured() -> None:
    cache: Final = DualCache()
    handler: Final = _PROXY_MaxBudgetPerSessionHandler(InternalUsageCache(cache))
    session_id: Final = "unlimited-session"
    agent: Final = _make_mock_agent(1.0)
    agent.litellm_params = {}
    registry: Final = AgentRegistry()
    registry.register_agent(agent)

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", registry):
        await handler.async_log_success_event(
            {
                "litellm_params": {
                    "metadata": {"session_id": session_id, "agent_id": agent.agent_id},
                },
                "response_cost": 1.25,
            },
            None,
            None,
            None,
        )
        agent.litellm_params["max_budget_per_session"] = 1.0

        with pytest.raises(HTTPException) as rejected:
            await handler.async_pre_call_hook(
                UserAPIKeyAuth(agent_id=agent.agent_id),
                cache,
                {"metadata": {"session_id": session_id}},
                "",
            )

    assert rejected.value.status_code == 429
    assert "Current spend: $1.2500" in str(rejected.value.detail)


@pytest.mark.asyncio
async def test_agent_budget_redis_errors_are_not_retried_or_read_locally() -> None:
    cache: Final = DualCache()
    handler: Final = _PROXY_MaxBudgetPerSessionHandler(InternalUsageCache(cache))
    redis_cache: Final = object()
    cache.redis_cache = redis_cache
    handler._registered_redis_cache = redis_cache
    increment_calls = 0
    read_calls = 0

    async def fail_increment(**_kwargs: object) -> object:
        nonlocal increment_calls
        increment_calls += 1
        raise TimeoutError("reply timed out after Redis may have applied the script")

    async def fail_read(**_kwargs: object) -> object:
        nonlocal read_calls
        read_calls += 1
        raise TimeoutError("Redis read timed out")

    handler.increment_script = fail_increment
    handler.get_agent_spend_script = fail_read
    with patch.object(handler, "_get_local_spend", new=AsyncMock(side_effect=AssertionError("must fail closed"))):
        with pytest.raises(TimeoutError):
            await handler._increment_agent_spend("uncertain-session", "agent-a", 0.1)
        with pytest.raises(TimeoutError):
            await handler._get_agent_spend("uncertain-session", "agent-a")

    assert increment_calls == 1
    assert read_calls == 1


@pytest.mark.asyncio
@pytest.mark.skipif(_redis_port_for_migration_test() is None, reason="requires local Redis for Lua migration path")
async def test_redis_budget_migration_is_atomic_and_preserves_session_ttl() -> None:
    from litellm.caching.redis_cache import RedisCache

    port: Final = _redis_port_for_migration_test()
    assert port is not None
    redis: Final = RedisCache(host="127.0.0.1", port=port)
    redis_client: Final = redis.init_async_client()
    handler: Final = _PROXY_MaxBudgetPerSessionHandler(InternalUsageCache(DualCache(redis_cache=redis)))
    handler.ttl = 30
    registry: Final = AgentRegistry()
    registry.register_agent(_make_mock_agent(10.0, "agent-a"))
    registry.register_agent(_make_mock_agent(10.0, "agent-b"))
    session_id: Final = f"budget-migration-{uuid.uuid4().hex}"
    concurrent_session: Final = f"budget-concurrent-{uuid.uuid4().hex}"
    keys: Final = []

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", registry):
        try:
            legacy_key: Final = handler._make_legacy_cache_key(session_id)
            agent_scope_key: Final = handler._make_agent_scope_cache_key(session_id)
            keys.extend((legacy_key, agent_scope_key))
            await redis.async_set_cache(key=legacy_key, value=0.5, ttl=30)
            legacy_redis_key: Final = redis.check_and_fix_namespace(legacy_key)
            initial_ttl: Final = await redis_client.pttl(legacy_redis_key)

            await handler._increment_agent_spend(session_id, "agent-a", 0.4)
            await redis.async_increment(legacy_key, 0.2, ttl=30)  # Old pod updates only the aggregate.
            await handler._increment_agent_spend(session_id, "agent-a", 0.1)
            await handler._increment_agent_spend(session_id, "agent-b", 0.1)
            assert await handler._get_agent_spend(session_id, "agent-a") == pytest.approx(1.2)
            assert await handler._get_agent_spend(session_id, "agent-b") == pytest.approx(0.8)

            hash_tags: Final = {key[key.index("{") : key.index("}") + 1] for key in keys[:2]}
            assert len(hash_tags) == 1
            sidecar_ttl: Final = await redis_client.pttl(redis.check_and_fix_namespace(agent_scope_key))
            current_ttl: Final = await redis_client.pttl(legacy_redis_key)
            assert 0 <= sidecar_ttl <= current_ttl
            assert current_ttl <= initial_ttl
            await asyncio.sleep(0.25)
            before_increment_ttl: Final = await redis_client.pttl(legacy_redis_key)
            before_increment_sidecar_ttl: Final = await redis_client.pttl(
                redis.check_and_fix_namespace(agent_scope_key)
            )
            await handler._increment_agent_spend(session_id, "agent-a", 0.01)
            after_increment_ttl: Final = await redis_client.pttl(legacy_redis_key)
            after_increment_sidecar_ttl: Final = await redis_client.pttl(redis.check_and_fix_namespace(agent_scope_key))
            assert after_increment_ttl <= before_increment_ttl + 50
            assert after_increment_sidecar_ttl <= before_increment_sidecar_ttl + 50

            # Losing the hash evicts migration and agent counters together; the aggregate remains a safe baseline.
            await redis_client.delete(redis.check_and_fix_namespace(agent_scope_key))
            assert await handler._get_agent_spend(session_id, "agent-b") == pytest.approx(1.31)
            await handler._increment_agent_spend(session_id, "agent-a", 0.01)
            assert await handler._get_agent_spend(session_id, "agent-a") == pytest.approx(1.32)

            concurrent_legacy_key: Final = handler._make_legacy_cache_key(concurrent_session)
            concurrent_scope_key: Final = handler._make_agent_scope_cache_key(concurrent_session)
            keys.extend((concurrent_legacy_key, concurrent_scope_key))
            await redis.async_set_cache(key=concurrent_legacy_key, value=5.0, ttl=30)
            await asyncio.gather(
                *(handler._increment_agent_spend(concurrent_session, "agent-a", 0.01) for _ in range(100)),
                *(handler._increment_agent_spend(concurrent_session, "agent-b", 0.01) for _ in range(50)),
            )
            assert await handler._get_agent_spend(concurrent_session, "agent-a") == pytest.approx(6.0)
            assert await handler._get_agent_spend(concurrent_session, "agent-b") == pytest.approx(5.5)

            saved_total_new: Final = await redis_client.hget(
                redis.check_and_fix_namespace(agent_scope_key), "__total_new"
            )
            await redis_client.hdel(redis.check_and_fix_namespace(agent_scope_key), "__total_new")
            with pytest.raises(Exception, match="agent session scope is missing its migration total"):
                await handler._get_agent_spend(session_id, "agent-a")
            assert saved_total_new is not None
            await redis_client.hset(redis.check_and_fix_namespace(agent_scope_key), "__total_new", saved_total_new)

            await redis_client.pexpire(legacy_redis_key, 100)
            await asyncio.sleep(0.15)
            with pytest.raises(Exception, match="legacy session spend expired before agent scope"):
                await handler._get_agent_spend(session_id, "agent-a")
        finally:
            await redis_client.delete(*(redis.check_and_fix_namespace(key) for key in keys))
