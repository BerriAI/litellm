"""
Unit Tests for the per-session budget limiter for the proxy.

Tests that session-scoped budget tracking works correctly:
- Enforces max_budget_per_session per session_id (read from agent litellm_params)
- Different sessions have independent budgets
- Requests under budget pass through
- Requests without agent_id pass through
"""

import asyncio
import json
import logging
from typing import Final
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from litellm.caching.caching import DualCache
from litellm.caching.redis_cache import RedisCircuitBreakerOpenError, _redis_circuit_breaker_guard
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.agent_endpoints.agent_registry import AgentRegistry
from litellm.proxy.hooks.max_budget_per_session_limiter import (
    PROXY_MaxBudgetPerSessionHandler,
)
from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup
from litellm.proxy.utils import InternalUsageCache
from litellm.types.agents import AgentResponse


def _make_mock_agent(max_budget_per_session: float, agent_id: str = "agent-budget-123") -> AgentResponse:
    return AgentResponse(
        agent_id=agent_id,
        agent_name="budget-agent",
        litellm_params={"max_budget_per_session": max_budget_per_session},
        agent_card_params={"name": "budget-agent", "version": "1.0.0"},
    )


@pytest.mark.asyncio
async def test_budget_per_session_under_budget_passes():
    """
    Requests under budget should pass through without error.
    """
    local_cache = DualCache()
    handler = PROXY_MaxBudgetPerSessionHandler(
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
    handler = PROXY_MaxBudgetPerSessionHandler(
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
    handler = PROXY_MaxBudgetPerSessionHandler(
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
    handler = PROXY_MaxBudgetPerSessionHandler(
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
@pytest.mark.parametrize("response_cost", (0.125, 0.25))
async def test_concurrent_calls_are_admitted_before_their_costs_reach_the_stop_threshold(response_cost: float) -> None:
    cache: Final = DualCache()
    handler: Final = PROXY_MaxBudgetPerSessionHandler(InternalUsageCache(cache))
    agent: Final = _make_mock_agent(1.0)
    registry: Final = AgentRegistry()
    registry.register_agent(agent)
    key: Final = UserAPIKeyAuth(agent_id=agent.agent_id)
    session_id: Final = "concurrent-admission"
    data: Final = {"metadata": {"session_id": session_id}}
    metadata: Final = {"session_id": session_id, "agent_id": agent.agent_id}

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", registry):
        await handler.async_log_success_event(
            {"litellm_params": {"metadata": metadata}, "response_cost": 0.75}, None, None, None
        )
        admitted: Final = await asyncio.gather(
            handler.async_pre_call_hook(key, cache, data, ""),
            handler.async_pre_call_hook(key, cache, data, ""),
        )
        assert admitted == [None, None]
        assert await handler._get_agent_spend(session_id, agent.agent_id) == pytest.approx(0.75)

        await asyncio.gather(
            handler.async_log_success_event(
                {"litellm_params": {"metadata": metadata}, "response_cost": response_cost}, None, None, None
            ),
            handler.async_log_success_event(
                {"litellm_params": {"metadata": metadata}, "response_cost": response_cost}, None, None, None
            ),
        )
        assert await handler._get_agent_spend(session_id, agent.agent_id) == pytest.approx(0.75 + 2 * response_cost)
        with pytest.raises(HTTPException) as rejected:
            await handler.async_pre_call_hook(key, cache, data, "")
        assert rejected.value.status_code == 429


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "counter,invalid_value",
    tuple(
        (counter, invalid_value)
        for counter in ("legacy", "total_new", "agent")
        for invalid_value in ({}, "bad", " 0.5", "0.5 ", True, -1.0, float("nan"), float("inf"))
    )
    + (("total_new", None), ("agent", None)),
)
async def test_malformed_budget_ledger_is_rejected(counter: str, invalid_value: object) -> None:
    cache: Final = DualCache()
    handler: Final = PROXY_MaxBudgetPerSessionHandler(InternalUsageCache(cache))
    session_id: Final = "malformed-budget-ledger"
    agent_field: Final = handler._make_agent_scope_field("agent-a")
    legacy_value: Final = invalid_value if counter == "legacy" else 2.0
    await cache.async_set_cache(
        key=handler._make_legacy_cache_key(session_id),
        value=json.dumps(legacy_value) if isinstance(legacy_value, str) else legacy_value,
    )
    if counter != "legacy":
        await cache.async_set_cache(
            key=handler._make_agent_scope_cache_key(session_id),
            value={
                "__total_new": invalid_value if counter == "total_new" else 0.0,
                agent_field: invalid_value if counter == "agent" else 0.0,
            },
        )

    with pytest.raises((RuntimeError, ValueError)):
        await handler._get_agent_spend(session_id, "agent-a")
    with pytest.raises((RuntimeError, ValueError)):
        await handler._increment_agent_spend(session_id, "agent-a", 0.25)


@pytest.mark.asyncio
@pytest.mark.parametrize("counter", ("legacy", "total_new", "agent"))
async def test_overflowing_spend_increment_leaves_the_session_ledger_unchanged(counter: str) -> None:
    cache: Final = DualCache()
    handler: Final = PROXY_MaxBudgetPerSessionHandler(InternalUsageCache(cache))
    session_id: Final = "overflowing-spend-increment"
    legacy_key: Final = handler._make_legacy_cache_key(session_id)
    scope_key: Final = handler._make_agent_scope_cache_key(session_id)
    agent_field: Final = handler._make_agent_scope_field("agent-a")
    legacy_value: Final = 1e308 if counter == "legacy" else 0.0
    scope_value: Final = {
        "__total_new": 1e308 if counter == "total_new" else 0.0,
        agent_field: 1e308 if counter == "agent" else 0.0,
    }
    await cache.async_set_cache(key=legacy_key, value=legacy_value)
    await cache.async_set_cache(key=scope_key, value=scope_value)

    with pytest.raises(RuntimeError):
        await handler._increment_agent_spend(session_id, "agent-a", 1e308)

    assert await cache.async_get_cache(key=legacy_key) == legacy_value
    assert await cache.async_get_cache(key=scope_key) == scope_value


@pytest.mark.asyncio
@pytest.mark.parametrize("amount", (True, 0.0, -1.0, float("nan"), float("inf")))
async def test_invalid_spend_increment_leaves_the_session_ledger_unchanged(amount: float) -> None:
    cache: Final = DualCache()
    handler: Final = PROXY_MaxBudgetPerSessionHandler(InternalUsageCache(cache))
    session_id: Final = "invalid-spend-increment"
    assert await handler._increment_agent_spend(session_id, "agent-a", 0.5) == pytest.approx(0.5)

    with pytest.raises(RuntimeError):
        await handler._increment_agent_spend(session_id, "agent-a", amount)

    assert await handler._get_agent_spend(session_id, "agent-a") == pytest.approx(0.5)
    assert await handler._get_agent_spend(session_id, "agent-b") == 0.0


@pytest.mark.asyncio
async def test_invoking_another_agent_charges_the_authenticated_agents_session_budget() -> None:
    cache: Final = DualCache()
    handler: Final = PROXY_MaxBudgetPerSessionHandler(InternalUsageCache(cache))
    registry: Final = AgentRegistry()
    registry.register_agent(_make_mock_agent(1.0, "actor-agent"))
    registry.register_agent(_make_mock_agent(1.0, "target-agent"))
    key: Final = UserAPIKeyAuth(agent_id="actor-agent").model_copy(update={"invoked_agent_id": "target-agent"})
    data: Final = LiteLLMProxyRequestSetup.add_user_api_key_auth_to_request_metadata(
        {"metadata": {"session_id": "agent-delegation"}}, key, "metadata"
    )
    assert data["metadata"]["agent_id"] == "target-agent"
    assert data["metadata"]["billing_agent_id"] == "actor-agent"

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", registry):
        assert await handler.async_pre_call_hook(key, cache, data, "") is None
        await handler.async_log_success_event(
            {"litellm_params": {"metadata": data["metadata"]}, "response_cost": 1.0}, None, None, None
        )
        with pytest.raises(HTTPException) as rejected:
            await handler.async_pre_call_hook(key, cache, data, "")
        assert rejected.value.status_code == 429
        assert await handler._get_agent_spend("agent-delegation", "actor-agent") == pytest.approx(1.0)
        assert await handler._get_agent_spend("agent-delegation", "target-agent") == 0.0


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
    handler: Final = PROXY_MaxBudgetPerSessionHandler(InternalUsageCache(cache))
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
    handler: Final = PROXY_MaxBudgetPerSessionHandler(InternalUsageCache(cache))
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
    handler: Final = PROXY_MaxBudgetPerSessionHandler(InternalUsageCache(cache))
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
    handler: Final = PROXY_MaxBudgetPerSessionHandler(InternalUsageCache(cache))
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
    handler: Final = PROXY_MaxBudgetPerSessionHandler(InternalUsageCache(cache))
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
    handler: Final = PROXY_MaxBudgetPerSessionHandler(InternalUsageCache(cache))
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
    handler: Final = PROXY_MaxBudgetPerSessionHandler(InternalUsageCache(cache))
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
    handler: Final = PROXY_MaxBudgetPerSessionHandler(InternalUsageCache(cache))
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
    handler: Final = PROXY_MaxBudgetPerSessionHandler(InternalUsageCache(cache))
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
    handler: Final = PROXY_MaxBudgetPerSessionHandler(InternalUsageCache(cache))
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


class _OpenBreakerRedis:
    def __init__(self) -> None:
        from litellm.caching.redis_cache import RedisCircuitBreaker

        self._circuit_breaker = RedisCircuitBreaker(failure_threshold=3, recovery_timeout=60)
        for _ in range(3):
            self._circuit_breaker.record_failure()

    @_redis_circuit_breaker_guard
    async def async_get_cache(self, key, **kwargs):
        raise AssertionError("never reached")

    def async_register_script(self, script):
        @_redis_circuit_breaker_guard
        async def refused(_self, keys, args):
            raise AssertionError("never reached")

        return lambda keys, args: refused(self, keys, args)


@pytest.mark.asyncio
async def test_an_open_circuit_breaker_rejects_shared_spend_reads_without_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cache = DualCache(redis_cache=_OpenBreakerRedis())  # pyright: ignore[reportArgumentType]  # duck-typed Redis double
    handler = PROXY_MaxBudgetPerSessionHandler(internal_usage_cache=InternalUsageCache(cache))
    caplog.clear()

    with caplog.at_level(logging.DEBUG, logger="LiteLLM Proxy"):
        with pytest.raises(RedisCircuitBreakerOpenError):
            await handler._get_agent_spend("quiet", "agent-a")

    assert [record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING] == []
    assert any("circuit breaker is open" in record.getMessage() for record in caplog.records)
