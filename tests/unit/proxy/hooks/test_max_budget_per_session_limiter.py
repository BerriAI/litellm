"""
Unit Tests for the per-session budget limiter for the proxy.

Tests that session-scoped budget tracking works correctly:
- Enforces max_budget_per_session per session_id (read from agent litellm_params)
- Different sessions have independent budgets
- Requests under budget pass through
- Requests without agent_id pass through
"""

import logging
from unittest.mock import patch

import fakeredis
import pytest
from fastapi import HTTPException

from litellm.caching.caching import DualCache
from litellm.caching.redis_cache import RedisCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.max_budget_per_session_limiter import (
    PROXY_MaxBudgetPerSessionHandler,
)
from litellm.proxy.utils import InternalUsageCache
from litellm.types.agents import AgentResponse
from tests._support.fake_redis import fake_redis_cache


def _make_mock_agent(max_budget_per_session: float) -> AgentResponse:
    return AgentResponse(
        agent_id="agent-budget-123",
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

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry") as mock_registry:
        mock_registry.get_agent_by_id.return_value = mock_agent

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
    await handler._increment_spend(session_id, 1.50)

    mock_agent = _make_mock_agent(max_budget_per_session=1.0)

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry") as mock_registry:
        mock_registry.get_agent_by_id.return_value = mock_agent

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

    await handler._increment_spend("session-A", 3.0)

    mock_agent = _make_mock_agent(max_budget_per_session=2.0)

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry") as mock_registry:
        mock_registry.get_agent_by_id.return_value = mock_agent

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


def _redis_handler(server: fakeredis.FakeServer, namespace: str | None = None) -> tuple[RedisCache, object]:
    cache = fake_redis_cache(server, namespace=namespace)
    return cache, PROXY_MaxBudgetPerSessionHandler(
        internal_usage_cache=InternalUsageCache(DualCache(redis_cache=cache))
    )


async def _pre_call(handler, session_id: str, max_budget: float) -> None:
    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry") as mock_registry:
        mock_registry.get_agent_by_id.return_value = _make_mock_agent(max_budget_per_session=max_budget)
        await handler.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(api_key="sk-test-key-budget", agent_id="agent-budget-123"),
            cache=DualCache(),
            data={"metadata": {"session_id": session_id}},
            call_type="",
        )


@pytest.mark.asyncio
async def test_spend_written_through_redis_is_shared_with_old_pods_and_expires():
    server = fakeredis.FakeServer()
    cache, handler = _redis_handler(server, namespace="ns")
    raw = fakeredis.FakeAsyncRedis(server=server)

    assert await handler._increment_spend("s1", 1.25) == 1.25
    assert await handler._increment_spend("s1", 0.5) == 1.75

    assert await cache.async_get_cache(key="{session_budget:s1}:spend") == 1.75
    assert 3_590_000 < await raw.pttl("ns:{session_budget:s1}:spend") <= 3_600_000
    with pytest.raises(HTTPException):
        await _pre_call(handler, "s1", max_budget=1.5)


@pytest.mark.asyncio
async def test_spend_an_old_pod_wrote_blocks_the_session():
    server = fakeredis.FakeServer()
    _, handler = _redis_handler(server, namespace="ns")
    await fakeredis.FakeAsyncRedis(server=server).incrbyfloat("ns:{session_budget:s1}:spend", 2.5)

    assert await handler._get_current_spend("s1") == 2.5
    await _pre_call(handler, "s1", max_budget=3.0)
    with pytest.raises(HTTPException):
        await _pre_call(handler, "s1", max_budget=2.5)


@pytest.mark.asyncio
async def test_a_spend_that_does_not_decode_falls_back_to_local_spend():
    server = fakeredis.FakeServer()
    _, handler = _redis_handler(server)
    await handler._in_memory_increment_spend(handler._make_cache_key("s1"), 2.0)
    await fakeredis.FakeAsyncRedis(server=server).set("{session_budget:s1}:spend", '"not-a-number"')

    assert await handler._get_current_spend("s1") == 2.0


@pytest.mark.asyncio
async def test_an_unreachable_redis_counts_spend_locally_and_reads_it_as_zero():
    server = fakeredis.FakeServer()
    _, handler = _redis_handler(server)
    server.connected = False

    assert await handler._increment_spend("s1", 1.0) == 1.0
    assert await handler._increment_spend("s1", 2.0) == 3.0
    assert await handler._get_current_spend("s1") == 0.0
    await _pre_call(handler, "s1", max_budget=1.0)


@pytest.mark.asyncio
async def test_an_open_circuit_breaker_reads_session_spend_locally_without_a_warning(caplog):
    server = fakeredis.FakeServer()
    cache, handler = _redis_handler(server)
    await handler._in_memory_increment_spend(handler._make_cache_key("quiet"), 2.0)
    for _ in range(cache._circuit_breaker.failure_threshold):
        cache._circuit_breaker.record_failure()
    caplog.clear()

    with caplog.at_level(logging.DEBUG):
        spend = await handler._get_current_spend("quiet")

    assert spend == 2.0
    assert [record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING] == []
    assert any("circuit breaker is open" in record.getMessage() for record in caplog.records)
