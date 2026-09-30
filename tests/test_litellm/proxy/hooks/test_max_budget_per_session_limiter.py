"""
Unit Tests for the per-session budget limiter for the proxy.

Tests that session-scoped budget tracking works correctly:
- Enforces max_budget_per_session per session_id (read from agent litellm_params)
- Different sessions have independent budgets
- Requests under budget pass through
- Requests without agent_id pass through
"""

from unittest.mock import patch

import logging
import pytest
from fastapi import HTTPException

from litellm.caching.caching import DualCache
from litellm.caching.redis_cache import _redis_circuit_breaker_guard
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.max_budget_per_session_limiter import (
    _PROXY_MaxBudgetPerSessionHandler,
)
from litellm.proxy.utils import InternalUsageCache
from litellm.types.agents import AgentResponse


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
    handler = _PROXY_MaxBudgetPerSessionHandler(
        internal_usage_cache=InternalUsageCache(local_cache),
    )
    user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-test-key-budget",
        agent_id="agent-budget-123",
    )

    mock_agent = _make_mock_agent(max_budget_per_session=5.0)

    with patch(
        "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry"
    ) as mock_registry:
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
    handler = _PROXY_MaxBudgetPerSessionHandler(
        internal_usage_cache=InternalUsageCache(local_cache),
    )
    user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-test-key-budget",
        agent_id="agent-budget-123",
    )

    session_id = "session-over-budget"
    cache_key = handler._make_cache_key(session_id)
    await handler._increment_spend(cache_key, 1.50)

    mock_agent = _make_mock_agent(max_budget_per_session=1.0)

    with patch(
        "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry"
    ) as mock_registry:
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
    handler = _PROXY_MaxBudgetPerSessionHandler(
        internal_usage_cache=InternalUsageCache(local_cache),
    )
    user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-test-key-budget",
        agent_id="agent-budget-123",
    )

    cache_key_a = handler._make_cache_key("session-A")
    await handler._increment_spend(cache_key_a, 3.0)

    mock_agent = _make_mock_agent(max_budget_per_session=2.0)

    with patch(
        "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry"
    ) as mock_registry:
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
async def test_an_open_circuit_breaker_reads_session_spend_locally_without_a_warning(caplog):
    cache = DualCache(redis_cache=_OpenBreakerRedis())  # pyright: ignore[reportArgumentType]  # duck-typed Redis double
    handler = _PROXY_MaxBudgetPerSessionHandler(internal_usage_cache=InternalUsageCache(cache))
    caplog.clear()

    with caplog.at_level(logging.DEBUG, logger="LiteLLM Proxy"):
        spend = await handler._get_current_spend("{session_budget:quiet}:spend")

    assert spend == 0.0
    assert [record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING] == []
    assert any("circuit breaker is open" in record.getMessage() for record in caplog.records)
