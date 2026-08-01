"""
Unit Tests for the per-session budget limiter for the proxy.

Tests that session-scoped budget tracking works correctly:
- Enforces max_budget_per_session per session_id (read from agent litellm_params)
- Different sessions have independent budgets
- Requests under budget pass through
- Requests without agent_id pass through
"""

import asyncio
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from litellm.caching.caching import DualCache
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


def _make_handler() -> _PROXY_MaxBudgetPerSessionHandler:
    return _PROXY_MaxBudgetPerSessionHandler(internal_usage_cache=InternalUsageCache(DualCache()))


async def _pre_call(handler, session_id, reservation_cost, agent, user_api_key_dict):
    data = {"metadata": {"session_id": session_id}, "model": "gpt-4o"}
    with patch(
        "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry"
    ) as mock_registry, patch.object(
        handler, "_estimate_reservation_cost", return_value=reservation_cost
    ):
        mock_registry.get_agent_by_id.return_value = agent
        try:
            result = await handler.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=handler.internal_usage_cache.dual_cache,
                data=data,
                call_type="acompletion",
            )
        except HTTPException as e:
            result = e
    return result, data


@pytest.mark.asyncio
async def test_concurrent_requests_do_not_bypass_budget():
    """
    Regression for the admission race: several requests firing concurrently
    against a fresh session budget must not all slip past the gate. With
    per-request reservation only enough requests to fill the budget are
    admitted, and the session counter never exceeds max_budget_per_session.
    """
    handler = _make_handler()
    user_api_key_dict = UserAPIKeyAuth(api_key="sk-test-key-budget", agent_id="agent-budget-123")
    agent = _make_mock_agent(max_budget_per_session=1.0)
    session_id = "session-concurrent"

    results = await asyncio.gather(
        *[_pre_call(handler, session_id, 0.60, agent, user_api_key_dict) for _ in range(5)]
    )

    admitted = [r for r, _ in results if r is None]
    rejected = [r for r, _ in results if isinstance(r, HTTPException)]

    assert len(admitted) == 2
    assert len(rejected) == 3
    assert all(r.status_code == 429 for r in rejected)

    spend = await handler._get_current_spend(handler._make_cache_key(session_id))
    assert spend <= 1.0 + 1e-9


@pytest.mark.asyncio
async def test_first_request_admitted_when_estimate_exceeds_budget():
    """
    A single request whose worst-case estimate exceeds the whole budget is
    still admitted (reservation resized to the remaining budget), but pins the
    counter at the cap so a concurrent sibling is rejected.
    """
    handler = _make_handler()
    user_api_key_dict = UserAPIKeyAuth(api_key="sk-test-key-budget", agent_id="agent-budget-123")
    agent = _make_mock_agent(max_budget_per_session=1.0)
    session_id = "session-big-estimate"

    first, _ = await _pre_call(handler, session_id, 5.0, agent, user_api_key_dict)
    assert first is None

    spend = await handler._get_current_spend(handler._make_cache_key(session_id))
    assert spend == pytest.approx(1.0)

    second, _ = await _pre_call(handler, session_id, 5.0, agent, user_api_key_dict)
    assert isinstance(second, HTTPException)
    assert second.status_code == 429


@pytest.mark.asyncio
async def test_reservation_reconciled_to_actual_cost_on_success():
    """
    After admission the reservation is reconciled down to the actual response
    cost, so the reserved worst-case does not permanently inflate the session
    spend.
    """
    handler = _make_handler()
    user_api_key_dict = UserAPIKeyAuth(api_key="sk-test-key-budget", agent_id="agent-budget-123")
    agent = _make_mock_agent(max_budget_per_session=5.0)
    session_id = "session-reconcile"

    result, data = await _pre_call(handler, session_id, 0.80, agent, user_api_key_dict)
    assert result is None

    reserved_spend = await handler._get_current_spend(handler._make_cache_key(session_id))
    assert reserved_spend == pytest.approx(0.80)

    await handler.async_log_success_event(
        kwargs={"litellm_params": {"metadata": data["metadata"]}, "response_cost": 0.10},
        response_obj=None,
        start_time=None,
        end_time=None,
    )

    final_spend = await handler._get_current_spend(handler._make_cache_key(session_id))
    assert final_spend == pytest.approx(0.10)


@pytest.mark.asyncio
async def test_reservation_refunded_on_failure():
    """A failed call refunds its reservation so it doesn't consume budget."""
    handler = _make_handler()
    user_api_key_dict = UserAPIKeyAuth(api_key="sk-test-key-budget", agent_id="agent-budget-123")
    agent = _make_mock_agent(max_budget_per_session=5.0)
    session_id = "session-refund"

    result, data = await _pre_call(handler, session_id, 0.80, agent, user_api_key_dict)
    assert result is None
    assert await handler._get_current_spend(handler._make_cache_key(session_id)) == pytest.approx(0.80)

    await handler.async_log_failure_event(
        kwargs={"litellm_params": {"metadata": data["metadata"]}},
        response_obj=None,
        start_time=None,
        end_time=None,
    )

    assert await handler._get_current_spend(handler._make_cache_key(session_id)) == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_forged_reservation_metadata_is_ignored():
    """
    Reconciliation must only honor reservations this handler created. A caller
    that injects reservation keys into request metadata must not be able to
    apply a negative adjustment to (and drain the budget of) an arbitrary
    session.
    """
    handler = _make_handler()
    victim_session = "victim-session"
    victim_key = handler._make_cache_key(victim_session)

    await handler._increment_spend(victim_key, 5.0)
    assert await handler._get_current_spend(victim_key) == pytest.approx(5.0)

    forged_metadata = {
        "session_id": victim_session,
        "_litellm_session_budget_reservation": {
            "token": "attacker-guessed-token",
            "session_id": victim_session,
            "reserved_cost": 1000.0,
            "released": False,
        },
    }

    await handler.async_log_failure_event(
        kwargs={"litellm_params": {"metadata": forged_metadata}},
        response_obj=None,
        start_time=None,
        end_time=None,
    )
    await handler.async_post_call_failure_hook(
        request_data={"metadata": forged_metadata},
        original_exception=Exception("boom"),
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-attacker"),
    )

    assert await handler._get_current_spend(victim_key) == pytest.approx(5.0)


@pytest.mark.asyncio
async def test_reservation_refund_is_idempotent_across_failure_hooks():
    """
    The reservation must be refunded at most once even if both
    async_post_call_failure_hook and async_log_failure_event fire.
    """
    handler = _make_handler()
    user_api_key_dict = UserAPIKeyAuth(api_key="sk-test-key-budget", agent_id="agent-budget-123")
    agent = _make_mock_agent(max_budget_per_session=5.0)
    session_id = "session-idempotent"

    result, data = await _pre_call(handler, session_id, 0.80, agent, user_api_key_dict)
    assert result is None

    await handler.async_post_call_failure_hook(
        request_data=data,
        original_exception=Exception("boom"),
        user_api_key_dict=user_api_key_dict,
    )
    await handler.async_log_failure_event(
        kwargs={"litellm_params": {"metadata": data["metadata"]}},
        response_obj=None,
        start_time=None,
        end_time=None,
    )

    assert await handler._get_current_spend(handler._make_cache_key(session_id)) == pytest.approx(0.0)
