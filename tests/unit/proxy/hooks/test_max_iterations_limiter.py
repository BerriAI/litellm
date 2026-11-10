"""
Unit Tests for the max iterations limiter for the proxy.

Tests that session-scoped iteration counting works correctly:
- Enforces max_iterations per session_id (read from agent litellm_params)
- Different sessions have independent counters
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.max_iterations_limiter import PROXY_MaxIterationsHandler
from litellm.proxy.utils import InternalUsageCache
from litellm.types.agents import AgentResponse


def _make_mock_agent(max_iterations: int) -> AgentResponse:
    return AgentResponse(
        agent_id="agent-test-123",
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

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry") as mock_registry:
        mock_registry.get_agent_by_id.return_value = mock_agent

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

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry") as mock_registry:
        mock_registry.get_agent_by_id.return_value = mock_agent

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


class _SharedCounterRedis:
    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.script_args: list[list[int]] = []

    def async_register_script(self, script: str):
        async def increment(keys: list[str], args: list[int]) -> int:
            self.script_args.append(args)
            self.counts[keys[0]] = self.counts.get(keys[0], 0) + 1
            return self.counts[keys[0]]

        return increment


@pytest.mark.asyncio
async def test_max_iterations_counts_one_session_across_proxy_instances_through_redis():
    shared_redis = _SharedCounterRedis()
    first, second = (
        PROXY_MaxIterationsHandler(internal_usage_cache=InternalUsageCache(DualCache(redis_cache=shared_redis)))
        for _ in range(2)
    )
    user_api_key_dict = UserAPIKeyAuth(api_key="sk-test-key-1234", metadata={"max_iterations": 3})
    data = {"metadata": {"session_id": "session-shared"}}

    for handler in (first, second, first):
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict, cache=DualCache(), data=data, call_type=""
        )

    with pytest.raises(HTTPException) as exc_info:
        await second.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict, cache=DualCache(), data=data, call_type=""
        )

    assert exc_info.value.status_code == 429
    assert "Current count: 4" in str(exc_info.value.detail)
    assert shared_redis.script_args == [[first.ttl]] * 4
