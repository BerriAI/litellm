"""
Unit Tests for the per-session budget limiter for the proxy.

Tests that session-scoped budget tracking works correctly:
- Enforces max_budget_per_session per session_id (read from agent litellm_params)
- Different sessions have independent budgets
- Requests under budget pass through
- Requests without agent_id pass through
"""

import asyncio
import contextlib
from datetime import datetime
from typing import Final
from unittest.mock import patch

import httpx
import logging
import pytest
import respx
from fastapi import HTTPException

import litellm
from litellm import Router

from litellm.caching.caching import DualCache
from litellm.caching.redis_cache import _redis_circuit_breaker_guard
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.llms.gemini.empty_image_response import ImageRefusalError
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.max_budget_per_session_limiter import (
    PROXY_MaxBudgetPerSessionHandler,
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
    handler = PROXY_MaxBudgetPerSessionHandler(
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
    handler = PROXY_MaxBudgetPerSessionHandler(
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
    handler = PROXY_MaxBudgetPerSessionHandler(
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
async def test_a_failure_the_provider_billed_uses_up_the_session_budget():
    handler = PROXY_MaxBudgetPerSessionHandler(internal_usage_cache=InternalUsageCache(DualCache()))
    session_metadata: Final = {"session_id": "session-refused-stream", "agent_id": "agent-budget-123"}

    with patch(
        "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry"
    ) as mock_registry:
        mock_registry.get_agent_by_id.return_value = _make_mock_agent(max_budget_per_session=1.0)
        await handler.async_log_failure_event(
            {"call_type": "anthropic_messages", "response_cost": 1.5, "litellm_params": {"metadata": session_metadata}},
            None,
            None,
            None,
        )
        with pytest.raises(HTTPException) as exc_info:
            await handler.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="sk-test-key-budget", agent_id="agent-budget-123"),
                cache=DualCache(),
                data={"metadata": {"session_id": "session-refused-stream"}},
                call_type="",
            )

    assert exc_info.value.status_code == 429


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
    handler = PROXY_MaxBudgetPerSessionHandler(internal_usage_cache=InternalUsageCache(cache))
    caplog.clear()

    with caplog.at_level(logging.DEBUG, logger="LiteLLM Proxy"):
        spend = await handler._get_current_spend("{session_budget:quiet}:spend")

    assert spend == 0.0
    assert [record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING] == []
    assert any("circuit breaker is open" in record.getMessage() for record in caplog.records)


_IMAGE_USAGE: Final = {"promptTokenCount": 777, "candidatesTokenCount": 333, "totalTokenCount": 1110}


async def _session_spend_after(
    router: Router, handler: PROXY_MaxBudgetPerSessionHandler, model: str, session_id: str
) -> float:
    metadata: Final = {"session_id": session_id, "agent_id": "agent-budget-123"}
    logging_obj, _ = litellm.utils.function_setup(
        original_function="aimage_generation",
        rules_obj=litellm.utils.Rules(),
        start_time=datetime.now(),
        model=model,
        prompt="draw a circle",
        metadata=metadata,
        litellm_call_id=session_id,
    )
    with contextlib.suppress(ImageRefusalError):
        await router.aimage_generation(
            model=model,
            prompt="draw a circle",
            metadata=metadata,
            litellm_logging_obj=logging_obj,
            litellm_call_id=session_id,
        )
    await asyncio.sleep(0)
    await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10)
    return await handler._get_current_spend(handler._make_cache_key(session_id))


@pytest.mark.asyncio
async def test_a_billed_image_refusal_counts_once_toward_the_session_budget(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler: Final = PROXY_MaxBudgetPerSessionHandler(internal_usage_cache=InternalUsageCache(DualCache()))
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "callbacks", [handler])
    monkeypatch.setattr(litellm, "_async_success_callback", [])
    monkeypatch.setattr(litellm, "_async_failure_callback", [])
    respx_mock.post(url__startswith="https://refuses.example/").mock(
        return_value=httpx.Response(
            200, json={"candidates": [{"finishReason": "IMAGE_PROHIBITED_CONTENT"}], "usageMetadata": _IMAGE_USAGE}
        )
    )
    respx_mock.post(url__startswith="https://rejects-the-key.example/").mock(
        return_value=httpx.Response(401, json={"error": {"code": 401, "message": "bad key"}})
    )
    respx_mock.post(url__startswith="https://backup.example/").mock(
        return_value=httpx.Response(
            200,
            json={
                "candidates": [
                    {"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": "aW1hZ2U="}}]}}
                ],
                "usageMetadata": _IMAGE_USAGE,
            },
        )
    )
    router: Final = Router(
        model_list=[
            {
                "model_name": group,
                "litellm_params": {
                    "model": "gemini/gemini-3.1-flash-image",
                    "api_key": "test-key",
                    "api_base": f"https://{host}.example",
                },
            }
            for group, host in (
                ("refuses", "refuses"),
                ("refuses-alone", "refuses"),
                ("rejects-the-key", "rejects-the-key"),
                ("backup", "backup"),
            )
        ],
        content_policy_fallbacks=[{"refuses": ["backup"]}],
        fallbacks=[{"rejects-the-key": ["backup"]}],
        num_retries=0,
    )

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry") as mock_registry:
        mock_registry.get_agent_by_id.return_value = _make_mock_agent(max_budget_per_session=100.0)
        refused_alone: Final = await _session_spend_after(router, handler, "refuses-alone", "session-refused")
        served_after_unbilled: Final = await _session_spend_after(router, handler, "rejects-the-key", "session-401")
        served_after_refused: Final = await _session_spend_after(router, handler, "refuses", "session-fallback")

    assert refused_alone > 0
    assert served_after_unbilled > 0
    assert served_after_refused == pytest.approx(refused_alone + served_after_unbilled)
