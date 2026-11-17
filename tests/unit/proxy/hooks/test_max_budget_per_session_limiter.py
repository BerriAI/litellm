"""
Unit Tests for the per-session budget limiter for the proxy.

Tests that session-scoped budget tracking works correctly:
- Enforces max_budget_per_session per session_id (read from agent litellm_params)
- Different sessions have independent budgets
- Requests under budget pass through
- Requests without agent_id pass through
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final
from unittest.mock import patch

import logging
import pytest
from fastapi import HTTPException

from litellm.caching.caching import DualCache
from litellm.caching.redis_cache import _redis_circuit_breaker_guard
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.max_budget_per_session_limiter import (
    PROXY_MaxBudgetPerSessionHandler,
)
from litellm.proxy.utils import InternalUsageCache
from litellm.types.agents import AgentResponse
from litellm.types.videos.utils import POLLED_VIDEO_STATUS_KWARG


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
async def test_repeated_polls_of_a_finished_job_charge_the_session_once():
    from openai.types import Batch

    from litellm.types.videos.main import VideoObject

    handler = PROXY_MaxBudgetPerSessionHandler(internal_usage_cache=InternalUsageCache(DualCache()))
    video = VideoObject(id="job-1", object="video", status="completed")
    batch = Batch(
        id="job-1", completion_window="24h", created_at=1, endpoint="/v1/chat/completions",
        input_file_id="file-1", object="batch", status="completed", output_file_id="file-out",
    )
    polls = (("avideo_status", video, 0.4), ("aretrieve_batch", batch, 0.1), ("acompletion", None, 0.01))

    with patch(
        "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry"
    ) as mock_registry:
        mock_registry.get_agent_by_id.return_value = _make_mock_agent(max_budget_per_session=5.0)
        for call_type, response_obj, cost in polls * 3:
            await handler.async_log_success_event(
                {
                    "call_type": call_type,
                    "response_cost": cost,
                    POLLED_VIDEO_STATUS_KWARG: "completed",
                    "litellm_params": {"metadata": {"session_id": "session-jobs", "agent_id": "agent-budget-123"}},
                },
                response_obj,
                None,
                None,
            )

    spend = await handler._get_current_spend(handler._make_cache_key("session-jobs"))
    assert spend == pytest.approx(0.4 + 0.1 + 3 * 0.01)


@pytest.mark.asyncio
async def test_a_batch_retrieve_before_its_cost_is_final_charges_the_session_nothing():
    from openai.types import Batch

    handler: Final = PROXY_MaxBudgetPerSessionHandler(internal_usage_cache=InternalUsageCache(DualCache()))

    def polled(status: str, output_file_id: str | None = None) -> Batch:
        return Batch(
            id="batch-1", completion_window="24h", created_at=1, endpoint="/v1/chat/completions",
            input_file_id="file-1", object="batch", status=status, output_file_id=output_file_id,
        )

    with patch(
        "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry"
    ) as mock_registry:
        mock_registry.get_agent_by_id.return_value = _make_mock_agent(max_budget_per_session=5.0)
        for response_obj, cost in (
            (polled("in_progress"), 0.01),
            (polled("completed"), 0.01),
            (polled("completed", "file-out"), 0.25),
            (polled("completed", "file-out"), 0.25),
        ):
            await handler.async_log_success_event(
                {
                    "call_type": "aretrieve_batch",
                    "response_cost": cost,
                    "litellm_params": {"metadata": {"session_id": "session-batch", "agent_id": "agent-budget-123"}},
                },
                response_obj,
                None,
                None,
            )

    spend: Final = await handler._get_current_spend(handler._make_cache_key("session-batch"))
    assert spend == pytest.approx(0.25)


def _video_poll(
    video_cost: float, guardrail_cost: float = 0.0, session_id: str = "session-jobs", status: str = "completed"
) -> dict[str, object]:
    return {
        "call_type": "avideo_status",
        "response_cost": video_cost,
        POLLED_VIDEO_STATUS_KWARG: status,
        "standard_logging_object": {
            "response_cost": video_cost + guardrail_cost,
            "guardrail_information": [{"guardrail_name": "priced", "guardrail_cost": guardrail_cost}],
        },
        "litellm_params": {"metadata": {"session_id": session_id, "agent_id": "agent-budget-123"}},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("video_cost", "guardrail_cost"),
    [(0.40, 0.01), (0.05, 0.10)],
    ids=["video_costs_more_than_its_guardrail", "video_costs_no_more_than_its_guardrail"],
)
async def test_every_video_poll_pays_its_guardrail_and_only_the_finished_poll_pays_the_session_for_the_video(
    video_cost: float, guardrail_cost: float
):
    from litellm.types.videos.main import VideoObject

    handler = PROXY_MaxBudgetPerSessionHandler(internal_usage_cache=InternalUsageCache(DualCache()))
    completed_polls: Final = 3

    with patch(
        "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry"
    ) as mock_registry:
        mock_registry.get_agent_by_id.return_value = _make_mock_agent(max_budget_per_session=5.0)
        await handler.async_log_success_event(
            _video_poll(0.0, guardrail_cost=guardrail_cost, status="in_progress"),
            VideoObject(id="job-1", object="video", status="in_progress"),
            None,
            None,
        )
        for _ in range(completed_polls):
            await handler.async_log_success_event(
                _video_poll(video_cost, guardrail_cost=guardrail_cost),
                VideoObject(id="job-1", object="video", status="completed"),
                None,
                None,
            )

    spend = await handler._get_current_spend(handler._make_cache_key("session-jobs"))
    assert spend == pytest.approx(video_cost + (1 + completed_polls) * guardrail_cost)


class _ClockedRedis:
    """The Redis calls the session limiter makes, on a clock the test moves."""

    def __init__(self) -> None:
        self.now = 0.0
        self.entries: Mapping[str, tuple[float, float | None]] = MappingProxyType({})
        self.scripts_fail = False

    def _live(self, key: str) -> tuple[float, float | None] | None:
        entry = self.entries.get(key)
        if entry is None or (entry[1] is not None and entry[1] <= self.now):
            return None
        return entry

    def _store(self, key: str, value: float, expires_at: float | None) -> float:
        self.entries = MappingProxyType({**self.entries, key: (value, expires_at)})
        return value

    async def async_get_cache(self, key: str, **kwargs: object) -> float | None:
        entry = self._live(key)
        return None if entry is None else entry[0]

    async def async_increment(
        self, key: str, value: float, ttl: int | None = None, parent_otel_span: object = None, refresh_ttl: bool = False
    ) -> float:
        live = self._live(key)
        kept_expiry = None if live is None else live[1]
        expires_at = kept_expiry if ttl is None or (live is not None and not refresh_ttl) else self.now + ttl
        return self._store(key, value + (0.0 if live is None else live[0]), expires_at)

    def _increment_expiring_on_first_write(self, key: str, amount: float, ttl: float) -> float:
        live = self._live(key)
        expires_at = self.now + ttl if live is None else live[1]
        return self._store(key, amount + (0.0 if live is None else live[0]), expires_at)

    def async_register_script(self, script: str):
        async def run(keys: list[str], args: list[object]) -> float:
            if self.scripts_fail:
                raise ConnectionError("Redis dropped the script call")
            if len(keys) == 1:
                return self._increment_expiring_on_first_write(keys[0], float(str(args[0])), float(str(args[1])))
            marker = self._live(keys[1])
            polls = self._store(keys[1], 1.0 + (0.0 if marker is None else marker[0]), self.now + float(str(args[2])))
            amount = float(str(args[0] if polls == 1 else args[1]))
            if amount > 0:
                self._increment_expiring_on_first_write(keys[0], amount, float(str(args[2])))
            return amount

        return run


@pytest.mark.asyncio
async def test_a_finished_video_polled_without_pause_is_not_charged_again_once_the_session_window_rolls_over():
    from litellm.types.videos.main import VideoObject

    redis = _ClockedRedis()
    handler = PROXY_MaxBudgetPerSessionHandler(
        internal_usage_cache=InternalUsageCache(DualCache(redis_cache=redis))  # pyright: ignore[reportArgumentType]  # duck-typed Redis double
    )
    video = VideoObject(id="job-1", object="video", status="completed")
    chat = {
        "call_type": "acompletion",
        "response_cost": 0.01,
        "litellm_params": {"metadata": {"session_id": "session-jobs", "agent_id": "agent-budget-123"}},
    }

    with patch(
        "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry"
    ) as mock_registry:
        mock_registry.get_agent_by_id.return_value = _make_mock_agent(max_budget_per_session=5.0)
        for _ in range(3):
            await handler.async_log_success_event(_video_poll(0.4), video, None, None)
            redis.now += 0.6 * handler.ttl
        await handler.async_log_success_event(chat, None, None, None)

    assert await handler._get_current_spend(handler._make_cache_key("session-jobs")) == pytest.approx(0.01)


@pytest.mark.asyncio
async def test_a_finished_video_whose_charge_missed_redis_is_charged_on_the_next_poll_once_redis_answers():
    from litellm.types.videos.main import VideoObject

    redis: Final = _ClockedRedis()
    handler: Final = PROXY_MaxBudgetPerSessionHandler(
        internal_usage_cache=InternalUsageCache(DualCache(redis_cache=redis))  # pyright: ignore[reportArgumentType]  # duck-typed Redis double
    )
    video: Final = VideoObject(id="job-1", object="video", status="completed")

    with patch(
        "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry"
    ) as mock_registry:
        mock_registry.get_agent_by_id.return_value = _make_mock_agent(max_budget_per_session=5.0)
        redis.scripts_fail = True
        await handler.async_log_success_event(_video_poll(0.4), video, None, None)
        redis.scripts_fail = False
        for _ in range(2):
            await handler.async_log_success_event(_video_poll(0.4), video, None, None)

    assert await handler._get_current_spend(handler._make_cache_key("session-jobs")) == pytest.approx(0.4)


@pytest.mark.asyncio
@pytest.mark.parametrize("response_cost", [float("nan"), float("inf"), True])
async def test_a_cost_that_is_not_a_finite_number_leaves_the_session_spend_alone(response_cost):
    handler = PROXY_MaxBudgetPerSessionHandler(internal_usage_cache=InternalUsageCache(DualCache()))
    cache_key = handler._make_cache_key("session-unpriced")
    await handler._increment_spend(cache_key, 0.25)

    with patch(
        "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry"
    ) as mock_registry:
        mock_registry.get_agent_by_id.return_value = _make_mock_agent(max_budget_per_session=5.0)
        await handler.async_log_success_event(
            {
                "call_type": "acompletion",
                "response_cost": response_cost,
                "litellm_params": {"metadata": {"session_id": "session-unpriced", "agent_id": "agent-budget-123"}},
            },
            None,
            None,
            None,
        )

    assert await handler._get_current_spend(cache_key) == 0.25


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
        from litellm._service_logger import ServiceLogging
        from litellm.caching.redis_cache import RedisHealth
        from litellm.redis.store_settings import RedisBreakerSettings
        from litellm.rust_bridge._native import NativeRedisBreaker

        breaker: Final = NativeRedisBreaker(
            RedisBreakerSettings(failure_threshold=1, recovery_timeout=60.0, timeout_min_duration=5.0)
        )
        breaker.failed(False)
        self._health = RedisHealth(breaker, ServiceLogging())

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
