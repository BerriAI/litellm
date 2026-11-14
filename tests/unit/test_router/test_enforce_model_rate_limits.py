"""
Tests for enforce_model_rate_limits feature.

This feature allows users to enforce TPM/RPM limits set on model deployments
regardless of the routing strategy being used.
"""

import asyncio
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Final

import pytest

import litellm
from litellm import Router
from litellm.redis.stores.connection import connect
from litellm.redis.stores.router_usage import CounterIncrement, CounterKind, RouterUsage, RouterUsageStore, UsageCounter
from litellm.router_utils.pre_call_checks.model_rate_limit_check import (
    ModelRateLimitingCheck,
)
from litellm.utils import get_utc_datetime
from tests.unit.redis.stores.router_usage_double import FakeUsageHandle, usage_store

TPM_DEPLOYMENT = {
    "tpm": 1000,
    "litellm_params": {"model": "gpt-4"},
    "model_info": {"id": "replica-test-id"},
    "model_name": "test-model",
}


def _memory_usage() -> RouterUsageStore:
    return RouterUsageStore(connect(None).router_usage())


def _check(
    usage: RouterUsage | None = None, clock: Callable[[], datetime] = get_utc_datetime
) -> ModelRateLimitingCheck:
    return ModelRateLimitingCheck(usage=usage or _memory_usage(), clock=clock)


def _counter(deployment: dict, kind: CounterKind, window: str) -> UsageCounter:
    model_id: Final[object] = deployment["model_info"]["id"]
    model: Final[object] = deployment["litellm_params"]["model"]
    return UsageCounter.for_deployment(f"{model_id}", f"{model}", kind, window)


def _this_minute(deployment: dict, kind: CounterKind, value: float) -> dict[UsageCounter, float]:
    """``value`` under this minute's counter and the next, so a test crossing a minute still finds it."""
    now: Final = get_utc_datetime()
    return {_counter(deployment, kind, minute.strftime("%H-%M")): value for minute in (now, now + timedelta(minutes=1))}


RPM_DEPLOYMENT: Final = {
    "rpm": 10,
    "litellm_params": {"model": "gpt-4"},
    "model_info": {"id": "test-id"},
    "model_name": "test-model",
}


_SUCCESS_KWARGS: Final = {
    "standard_logging_object": {
        "model_id": "test-id",
        "total_tokens": 50,
        "hidden_params": {"litellm_model_name": "gpt-4"},
    }
}


def _tpm_increments(handle: FakeUsageHandle) -> tuple[tuple[object, object], ...]:
    """The (tokens, ttl) each success charged to test-id's gpt-4 tpm counter this minute or the next."""
    tpm_counters: Final = frozenset(_this_minute(RPM_DEPLOYMENT, "tpm", 0))
    return tuple((args[1], args[2]) for name, args in handle.calls if name == "increment" and args[0] in tpm_counters)


class TestModelRateLimitingCheck:
    """Test the ModelRateLimitingCheck class directly."""

    def test_get_deployment_limits_from_top_level(self):
        """Test extracting limits from top-level deployment config."""
        check = _check()

        deployment = {
            "tpm": 1000,
            "rpm": 10,
            "litellm_params": {"model": "gpt-4"},
            "model_info": {"id": "test-id"},
        }

        tpm, rpm = check._get_deployment_limits(deployment)
        assert tpm == 1000
        assert rpm == 10

    def test_get_deployment_limits_from_litellm_params(self):
        """Test extracting limits from litellm_params."""
        check = _check()

        deployment = {
            "litellm_params": {"model": "gpt-4", "tpm": 2000, "rpm": 20},
            "model_info": {"id": "test-id"},
        }

        tpm, rpm = check._get_deployment_limits(deployment)
        assert tpm == 2000
        assert rpm == 20

    def test_get_deployment_limits_from_model_info(self):
        """Test extracting limits from model_info."""
        check = _check()

        deployment = {
            "litellm_params": {"model": "gpt-4"},
            "model_info": {"id": "test-id", "tpm": 3000, "rpm": 30},
        }

        tpm, rpm = check._get_deployment_limits(deployment)
        assert tpm == 3000
        assert rpm == 30

    def test_get_deployment_limits_none_when_not_set(self):
        """Test that None is returned when limits are not set."""
        check = _check()

        deployment = {
            "litellm_params": {"model": "gpt-4"},
            "model_info": {"id": "test-id"},
        }

        tpm, rpm = check._get_deployment_limits(deployment)
        assert tpm is None
        assert rpm is None

    def test_pre_call_check_allows_request_when_no_limits(self):
        """Test that requests are allowed when no limits are set."""
        check = _check()

        deployment = {
            "litellm_params": {"model": "gpt-4"},
            "model_info": {"id": "test-id"},
        }

        result = check.pre_call_check(deployment)
        assert result == deployment

    def test_pre_call_check_raises_rate_limit_error_when_over_rpm(self):
        """Test that RateLimitError is raised when RPM limit is exceeded."""
        check = _check(usage_store(FakeUsageHandle(redis=_this_minute(RPM_DEPLOYMENT, "rpm", 10))))

        with pytest.raises(litellm.RateLimitError) as exc_info:
            check.pre_call_check(RPM_DEPLOYMENT)

        assert "RPM limit=10" in str(exc_info.value)
        assert "current usage=11" in str(exc_info.value)

    def test_pre_call_check_allows_request_under_limit(self):
        """Test that requests are allowed when under the limit."""
        check = _check(usage_store(FakeUsageHandle(redis=_this_minute(RPM_DEPLOYMENT, "rpm", 5))))

        assert check.pre_call_check(RPM_DEPLOYMENT) == RPM_DEPLOYMENT

    def test_pre_call_check_raises_rate_limit_error_when_over_tpm(self):
        """Test that RateLimitError is raised when TPM limit is exceeded."""
        check = _check(usage_store(FakeUsageHandle(local=_this_minute(TPM_DEPLOYMENT, "tpm", 1000))))

        with pytest.raises(litellm.RateLimitError) as exc_info:
            check.pre_call_check(TPM_DEPLOYMENT)

        assert "TPM limit=1000" in str(exc_info.value)
        assert "current usage=1000" in str(exc_info.value)

    def test_pre_call_check_rejects_when_shared_tpm_is_over_limit_but_local_is_under(self):
        handle = FakeUsageHandle(
            local=_this_minute(TPM_DEPLOYMENT, "tpm", 5), redis=_this_minute(TPM_DEPLOYMENT, "tpm", 1000)
        )
        check = _check(usage_store(handle))

        with pytest.raises(litellm.RateLimitError) as exc_info:
            check.pre_call_check(TPM_DEPLOYMENT)

        assert "current usage=1000" in str(exc_info.value)

    @pytest.mark.parametrize("failure", ["redis_down", "breaker_open"])
    def test_pre_call_check_keeps_rejecting_on_local_usage_when_redis_read_fails(self, failure: str):
        handle = FakeUsageHandle(
            local=_this_minute(TPM_DEPLOYMENT, "tpm", 1000),
            redis={},
            breaker_open=failure == "breaker_open",
            redis_down=failure == "redis_down",
        )
        check = _check(usage_store(handle))

        with pytest.raises(litellm.RateLimitError) as exc_info:
            check.pre_call_check(TPM_DEPLOYMENT)

        assert "current usage=1000" in str(exc_info.value)

    def test_pre_call_check_falls_back_to_local_tpm_and_still_checks_rpm_when_redis_circuit_is_open(self):
        handle = FakeUsageHandle(
            local={**_this_minute(TPM_DEPLOYMENT, "tpm", 5), **_this_minute(TPM_DEPLOYMENT, "rpm", 1)},
            redis={},
            breaker_open=True,
        )
        check = _check(usage_store(handle))

        with pytest.raises(litellm.RateLimitError) as exc_info:
            check.pre_call_check({**TPM_DEPLOYMENT, "rpm": 1})

        assert "RPM limit=1" in str(exc_info.value)

    def test_pre_call_check_without_redis_enforces_local_tpm_and_rpm(self):
        check = _check(usage_store(FakeUsageHandle(local=_this_minute(TPM_DEPLOYMENT, "tpm", 5))))
        deployment = {**TPM_DEPLOYMENT, "rpm": 1}

        assert check.pre_call_check(deployment) == deployment
        with pytest.raises(litellm.RateLimitError) as exc_info:
            check.pre_call_check(deployment)

        assert "RPM limit=1" in str(exc_info.value)

    def test_log_success_event_increments_cache(self):
        """Test that log_success_event correctly increments the cache."""
        handle = FakeUsageHandle()
        check = _check(usage_store(handle))

        check.log_success_event(_SUCCESS_KWARGS, None, None, None)

        assert _tpm_increments(handle) == ((50, 60),)


class TestModelRateLimitingCheckAsync:
    """Test async methods of ModelRateLimitingCheck."""

    @pytest.mark.asyncio
    async def test_async_pre_call_check_allows_request_when_no_limits(self):
        """Test that requests are allowed when no limits are set (async)."""
        check = _check()

        deployment = {
            "litellm_params": {"model": "gpt-4"},
            "model_info": {"id": "test-id"},
        }

        result = await check.async_pre_call_check(deployment)
        assert result == deployment

    @pytest.mark.asyncio
    async def test_async_pre_call_check_raises_rate_limit_error_when_over_rpm(self):
        """Test that RateLimitError is raised when RPM limit is exceeded (async)."""
        check = _check(usage_store(FakeUsageHandle(redis=_this_minute(RPM_DEPLOYMENT, "rpm", 10))))

        with pytest.raises(litellm.RateLimitError) as exc_info:
            await check.async_pre_call_check(RPM_DEPLOYMENT)

        assert "RPM limit=10" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_async_pre_call_check_allows_request_under_limit(self):
        """Test that requests are allowed when under the limit (async)."""
        check = _check(usage_store(FakeUsageHandle(redis=_this_minute(RPM_DEPLOYMENT, "rpm", 5))))

        assert await check.async_pre_call_check(RPM_DEPLOYMENT) == RPM_DEPLOYMENT

    @pytest.mark.asyncio
    async def test_async_pre_call_check_raises_rate_limit_error_when_over_tpm(self):
        """Test that RateLimitError is raised when TPM limit is exceeded (async)."""
        check = _check(usage_store(FakeUsageHandle(local=_this_minute(TPM_DEPLOYMENT, "tpm", 1000))))

        with pytest.raises(litellm.RateLimitError) as exc_info:
            await check.async_pre_call_check(TPM_DEPLOYMENT)

        assert "TPM limit=1000" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_async_pre_call_check_rejects_when_shared_tpm_is_over_limit_but_local_is_under(self):
        handle = FakeUsageHandle(
            local=_this_minute(TPM_DEPLOYMENT, "tpm", 5), redis=_this_minute(TPM_DEPLOYMENT, "tpm", 1000)
        )
        check = _check(usage_store(handle))

        with pytest.raises(litellm.RateLimitError) as exc_info:
            await check.async_pre_call_check(TPM_DEPLOYMENT)

        assert "current usage=1000" in str(exc_info.value)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("failure", ["redis_down", "breaker_open"])
    async def test_async_pre_call_check_keeps_rejecting_on_local_usage_when_redis_read_fails(self, failure: str):
        handle = FakeUsageHandle(
            local=_this_minute(TPM_DEPLOYMENT, "tpm", 1000),
            redis={},
            breaker_open=failure == "breaker_open",
            redis_down=failure == "redis_down",
        )
        check = _check(usage_store(handle))

        with pytest.raises(litellm.RateLimitError) as exc_info:
            await check.async_pre_call_check(TPM_DEPLOYMENT)

        assert "current usage=1000" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_async_pre_call_check_falls_back_to_local_tpm_and_still_checks_rpm_when_redis_circuit_is_open(
        self,
    ):
        handle = FakeUsageHandle(
            local={**_this_minute(TPM_DEPLOYMENT, "tpm", 5), **_this_minute(TPM_DEPLOYMENT, "rpm", 1)},
            redis={},
            breaker_open=True,
        )
        check = _check(usage_store(handle))

        with pytest.raises(litellm.RateLimitError) as exc_info:
            await check.async_pre_call_check({**TPM_DEPLOYMENT, "rpm": 1})

        assert "RPM limit=1" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_async_pre_call_check_without_redis_enforces_local_tpm_and_rpm(self):
        check = _check(usage_store(FakeUsageHandle(local=_this_minute(TPM_DEPLOYMENT, "tpm", 5))))
        deployment = {**TPM_DEPLOYMENT, "rpm": 1}

        assert await check.async_pre_call_check(deployment) == deployment
        with pytest.raises(litellm.RateLimitError) as exc_info:
            await check.async_pre_call_check(deployment)

        assert "RPM limit=1" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_async_log_success_event_increments_cache(self):
        """Test that async_log_success_event correctly increments the cache."""
        handle = FakeUsageHandle(redis={})
        check = _check(usage_store(handle))

        await check.async_log_success_event(_SUCCESS_KWARGS, None, None, None)

        assert _tpm_increments(handle) == ((50, 60),)


class TestRouterWithEnforceModelRateLimits:
    """Test Router integration with enforce_model_rate_limits."""

    def test_router_initializes_with_enforce_model_rate_limits(self):
        """Test that Router properly initializes the ModelRateLimitingCheck."""
        model_list = [
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-4", "api_key": "test"},
                "rpm": 10,
            }
        ]

        router = Router(
            model_list=model_list,
            optional_pre_call_checks=["enforce_model_rate_limits"],
        )

        # Check that the callback was added
        assert router.optional_callbacks is not None
        assert len(router.optional_callbacks) == 1
        assert isinstance(router.optional_callbacks[0], ModelRateLimitingCheck)

    def test_router_optional_callbacks_contains_model_rate_limiting(self):
        """Test that ModelRateLimitingCheck is in the callbacks list."""
        model_list = [
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-4", "api_key": "test"},
                "rpm": 10,
            }
        ]

        Router(
            model_list=model_list,
            optional_pre_call_checks=["enforce_model_rate_limits"],
        )

        # Find the ModelRateLimitingCheck in litellm.callbacks
        found = False
        for callback in litellm.callbacks:
            if isinstance(callback, ModelRateLimitingCheck):
                found = True
                break

        assert found, "ModelRateLimitingCheck should be in litellm.callbacks"


class TestModelRateLimitConcurrency:
    """Test that RPM rate limiting is atomic under concurrent requests."""

    @pytest.mark.asyncio
    async def test_concurrent_requests_respect_rpm_limit(self):
        """
        Fire 4 concurrent async requests with RPM limit of 2.
        Exactly 2 should succeed and 2 should raise RateLimitError.

        This test validates the atomic increment-first pattern:
        the old check-then-increment pattern would let 3+ through
        due to a race condition on the local cache read.
        """
        check = _check()

        deployment = {
            "rpm": 2,
            "litellm_params": {"model": "gpt-4"},
            "model_info": {"id": "concurrent-test-id"},
            "model_name": "test-model",
        }

        async def attempt_request():
            return await check.async_pre_call_check(deployment)

        results = await asyncio.gather(
            *[attempt_request() for _ in range(4)],
            return_exceptions=True,
        )

        successes = [r for r in results if not isinstance(r, Exception)]
        failures = [r for r in results if isinstance(r, litellm.RateLimitError)]

        assert len(successes) == 2, f"Expected 2 successes, got {len(successes)}"
        assert len(failures) == 2, f"Expected 2 rate limit errors, got {len(failures)}"


DAY_ONE = datetime(2026, 10, 28, 23, 59, 30, tzinfo=timezone.utc)
RPD_DEPLOYMENT = {
    "litellm_params": {"model": "gpt-4.1-mini", "rpd": 2},
    "model_info": {"id": "daily-test-id"},
    "model_name": "daily-model",
}
RPM_AND_RPD_DEPLOYMENT = {**RPD_DEPLOYMENT, "litellm_params": {"model": "gpt-4.1-mini", "rpm": 1, "rpd": 2}}
_DAY_ONE_RPM: Final = UsageCounter.for_deployment("daily-test-id", "gpt-4.1-mini", "rpm", "23-59")
_DAY_ONE_RPD: Final = UsageCounter.for_deployment("daily-test-id", "gpt-4.1-mini", "rpd", "2026-10-28")


class Clock:
    def __init__(self, now: datetime):
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def _pre_call(check: ModelRateLimitingCheck, deployment: dict[str, object]) -> object:
    return check.pre_call_check(deployment)


async def _async_pre_call(check: ModelRateLimitingCheck, deployment: dict[str, object]) -> object:
    return await check.async_pre_call_check(deployment)


async def _async_log_success(check: ModelRateLimitingCheck, kwargs: dict[str, object]) -> None:
    await check.async_log_success_event(kwargs, None, None, None)


class TestDailyLimits:
    def test_pre_call_check_rejects_the_request_over_rpd(self):
        check = _check(clock=Clock(DAY_ONE))

        assert _pre_call(check, RPD_DEPLOYMENT) == RPD_DEPLOYMENT
        assert _pre_call(check, RPD_DEPLOYMENT) == RPD_DEPLOYMENT
        with pytest.raises(litellm.RateLimitError) as exc_info:
            _pre_call(check, RPD_DEPLOYMENT)

        assert "RPD limit=2, current usage=3" in str(exc_info.value)
        assert "rpd limit=2. current usage=3. id=daily-test-id, model_group=daily-model" in exc_info.value.response.text

    @pytest.mark.asyncio
    async def test_async_pre_call_check_rejects_the_request_over_rpd_and_lets_the_router_retry(self):
        check = _check(clock=Clock(DAY_ONE))

        assert await _async_pre_call(check, RPD_DEPLOYMENT) == RPD_DEPLOYMENT
        assert await _async_pre_call(check, RPD_DEPLOYMENT) == RPD_DEPLOYMENT
        with pytest.raises(litellm.RateLimitError) as exc_info:
            await _async_pre_call(check, RPD_DEPLOYMENT)

        assert "RPD limit=2" in str(exc_info.value)
        assert exc_info.value.num_retries is None

    @pytest.mark.asyncio
    async def test_async_pre_call_check_rejects_once_the_day_tokens_reach_tpd(self):
        deployment = {**RPD_DEPLOYMENT, "litellm_params": {"model": "gpt-4.1-mini", "tpd": 100}}
        check = _check(clock=Clock(DAY_ONE))
        kwargs = {
            "standard_logging_object": {
                "model_id": "daily-test-id",
                "total_tokens": 60,
                "hidden_params": {"litellm_model_name": "gpt-4.1-mini"},
            }
        }

        await _async_log_success(check, kwargs)
        assert await _async_pre_call(check, deployment) == deployment
        await _async_log_success(check, kwargs)
        with pytest.raises(litellm.RateLimitError) as exc_info:
            await _async_pre_call(check, deployment)

        assert "TPD limit=100, current usage=120" in str(exc_info.value)

    def test_pre_call_check_minute_rejected_request_does_not_spend_rpd(self):
        clock = Clock(DAY_ONE)
        check = _check(clock=clock)

        assert _pre_call(check, RPM_AND_RPD_DEPLOYMENT) == RPM_AND_RPD_DEPLOYMENT
        with pytest.raises(litellm.RateLimitError, match="RPM limit=1"):
            _pre_call(check, RPM_AND_RPD_DEPLOYMENT)
        clock.now = DAY_ONE - timedelta(minutes=1)

        assert _pre_call(check, RPM_AND_RPD_DEPLOYMENT) == RPM_AND_RPD_DEPLOYMENT
        clock.now = DAY_ONE - timedelta(minutes=2)
        with pytest.raises(litellm.RateLimitError, match="RPD limit=2, current usage=3"):
            _pre_call(check, RPM_AND_RPD_DEPLOYMENT)

    @pytest.mark.asyncio
    async def test_async_pre_call_check_minute_rejected_request_does_not_spend_rpd(self):
        clock = Clock(DAY_ONE)
        check = _check(clock=clock)

        assert await _async_pre_call(check, RPM_AND_RPD_DEPLOYMENT) == RPM_AND_RPD_DEPLOYMENT
        with pytest.raises(litellm.RateLimitError, match="RPM limit=1"):
            await _async_pre_call(check, RPM_AND_RPD_DEPLOYMENT)
        clock.now = DAY_ONE - timedelta(minutes=1)

        assert await _async_pre_call(check, RPM_AND_RPD_DEPLOYMENT) == RPM_AND_RPD_DEPLOYMENT
        clock.now = DAY_ONE - timedelta(minutes=2)
        with pytest.raises(litellm.RateLimitError, match="RPD limit=2, current usage=3"):
            await _async_pre_call(check, RPM_AND_RPD_DEPLOYMENT)

    def test_pre_call_check_rejected_requests_hand_back_rpm_and_rpd(self):
        clock = Clock(DAY_ONE)
        check = _check(clock=clock)

        assert _pre_call(check, RPM_AND_RPD_DEPLOYMENT) == RPM_AND_RPD_DEPLOYMENT
        with pytest.raises(litellm.RateLimitError, match="RPM limit=1, current usage=2"):
            _pre_call(check, RPM_AND_RPD_DEPLOYMENT)
        with pytest.raises(litellm.RateLimitError, match="RPM limit=1, current usage=2"):
            _pre_call(check, RPM_AND_RPD_DEPLOYMENT)
        clock.now = DAY_ONE - timedelta(minutes=1)
        assert _pre_call(check, RPM_AND_RPD_DEPLOYMENT) == RPM_AND_RPD_DEPLOYMENT

        clock.now = DAY_ONE - timedelta(minutes=2)
        with pytest.raises(litellm.RateLimitError, match="RPD limit=2, current usage=3"):
            _pre_call(check, RPM_AND_RPD_DEPLOYMENT)
        with pytest.raises(litellm.RateLimitError, match="RPD limit=2, current usage=3"):
            _pre_call(check, RPM_AND_RPD_DEPLOYMENT)

    @pytest.mark.asyncio
    async def test_async_pre_call_check_rejected_requests_hand_back_rpm_and_rpd(self):
        clock = Clock(DAY_ONE)
        check = _check(clock=clock)

        assert await _async_pre_call(check, RPM_AND_RPD_DEPLOYMENT) == RPM_AND_RPD_DEPLOYMENT
        with pytest.raises(litellm.RateLimitError, match="RPM limit=1, current usage=2"):
            await _async_pre_call(check, RPM_AND_RPD_DEPLOYMENT)
        with pytest.raises(litellm.RateLimitError, match="RPM limit=1, current usage=2"):
            await _async_pre_call(check, RPM_AND_RPD_DEPLOYMENT)
        clock.now = DAY_ONE - timedelta(minutes=1)
        assert await _async_pre_call(check, RPM_AND_RPD_DEPLOYMENT) == RPM_AND_RPD_DEPLOYMENT

        clock.now = DAY_ONE - timedelta(minutes=2)
        with pytest.raises(litellm.RateLimitError, match="RPD limit=2, current usage=3"):
            await _async_pre_call(check, RPM_AND_RPD_DEPLOYMENT)
        with pytest.raises(litellm.RateLimitError, match="RPD limit=2, current usage=3"):
            await _async_pre_call(check, RPM_AND_RPD_DEPLOYMENT)

    @pytest.mark.asyncio
    async def test_async_pre_call_check_refunds_rpm_and_rpd_in_one_cache_trip(self):
        handle = FakeUsageHandle(redis={_DAY_ONE_RPD: 2})
        check = _check(usage_store(handle), clock=Clock(DAY_ONE))

        with pytest.raises(litellm.RateLimitError, match="RPD limit=2, current usage=3"):
            await _async_pre_call(check, RPM_AND_RPD_DEPLOYMENT)

        assert handle.calls == [
            ("pipeline", (CounterIncrement(_DAY_ONE_RPM, 1, 60), CounterIncrement(_DAY_ONE_RPD, 1, 86400))),
            ("pipeline", (CounterIncrement(_DAY_ONE_RPM, -1, 60), CounterIncrement(_DAY_ONE_RPD, -1, 86400))),
        ]

    @pytest.mark.asyncio
    async def test_async_pre_call_check_increments_rpm_and_rpd_in_one_cache_trip(self):
        handle = FakeUsageHandle(redis={})
        check = _check(usage_store(handle), clock=Clock(DAY_ONE))

        assert await _async_pre_call(check, RPM_AND_RPD_DEPLOYMENT) == RPM_AND_RPD_DEPLOYMENT

        assert handle.calls == [
            ("pipeline", (CounterIncrement(_DAY_ONE_RPM, 1, 60), CounterIncrement(_DAY_ONE_RPD, 1, 86400))),
        ]

    def test_pre_call_check_reads_the_shared_tpd_count_from_redis(self):
        tpd: Final = UsageCounter.for_deployment("daily-test-id", "gpt-4.1-mini", "tpd", "2026-10-28")
        handle = FakeUsageHandle(redis={tpd: 100})
        deployment = {**RPD_DEPLOYMENT, "litellm_params": {"model": "gpt-4.1-mini", "tpd": 100}}
        check = _check(usage_store(handle), clock=Clock(DAY_ONE))

        with pytest.raises(litellm.RateLimitError) as exc_info:
            _pre_call(check, deployment)

        assert "TPD limit=100, current usage=100" in str(exc_info.value)
        assert handle.calls[0] == ("current", (tpd, 100))

    def test_rpd_count_starts_fresh_after_utc_midnight(self):
        clock = Clock(DAY_ONE)
        check = _check(clock=clock)
        _pre_call(check, RPD_DEPLOYMENT)
        _pre_call(check, RPD_DEPLOYMENT)
        with pytest.raises(litellm.RateLimitError):
            _pre_call(check, RPD_DEPLOYMENT)

        clock.now = DAY_ONE + timedelta(seconds=30)

        assert _pre_call(check, RPD_DEPLOYMENT) == RPD_DEPLOYMENT
        assert _pre_call(check, RPD_DEPLOYMENT) == RPD_DEPLOYMENT
        with pytest.raises(litellm.RateLimitError) as exc_info:
            _pre_call(check, RPD_DEPLOYMENT)
        assert "current usage=3" in str(exc_info.value)

    @pytest.mark.parametrize(
        ("now", "seconds_to_midnight"),
        [
            (DAY_ONE, 30),
            (datetime(2026, 10, 28, 0, 0, 0, tzinfo=timezone.utc), 86400),
            (datetime(2026, 10, 28, 12, 30, 15, 500000, tzinfo=timezone.utc), 41384),
        ],
    )
    def test_daily_429_retry_after_is_seconds_until_utc_midnight(self, now: datetime, seconds_to_midnight: int):
        deployment = {**RPD_DEPLOYMENT, "litellm_params": {"model": "gpt-4.1-mini", "rpd": 0}}
        check = _check(clock=Clock(now))

        with pytest.raises(litellm.RateLimitError) as exc_info:
            _pre_call(check, deployment)

        assert exc_info.value.response.headers["retry-after"] == str(seconds_to_midnight)

    @pytest.mark.asyncio
    async def test_router_serves_the_request_from_the_other_deployment_once_rpd_is_spent(self):
        router = Router(
            model_list=[
                {
                    "model_name": "daily-group",
                    "litellm_params": {
                        "model": "openai/gpt-4.1-mini",
                        "api_key": "k",
                        "mock_response": "a",
                        "rpd": 1,
                        "order": 1,
                    },
                    "model_info": {"id": "spent-first"},
                },
                {
                    "model_name": "daily-group",
                    "litellm_params": {
                        "model": "openai/gpt-4.1-mini",
                        "api_key": "k",
                        "mock_response": "b",
                        "order": 2,
                    },
                    "model_info": {"id": "always-free"},
                },
            ],
            num_retries=1,
        )
        check = _check(clock=Clock(DAY_ONE))
        router.optional_callbacks = [check]
        litellm.logging_callback_manager.add_litellm_callback(check)
        try:
            served = [
                (await router.acompletion(model="daily-group", messages=[{"role": "user", "content": "hi"}]))
                .choices[0]
                .message.content
                for _ in range(4)
            ]
        finally:
            router.discard()

        assert served == ["a", "b", "b", "b"]


async def _tpd_tokens_once_logged(usage: RouterUsage, counter: UsageCounter) -> int:
    for _ in range(200):
        logged: Final = usage.get_local(counter)
        if logged is not None:
            return int(logged)
        await asyncio.sleep(0.01)
    pytest.fail(f"the success callback never logged tpd tokens under {counter}")


RESET_AT_EIGHT = datetime(2026, 10, 28, 8, 0, 0, tzinfo=timezone.utc)
DAILY_RESET_DEPLOYMENT = {
    **RPD_DEPLOYMENT,
    "litellm_params": {"model": "gpt-4.1-mini", "rpd": 2, "daily_reset_time": "08:00"},
}


class TestDailyResetTime:
    def test_rpd_window_starts_over_at_daily_reset_time_not_midnight(self):
        clock = Clock(RESET_AT_EIGHT - timedelta(hours=9))
        check = _check(clock=clock)

        assert _pre_call(check, DAILY_RESET_DEPLOYMENT) == DAILY_RESET_DEPLOYMENT
        clock.now = RESET_AT_EIGHT - timedelta(hours=7)
        assert _pre_call(check, DAILY_RESET_DEPLOYMENT) == DAILY_RESET_DEPLOYMENT
        clock.now = RESET_AT_EIGHT - timedelta(minutes=1)
        with pytest.raises(litellm.RateLimitError, match="RPD limit=2, current usage=3") as exc_info:
            _pre_call(check, DAILY_RESET_DEPLOYMENT)
        assert exc_info.value.response.headers["retry-after"] == "60"

        clock.now = RESET_AT_EIGHT
        assert _pre_call(check, DAILY_RESET_DEPLOYMENT) == DAILY_RESET_DEPLOYMENT

    @pytest.mark.asyncio
    async def test_async_rpd_window_starts_over_at_daily_reset_time_not_midnight(self):
        clock = Clock(RESET_AT_EIGHT - timedelta(hours=9))
        check = _check(clock=clock)

        assert await _async_pre_call(check, DAILY_RESET_DEPLOYMENT) == DAILY_RESET_DEPLOYMENT
        clock.now = RESET_AT_EIGHT - timedelta(hours=7)
        assert await _async_pre_call(check, DAILY_RESET_DEPLOYMENT) == DAILY_RESET_DEPLOYMENT
        clock.now = RESET_AT_EIGHT - timedelta(minutes=1)
        with pytest.raises(litellm.RateLimitError, match="RPD limit=2, current usage=3") as exc_info:
            await _async_pre_call(check, DAILY_RESET_DEPLOYMENT)
        assert exc_info.value.response.headers["retry-after"] == "60"

        clock.now = RESET_AT_EIGHT
        assert await _async_pre_call(check, DAILY_RESET_DEPLOYMENT) == DAILY_RESET_DEPLOYMENT

    @pytest.mark.parametrize(
        ("now", "seconds_to_reset"),
        [
            (RESET_AT_EIGHT + timedelta(seconds=1), 86399),
            (datetime(2026, 10, 28, 23, 59, 30, tzinfo=timezone.utc), 28830),
            (datetime(2026, 10, 29, 7, 30, 0, tzinfo=timezone.utc), 1800),
        ],
    )
    def test_daily_429_retry_after_is_seconds_until_daily_reset_time(self, now: datetime, seconds_to_reset: int):
        deployment = {
            **RPD_DEPLOYMENT,
            "litellm_params": {"model": "gpt-4.1-mini", "rpd": 0, "daily_reset_time": "08:00"},
        }
        check = _check(clock=Clock(now))

        with pytest.raises(litellm.RateLimitError) as exc_info:
            _pre_call(check, deployment)

        assert exc_info.value.response.headers["retry-after"] == str(seconds_to_reset)

    @pytest.mark.asyncio
    async def test_tokens_logged_before_daily_reset_time_do_not_count_after_it(self):
        deployment = {
            **RPD_DEPLOYMENT,
            "litellm_params": {"model": "gpt-4.1-mini", "tpd": 100, "daily_reset_time": "08:00"},
        }
        clock = Clock(RESET_AT_EIGHT - timedelta(minutes=1))
        check = _check(clock=clock)
        await _async_log_success(
            check,
            {
                "litellm_params": {"daily_reset_time": "08:00", "metadata": {}},
                "standard_logging_object": {
                    "model_id": "daily-test-id",
                    "total_tokens": 120,
                    "hidden_params": {"litellm_model_name": "gpt-4.1-mini"},
                },
            },
        )
        with pytest.raises(litellm.RateLimitError, match="TPD limit=100, current usage=120"):
            await _async_pre_call(check, deployment)

        clock.now = RESET_AT_EIGHT
        assert await _async_pre_call(check, deployment) == deployment

    @pytest.mark.parametrize(
        "deployment",
        [
            {"litellm_params": {"model": "gpt-4.1-mini"}, "model_info": {"id": "x", "rpd": 0}, "model_name": "m"},
            {"litellm_params": {"model": "gpt-4.1-mini"}, "model_info": {"id": "x"}, "model_name": "m", "rpd": 0},
        ],
    )
    def test_rpd_outside_litellm_params_is_not_a_limit(self, deployment: dict[str, object]):
        check = _check(clock=Clock(DAY_ONE))

        assert _pre_call(check, deployment) == deployment

    @pytest.mark.parametrize("daily_reset_time", ["8am", "08:00:30", "08:00+05:00", "25:00"])
    def test_router_rejects_a_daily_reset_time_that_is_not_hh_mm(self, daily_reset_time: str):
        with pytest.raises(Exception, match="daily_reset_time"):
            Router(
                model_list=[
                    {
                        "model_name": "daily-group",
                        "litellm_params": {"model": "gpt-4.1-mini", "rpd": 2, "daily_reset_time": daily_reset_time},
                    }
                ]
            )

    @pytest.mark.asyncio
    async def test_a_daily_reset_time_sent_in_the_request_does_not_move_the_tokens_out_of_the_deployment_window(self):
        router = Router(
            model_list=[
                {
                    "model_name": "daily-group",
                    "litellm_params": {
                        "model": "openai/gpt-4.1-mini",
                        "api_key": "k",
                        "mock_response": "a",
                        "tpd": 100,
                    },
                    "model_info": {"id": "daily-test-id"},
                }
            ]
        )
        usage: Final = _memory_usage()
        check = _check(usage, clock=Clock(RESET_AT_EIGHT - timedelta(minutes=1)))
        router.optional_callbacks = [check]
        litellm.logging_callback_manager.add_litellm_callback(check)
        try:
            await router.acompletion(  # pyright: ignore[reportUnknownMemberType]  # Router.acompletion kwargs are untyped
                model="daily-group", messages=[{"role": "user", "content": "hi"}], daily_reset_time="08:00"
            )
            logged = await _tpd_tokens_once_logged(
                usage, UsageCounter.for_deployment("daily-test-id", "openai/gpt-4.1-mini", "tpd", "2026-10-28")
            )
        finally:
            router.discard()

        assert logged > 0
        day_before: Final = usage.get_local(
            UsageCounter.for_deployment("daily-test-id", "openai/gpt-4.1-mini", "tpd", "2026-10-27")
        )
        assert day_before is None
