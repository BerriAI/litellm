"""
Tests for enforce_model_rate_limits feature.

This feature allows users to enforce TPM/RPM limits set on model deployments
regardless of the routing strategy being used.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm
from litellm import Router
from litellm.caching.dual_cache import DualCache
from litellm.caching.redis_cache import RedisCircuitBreakerOpenError
from litellm.router_utils.pre_call_checks.model_rate_limit_check import (
    ModelRateLimitingCheck,
)

TPM_DEPLOYMENT = {
    "tpm": 1000,
    "litellm_params": {"model": "gpt-4"},
    "model_info": {"id": "replica-test-id"},
    "model_name": "test-model",
}


def _dual_cache_with_local_tpm(local_tpm: int, redis_cache: MagicMock | None) -> DualCache:
    dual_cache = DualCache(redis_cache=redis_cache)
    check = ModelRateLimitingCheck(dual_cache=dual_cache)
    now = litellm.utils.get_utc_datetime()
    for minute in (now, now + timedelta(minutes=1)):
        tpm_key, _ = check._get_cache_keys(TPM_DEPLOYMENT, minute.strftime("%H-%M"))
        dual_cache.set_cache(key=tpm_key, value=local_tpm, local_only=True)
    return dual_cache


class TestModelRateLimitingCheck:
    """Test the ModelRateLimitingCheck class directly."""

    def test_get_deployment_limits_from_top_level(self):
        """Test extracting limits from top-level deployment config."""
        check = ModelRateLimitingCheck(dual_cache=MagicMock())

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
        check = ModelRateLimitingCheck(dual_cache=MagicMock())

        deployment = {
            "litellm_params": {"model": "gpt-4", "tpm": 2000, "rpm": 20},
            "model_info": {"id": "test-id"},
        }

        tpm, rpm = check._get_deployment_limits(deployment)
        assert tpm == 2000
        assert rpm == 20

    def test_get_deployment_limits_from_model_info(self):
        """Test extracting limits from model_info."""
        check = ModelRateLimitingCheck(dual_cache=MagicMock())

        deployment = {
            "litellm_params": {"model": "gpt-4"},
            "model_info": {"id": "test-id", "tpm": 3000, "rpm": 30},
        }

        tpm, rpm = check._get_deployment_limits(deployment)
        assert tpm == 3000
        assert rpm == 30

    def test_get_deployment_limits_none_when_not_set(self):
        """Test that None is returned when limits are not set."""
        check = ModelRateLimitingCheck(dual_cache=MagicMock())

        deployment = {
            "litellm_params": {"model": "gpt-4"},
            "model_info": {"id": "test-id"},
        }

        tpm, rpm = check._get_deployment_limits(deployment)
        assert tpm is None
        assert rpm is None

    def test_pre_call_check_allows_request_when_no_limits(self):
        """Test that requests are allowed when no limits are set."""
        check = ModelRateLimitingCheck(dual_cache=MagicMock())

        deployment = {
            "litellm_params": {"model": "gpt-4"},
            "model_info": {"id": "test-id"},
        }

        result = check.pre_call_check(deployment)
        assert result == deployment

    def test_pre_call_check_raises_rate_limit_error_when_over_rpm(self):
        """Test that RateLimitError is raised when RPM limit is exceeded."""
        mock_cache = MagicMock()
        mock_cache.increment_cache.return_value = 11  # Over limit after increment

        check = ModelRateLimitingCheck(dual_cache=mock_cache)

        deployment = {
            "rpm": 10,
            "litellm_params": {"model": "gpt-4"},
            "model_info": {"id": "test-id"},
            "model_name": "test-model",
        }

        with pytest.raises(litellm.RateLimitError) as exc_info:
            check.pre_call_check(deployment)

        assert "RPM limit=10" in str(exc_info.value)
        assert "current usage=11" in str(exc_info.value)

    def test_pre_call_check_allows_request_under_limit(self):
        """Test that requests are allowed when under the limit."""
        mock_cache = MagicMock()
        mock_cache.increment_cache.return_value = 6

        check = ModelRateLimitingCheck(dual_cache=mock_cache)

        deployment = {
            "rpm": 10,
            "litellm_params": {"model": "gpt-4"},
            "model_info": {"id": "test-id"},
            "model_name": "test-model",
        }

        result = check.pre_call_check(deployment)
        assert result == deployment

    def test_pre_call_check_raises_rate_limit_error_when_over_tpm(self):
        """Test that RateLimitError is raised when TPM limit is exceeded."""
        mock_cache = MagicMock()
        mock_cache.get_cache.return_value = 1000  # Already at limit

        check = ModelRateLimitingCheck(dual_cache=mock_cache)

        deployment = {
            "tpm": 1000,
            "litellm_params": {"model": "gpt-4"},
            "model_info": {"id": "test-id"},
            "model_name": "test-model",
        }

        with pytest.raises(litellm.RateLimitError) as exc_info:
            check.pre_call_check(deployment)

        assert "TPM limit=1000" in str(exc_info.value)
        assert "current usage=1000" in str(exc_info.value)

    def test_pre_call_check_rejects_when_shared_tpm_is_over_limit_but_local_is_under(self):
        redis_cache = MagicMock()
        redis_cache.get_cache.return_value = 1000
        check = ModelRateLimitingCheck(dual_cache=_dual_cache_with_local_tpm(5, redis_cache))

        with pytest.raises(litellm.RateLimitError) as exc_info:
            check.pre_call_check(TPM_DEPLOYMENT)

        assert "current usage=1000" in str(exc_info.value)

    @pytest.mark.parametrize(
        "redis_get", [MagicMock(return_value=None), MagicMock(side_effect=RedisCircuitBreakerOpenError())]
    )
    def test_pre_call_check_keeps_rejecting_on_local_usage_when_redis_read_fails(self, redis_get):
        redis_cache = MagicMock()
        redis_cache.get_cache = redis_get
        check = ModelRateLimitingCheck(dual_cache=_dual_cache_with_local_tpm(1000, redis_cache))

        with pytest.raises(litellm.RateLimitError) as exc_info:
            check.pre_call_check(TPM_DEPLOYMENT)

        assert "current usage=1000" in str(exc_info.value)

    def test_pre_call_check_falls_back_to_local_tpm_and_still_checks_rpm_when_redis_circuit_is_open(self):
        redis_cache = MagicMock()
        redis_cache.get_cache.side_effect = RedisCircuitBreakerOpenError()
        redis_cache.increment_cache.return_value = 2
        check = ModelRateLimitingCheck(dual_cache=_dual_cache_with_local_tpm(5, redis_cache))

        with pytest.raises(litellm.RateLimitError) as exc_info:
            check.pre_call_check({**TPM_DEPLOYMENT, "rpm": 1})

        assert "RPM limit=1" in str(exc_info.value)

    def test_pre_call_check_without_redis_enforces_local_tpm_and_rpm(self):
        check = ModelRateLimitingCheck(dual_cache=_dual_cache_with_local_tpm(5, None))
        deployment = {**TPM_DEPLOYMENT, "rpm": 1}

        assert check.pre_call_check(deployment) == deployment
        with pytest.raises(litellm.RateLimitError) as exc_info:
            check.pre_call_check(deployment)

        assert "RPM limit=1" in str(exc_info.value)

    def test_log_success_event_increments_cache(self):
        """Test that log_success_event correctly increments the cache."""
        mock_cache = MagicMock()
        check = ModelRateLimitingCheck(dual_cache=mock_cache)

        kwargs = {
            "standard_logging_object": {
                "model_id": "test-id",
                "total_tokens": 50,
                "hidden_params": {"litellm_model_name": "gpt-4"},
            }
        }

        check.log_success_event(kwargs, None, None, None)

        keys = tuple(call.kwargs["key"] for call in mock_cache.increment_cache.call_args_list)
        assert any(key.startswith("test-id:gpt-4:tpm:") for key in keys), keys
        assert all(call.kwargs["value"] == 50 for call in mock_cache.increment_cache.call_args_list)


class TestModelRateLimitingCheckAsync:
    """Test async methods of ModelRateLimitingCheck."""

    @pytest.mark.asyncio
    async def test_async_pre_call_check_allows_request_when_no_limits(self):
        """Test that requests are allowed when no limits are set (async)."""
        mock_cache = MagicMock()
        mock_cache.async_get_cache = AsyncMock(return_value=None)

        check = ModelRateLimitingCheck(dual_cache=mock_cache)

        deployment = {
            "litellm_params": {"model": "gpt-4"},
            "model_info": {"id": "test-id"},
        }

        result = await check.async_pre_call_check(deployment)
        assert result == deployment

    @pytest.mark.asyncio
    async def test_async_pre_call_check_raises_rate_limit_error_when_over_rpm(self):
        """Test that RateLimitError is raised when RPM limit is exceeded (async)."""
        mock_cache = MagicMock()
        mock_cache.async_get_cache = AsyncMock(return_value=None)
        mock_cache.async_increment_cache = AsyncMock(return_value=11)  # Over limit

        check = ModelRateLimitingCheck(dual_cache=mock_cache)

        deployment = {
            "rpm": 10,
            "litellm_params": {"model": "gpt-4"},
            "model_info": {"id": "test-id"},
            "model_name": "test-model",
        }

        with pytest.raises(litellm.RateLimitError) as exc_info:
            await check.async_pre_call_check(deployment)

        assert "RPM limit=10" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_async_pre_call_check_allows_request_under_limit(self):
        """Test that requests are allowed when under the limit (async)."""
        mock_cache = MagicMock()
        mock_cache.async_get_cache = AsyncMock(return_value=None)
        mock_cache.async_increment_cache = AsyncMock(return_value=6)

        check = ModelRateLimitingCheck(dual_cache=mock_cache)

        deployment = {
            "rpm": 10,
            "litellm_params": {"model": "gpt-4"},
            "model_info": {"id": "test-id"},
            "model_name": "test-model",
        }

        result = await check.async_pre_call_check(deployment)
        assert result == deployment

    @pytest.mark.asyncio
    async def test_async_pre_call_check_raises_rate_limit_error_when_over_tpm(self):
        """Test that RateLimitError is raised when TPM limit is exceeded (async)."""
        mock_cache = MagicMock()
        mock_cache.async_get_cache = AsyncMock(return_value=1000)  # Already at limit

        check = ModelRateLimitingCheck(dual_cache=mock_cache)

        deployment = {
            "tpm": 1000,
            "litellm_params": {"model": "gpt-4"},
            "model_info": {"id": "test-id"},
            "model_name": "test-model",
        }

        with pytest.raises(litellm.RateLimitError) as exc_info:
            await check.async_pre_call_check(deployment)

        assert "TPM limit=1000" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_async_pre_call_check_rejects_when_shared_tpm_is_over_limit_but_local_is_under(self):
        redis_cache = MagicMock()
        redis_cache.async_get_cache = AsyncMock(return_value=1000)
        check = ModelRateLimitingCheck(dual_cache=_dual_cache_with_local_tpm(5, redis_cache))

        with pytest.raises(litellm.RateLimitError) as exc_info:
            await check.async_pre_call_check(TPM_DEPLOYMENT)

        assert "current usage=1000" in str(exc_info.value)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "redis_get", [AsyncMock(return_value=None), AsyncMock(side_effect=RedisCircuitBreakerOpenError())]
    )
    async def test_async_pre_call_check_keeps_rejecting_on_local_usage_when_redis_read_fails(self, redis_get):
        redis_cache = MagicMock()
        redis_cache.async_get_cache = redis_get
        check = ModelRateLimitingCheck(dual_cache=_dual_cache_with_local_tpm(1000, redis_cache))

        with pytest.raises(litellm.RateLimitError) as exc_info:
            await check.async_pre_call_check(TPM_DEPLOYMENT)

        assert "current usage=1000" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_async_pre_call_check_falls_back_to_local_tpm_and_still_checks_rpm_when_redis_circuit_is_open(
        self,
    ):
        redis_cache = MagicMock()
        redis_cache.async_get_cache = AsyncMock(side_effect=RedisCircuitBreakerOpenError())
        redis_cache.async_increment = AsyncMock(return_value=2)
        check = ModelRateLimitingCheck(dual_cache=_dual_cache_with_local_tpm(5, redis_cache))

        with pytest.raises(litellm.RateLimitError) as exc_info:
            await check.async_pre_call_check({**TPM_DEPLOYMENT, "rpm": 1})

        assert "RPM limit=1" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_async_pre_call_check_without_redis_enforces_local_tpm_and_rpm(self):
        check = ModelRateLimitingCheck(dual_cache=_dual_cache_with_local_tpm(5, None))
        deployment = {**TPM_DEPLOYMENT, "rpm": 1}

        assert await check.async_pre_call_check(deployment) == deployment
        with pytest.raises(litellm.RateLimitError) as exc_info:
            await check.async_pre_call_check(deployment)

        assert "RPM limit=1" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_async_log_success_event_increments_cache(self):
        """Test that async_log_success_event correctly increments the cache."""
        mock_cache = MagicMock()
        mock_cache.async_increment_cache = AsyncMock()
        check = ModelRateLimitingCheck(dual_cache=mock_cache)

        kwargs = {
            "standard_logging_object": {
                "model_id": "test-id",
                "total_tokens": 50,
                "hidden_params": {"litellm_model_name": "gpt-4"},
            }
        }

        await check.async_log_success_event(kwargs, None, None, None)

        keys = tuple(call.kwargs["key"] for call in mock_cache.async_increment_cache.call_args_list)
        assert any(key.startswith("test-id:gpt-4:tpm:") for key in keys), keys
        assert all(call.kwargs["value"] == 50 for call in mock_cache.async_increment_cache.call_args_list)


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
        dual_cache = DualCache()
        check = ModelRateLimitingCheck(dual_cache=dual_cache)

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
        check = ModelRateLimitingCheck(dual_cache=DualCache(), clock=Clock(DAY_ONE))

        assert _pre_call(check, RPD_DEPLOYMENT) == RPD_DEPLOYMENT
        assert _pre_call(check, RPD_DEPLOYMENT) == RPD_DEPLOYMENT
        with pytest.raises(litellm.RateLimitError) as exc_info:
            _pre_call(check, RPD_DEPLOYMENT)

        assert "RPD limit=2, current usage=3" in str(exc_info.value)
        assert "rpd limit=2. current usage=3. id=daily-test-id, model_group=daily-model" in exc_info.value.response.text

    @pytest.mark.asyncio
    async def test_async_pre_call_check_rejects_the_request_over_rpd_and_lets_the_router_retry(self):
        check = ModelRateLimitingCheck(dual_cache=DualCache(), clock=Clock(DAY_ONE))

        assert await _async_pre_call(check, RPD_DEPLOYMENT) == RPD_DEPLOYMENT
        assert await _async_pre_call(check, RPD_DEPLOYMENT) == RPD_DEPLOYMENT
        with pytest.raises(litellm.RateLimitError) as exc_info:
            await _async_pre_call(check, RPD_DEPLOYMENT)

        assert "RPD limit=2" in str(exc_info.value)
        assert exc_info.value.num_retries is None

    @pytest.mark.asyncio
    async def test_async_pre_call_check_rejects_once_the_day_tokens_reach_tpd(self):
        deployment = {**RPD_DEPLOYMENT, "litellm_params": {"model": "gpt-4.1-mini", "tpd": 100}}
        check = ModelRateLimitingCheck(dual_cache=DualCache(), clock=Clock(DAY_ONE))
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

    def test_pre_call_check_reads_the_shared_tpd_count_from_redis(self):
        redis_cache = MagicMock()
        redis_cache.get_cache.return_value = 100
        deployment = {**RPD_DEPLOYMENT, "litellm_params": {"model": "gpt-4.1-mini", "tpd": 100}}
        check = ModelRateLimitingCheck(dual_cache=DualCache(redis_cache=redis_cache), clock=Clock(DAY_ONE))

        with pytest.raises(litellm.RateLimitError) as exc_info:
            _pre_call(check, deployment)

        assert "TPD limit=100, current usage=100" in str(exc_info.value)
        assert redis_cache.get_cache.call_args.kwargs["key"] == "daily-test-id:gpt-4.1-mini:tpd:2026-10-28"

    def test_rpd_count_starts_fresh_after_utc_midnight(self):
        clock = Clock(DAY_ONE)
        check = ModelRateLimitingCheck(dual_cache=DualCache(), clock=clock)
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
        check = ModelRateLimitingCheck(dual_cache=DualCache(), clock=Clock(now))

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
        check = ModelRateLimitingCheck(dual_cache=router.cache, clock=Clock(DAY_ONE))
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
