"""
Tests for enforce_model_rate_limits feature.

This feature allows users to enforce TPM/RPM limits set on model deployments
regardless of the routing strategy being used.
"""

import asyncio
from datetime import datetime, timezone
from functools import partial
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

import litellm
from litellm import Router
from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_cache import RedisCache, RedisCircuitBreaker
from litellm.router_utils.pre_call_checks.model_rate_limit_check import (
    ModelRateLimitingCheck,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("local_tpm", [None, 100])
@pytest.mark.parametrize("shared_tpm", [1000, 1100])
async def test_deployment_tpm_reads_shared_usage(use_async: bool, local_tpm: int | None, shared_tpm: int) -> None:
    shared_store: Final = InMemoryCache()
    redis: Final = MagicMock(spec=RedisCache)
    redis.get_cache.side_effect = shared_store.get_cache
    redis.async_get_cache = AsyncMock(side_effect=shared_store.async_get_cache)
    redis.increment_cache.side_effect = shared_store.increment_cache
    redis.async_increment = AsyncMock(side_effect=shared_store.async_increment)
    first_cache: Final = DualCache(redis_cache=redis)
    second_cache: Final = DualCache(redis_cache=redis)
    first_worker: Final = ModelRateLimitingCheck(first_cache)
    second_worker: Final = ModelRateLimitingCheck(second_cache)
    deployment: Final = {
        "model_name": "test-model",
        "tpm": 1000,
        "litellm_params": {"model": "gpt-4"},
        "model_info": {"id": "test-id"},
    }
    fixed_time: Final = datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)
    tpm_key: Final = "test-id:gpt-4:tpm:12-00"
    if local_tpm is not None:
        second_cache.in_memory_cache.set_cache(tpm_key, local_tpm)
    success_kwargs: Final = {
        "standard_logging_object": {
            "model_id": "test-id",
            "total_tokens": shared_tpm,
            "hidden_params": {"litellm_model_name": "gpt-4"},
        }
    }
    with patch(  # test-quality-ok: Freeze only the clock to keep callbacks and checks in the same minute
        "litellm.router_utils.pre_call_checks.model_rate_limit_check.get_utc_datetime",
        return_value=fixed_time,
    ):
        if use_async:
            assert await second_worker.async_pre_call_check(deployment) == deployment
            await first_worker.async_log_success_event(success_kwargs, None, None, None)
            with pytest.raises(litellm.RateLimitError, match="TPM limit=1000"):
                await second_worker.async_pre_call_check(deployment)
        else:
            assert second_worker.pre_call_check(deployment) == deployment
            first_worker.log_success_event(success_kwargs, None, None, None)
            with pytest.raises(litellm.RateLimitError, match="TPM limit=1000"):
                second_worker.pre_call_check(deployment)
    assert shared_store.get_cache(tpm_key) == shared_tpm


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("shared_tpm", [None, 999])
async def test_deployment_tpm_rejects_local_limit_without_redis_read(use_async: bool, shared_tpm: int | None) -> None:
    redis: Final = MagicMock(spec=RedisCache)
    redis.get_cache.return_value = shared_tpm
    redis.async_get_cache = AsyncMock(return_value=shared_tpm)
    cache: Final = DualCache(redis_cache=redis)
    cache.in_memory_cache.set_cache("test-id:gpt-4:tpm:12-00", 1000)
    check: Final = ModelRateLimitingCheck(cache)
    deployment: Final = {
        "model_name": "test-model",
        "tpm": 1000,
        "litellm_params": {"model": "gpt-4"},
        "model_info": {"id": "test-id"},
    }
    with patch(  # test-quality-ok: Freeze only the clock to read the seeded minute's counter deterministically
        "litellm.router_utils.pre_call_checks.model_rate_limit_check.get_utc_datetime",
        return_value=datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc),
    ):
        if use_async:
            with pytest.raises(litellm.RateLimitError, match="TPM limit=1000"):
                await check.async_pre_call_check(deployment)
        else:
            with pytest.raises(litellm.RateLimitError, match="TPM limit=1000"):
                check.pre_call_check(deployment)
    redis.get_cache.assert_not_called()
    redis.async_get_cache.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("local_tpm", [None, 100, 1000])
@pytest.mark.parametrize("read_error", [None, ConnectionError, RedisConnectionError, RedisTimeoutError])
async def test_deployment_tpm_redis_failure_keeps_local_enforcement(
    use_async: bool, local_tpm: int | None, read_error: type[Exception] | None
) -> None:
    redis: Final = MagicMock(spec=RedisCache)
    redis.get_cache.return_value = None
    redis.async_get_cache = AsyncMock(return_value=None)
    if read_error is not None:
        redis.get_cache.side_effect = read_error("Redis unavailable")
        redis.async_get_cache.side_effect = read_error("Redis unavailable")
    cache: Final = DualCache(redis_cache=redis)
    if local_tpm is not None:
        cache.in_memory_cache.set_cache("test-id:gpt-4:tpm:12-00", local_tpm)
    check: Final = ModelRateLimitingCheck(cache)
    deployment: Final = {
        "model_name": "test-model",
        "tpm": 1000,
        "litellm_params": {"model": "gpt-4"},
        "model_info": {"id": "test-id"},
    }
    with patch(  # test-quality-ok: Freeze only the clock to read the seeded minute's counter deterministically
        "litellm.router_utils.pre_call_checks.model_rate_limit_check.get_utc_datetime",
        return_value=datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc),
    ):
        if use_async and local_tpm == 1000:
            with pytest.raises(litellm.RateLimitError, match="TPM limit=1000"):
                await check.async_pre_call_check(deployment)
        elif local_tpm == 1000:
            with pytest.raises(litellm.RateLimitError, match="TPM limit=1000"):
                check.pre_call_check(deployment)
        elif use_async:
            assert await check.async_pre_call_check(deployment) == deployment
        else:
            assert check.pre_call_check(deployment) == deployment


@pytest.mark.asyncio
@pytest.mark.parametrize("local_tpm", [None, 100])
async def test_open_redis_circuit_preserves_async_rpm_enforcement(local_tpm: int | None) -> None:
    breaker: Final = RedisCircuitBreaker(failure_threshold=1, recovery_timeout=60, enabled=True)
    breaker.record_failure()
    redis: Final = MagicMock(spec=RedisCache)
    redis._circuit_breaker = breaker
    redis.async_get_cache = partial(RedisCache.async_get_cache, redis)
    redis.async_increment = partial(RedisCache.async_increment, redis)
    cache: Final = DualCache(redis_cache=redis)
    if local_tpm is not None:
        cache.in_memory_cache.set_cache("test-id:gpt-4:tpm:12-00", local_tpm)
    check: Final = ModelRateLimitingCheck(cache)
    deployment: Final = {
        "model_name": "test-model",
        "tpm": 1000,
        "rpm": 1,
        "litellm_params": {"model": "gpt-4"},
        "model_info": {"id": "test-id"},
    }
    with patch(  # test-quality-ok: Freeze only the clock to keep both requests in the same rate-limit window
        "litellm.router_utils.pre_call_checks.model_rate_limit_check.get_utc_datetime",
        return_value=datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc),
    ):
        with pytest.raises(Exception, match="Redis circuit breaker is open"):
            await redis.async_get_cache(key="test-id:gpt-4:tpm:12-00")
        assert await check.async_pre_call_check(deployment) == deployment
        with pytest.raises(litellm.RateLimitError, match="RPM limit=1"):
            await check.async_pre_call_check(deployment)
    assert cache.in_memory_cache.get_cache("test-id:gpt-4:rpm:12-00") == 2


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
        mock_cache.redis_cache = None

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

        # Verify increment_cache was called
        mock_cache.increment_cache.assert_called_once()
        _, kwarg_params = mock_cache.increment_cache.call_args
        assert "test-id:gpt-4:tpm:" in kwarg_params["key"]
        assert kwarg_params["value"] == 50


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
        mock_cache.redis_cache = None

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

        # Verify async_increment_cache was called
        mock_cache.async_increment_cache.assert_called_once()
        _, kwarg_params = mock_cache.async_increment_cache.call_args
        assert "test-id:gpt-4:tpm:" in kwarg_params["key"]
        assert kwarg_params["value"] == 50


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
