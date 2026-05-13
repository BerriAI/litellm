import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy.health_check_utils.shared_health_check_manager import (
    SharedHealthCheckManager,
)


class TestSharedHealthCheckManager:
    """Test cases for SharedHealthCheckManager"""

    @pytest.fixture
    def mock_redis_cache(self):
        """Mock Redis cache for testing"""
        cache = AsyncMock()
        cache.async_set_cache = AsyncMock()
        cache.async_get_cache = AsyncMock()
        cache.async_delete_cache = AsyncMock()
        return cache

    @pytest.fixture
    def shared_health_manager(self, mock_redis_cache):
        """Create SharedHealthCheckManager instance for testing"""
        return SharedHealthCheckManager(
            redis_cache=mock_redis_cache,
            health_check_ttl=300,
            lock_ttl=60,
        )

    def test_initialization(self, mock_redis_cache):
        """Test SharedHealthCheckManager initialization"""
        manager = SharedHealthCheckManager(
            redis_cache=mock_redis_cache,
            health_check_ttl=300,
            lock_ttl=60,
        )

        assert manager.redis_cache == mock_redis_cache
        assert manager.health_check_ttl == 300
        assert manager.lock_ttl == 60
        assert manager.pod_id.startswith("pod_")

    def test_initialization_without_redis(self):
        """Test SharedHealthCheckManager initialization without Redis"""
        manager = SharedHealthCheckManager(redis_cache=None)

        assert manager.redis_cache is None
        assert manager.health_check_ttl == 300  # Default value
        assert manager.lock_ttl == 60  # Default value

    def test_get_health_check_lock_key(self):
        """Test getting health check lock key"""
        key = SharedHealthCheckManager.get_health_check_lock_key()
        assert key == "health_check_lock"

    def test_get_health_check_cache_key(self):
        """Test getting health check cache key"""
        key = SharedHealthCheckManager.get_health_check_cache_key()
        assert key == "health_check_results"

    def test_get_model_health_check_lock_key(self):
        """Test getting model-specific health check lock key"""
        key = SharedHealthCheckManager.get_model_health_check_lock_key("test-model")
        assert key == "health_check_lock:test-model"

    def test_get_model_health_check_cache_key(self):
        """Test getting model-specific health check cache key"""
        key = SharedHealthCheckManager.get_model_health_check_cache_key("test-model")
        assert key == "health_check_results:test-model"

    @pytest.mark.asyncio
    async def test_acquire_health_check_lock_success(
        self, shared_health_manager, mock_redis_cache
    ):
        """Test successful lock acquisition"""
        mock_redis_cache.async_set_cache.return_value = True

        result = await shared_health_manager.acquire_health_check_lock()

        assert result is True
        mock_redis_cache.async_set_cache.assert_called_once_with(
            "health_check_lock",
            shared_health_manager.pod_id,
            nx=True,
            ttl=60,
        )

    @pytest.mark.asyncio
    async def test_acquire_health_check_lock_failure(
        self, shared_health_manager, mock_redis_cache
    ):
        """Test failed lock acquisition"""
        mock_redis_cache.async_set_cache.return_value = False

        result = await shared_health_manager.acquire_health_check_lock()

        assert result is False

    @pytest.mark.asyncio
    async def test_acquire_health_check_lock_no_redis(self):
        """Test lock acquisition without Redis"""
        manager = SharedHealthCheckManager(redis_cache=None)

        result = await manager.acquire_health_check_lock()

        assert result is False

    @pytest.mark.asyncio
    async def test_acquire_health_check_lock_exception(
        self, shared_health_manager, mock_redis_cache
    ):
        """Test lock acquisition with exception"""
        mock_redis_cache.async_set_cache.side_effect = Exception("Redis error")

        result = await shared_health_manager.acquire_health_check_lock()

        assert result is False

    @pytest.mark.asyncio
    async def test_release_health_check_lock_success(
        self, shared_health_manager, mock_redis_cache
    ):
        """Test successful lock release"""
        mock_redis_cache.async_get_cache.return_value = shared_health_manager.pod_id

        await shared_health_manager.release_health_check_lock()

        mock_redis_cache.async_get_cache.assert_called_once_with("health_check_lock")
        mock_redis_cache.async_delete_cache.assert_called_once_with("health_check_lock")

    @pytest.mark.asyncio
    async def test_release_health_check_lock_wrong_owner(
        self, shared_health_manager, mock_redis_cache
    ):
        """Test lock release when not the owner"""
        mock_redis_cache.async_get_cache.return_value = "other_pod_id"

        await shared_health_manager.release_health_check_lock()

        mock_redis_cache.async_get_cache.assert_called_once_with("health_check_lock")
        mock_redis_cache.async_delete_cache.assert_not_called()

    @pytest.mark.asyncio
    async def test_release_health_check_lock_no_redis(self):
        """Test lock release without Redis"""
        manager = SharedHealthCheckManager(redis_cache=None)

        # Should not raise exception
        await manager.release_health_check_lock()

    @pytest.mark.asyncio
    async def test_get_cached_health_check_results_success(
        self, shared_health_manager, mock_redis_cache
    ):
        """Test getting cached health check results successfully"""
        current_time = time.time()
        cached_data = {
            "healthy_endpoints": [{"model": "test-model"}],
            "unhealthy_endpoints": [],
            "healthy_count": 1,
            "unhealthy_count": 0,
            "timestamp": current_time - 100,  # 100 seconds ago
            "checked_by": "test_pod",
        }
        mock_redis_cache.async_get_cache.return_value = json.dumps(cached_data)

        result = await shared_health_manager.get_cached_health_check_results()

        assert result is not None
        assert result["healthy_count"] == 1
        assert result["unhealthy_count"] == 0

    @pytest.mark.asyncio
    async def test_get_cached_health_check_results_expired(
        self, shared_health_manager, mock_redis_cache
    ):
        """Test getting expired cached health check results"""
        current_time = time.time()
        cached_data = {
            "healthy_endpoints": [{"model": "test-model"}],
            "unhealthy_endpoints": [],
            "healthy_count": 1,
            "unhealthy_count": 0,
            "timestamp": current_time - 400,  # 400 seconds ago (expired)
            "checked_by": "test_pod",
        }
        mock_redis_cache.async_get_cache.return_value = json.dumps(cached_data)

        result = await shared_health_manager.get_cached_health_check_results()

        assert result is None

    @pytest.mark.asyncio
    async def test_get_cached_health_check_results_no_cache(
        self, shared_health_manager, mock_redis_cache
    ):
        """Test getting cached results when no cache exists"""
        mock_redis_cache.async_get_cache.return_value = None

        result = await shared_health_manager.get_cached_health_check_results()

        assert result is None

    @pytest.mark.asyncio
    async def test_get_cached_health_check_results_no_redis(self):
        """Test getting cached results without Redis"""
        manager = SharedHealthCheckManager(redis_cache=None)

        result = await manager.get_cached_health_check_results()

        assert result is None

    @pytest.mark.asyncio
    async def test_cache_health_check_results_success(
        self, shared_health_manager, mock_redis_cache
    ):
        """Test caching health check results successfully"""
        healthy_endpoints = [{"model": "test-model-1"}]
        unhealthy_endpoints = [{"model": "test-model-2"}]

        await shared_health_manager.cache_health_check_results(
            healthy_endpoints, unhealthy_endpoints
        )

        mock_redis_cache.async_set_cache.assert_called_once()
        call_args = mock_redis_cache.async_set_cache.call_args
        assert call_args[0][0] == "health_check_results"  # key
        assert call_args[1]["ttl"] == 300  # ttl

        # Verify cached data structure
        cached_data = json.loads(call_args[0][1])
        assert cached_data["healthy_endpoints"] == healthy_endpoints
        assert cached_data["unhealthy_endpoints"] == unhealthy_endpoints
        assert cached_data["healthy_count"] == 1
        assert cached_data["unhealthy_count"] == 1
        assert "timestamp" in cached_data
        assert cached_data["checked_by"] == shared_health_manager.pod_id

    @pytest.mark.asyncio
    async def test_cache_health_check_results_no_redis(self):
        """Test caching results without Redis"""
        manager = SharedHealthCheckManager(redis_cache=None)

        # Should not raise exception
        await manager.cache_health_check_results([], [])

    @pytest.mark.asyncio
    async def test_perform_shared_health_check_with_cache(
        self, shared_health_manager, mock_redis_cache
    ):
        """Test performing shared health check when cache is available"""
        # Mock cached results
        cached_data = {
            "healthy_endpoints": [{"model": "cached-model"}],
            "unhealthy_endpoints": [],
            "healthy_count": 1,
            "unhealthy_count": 0,
            "timestamp": time.time() - 100,
        }
        mock_redis_cache.async_get_cache.return_value = json.dumps(cached_data)

        model_list = [
            {"model_name": "test-model", "litellm_params": {"model": "test-model"}}
        ]

        with patch(
            "litellm.proxy.health_check_utils.shared_health_check_manager.perform_health_check"
        ) as mock_perform:
            healthy, unhealthy, _ = (
                await shared_health_manager.perform_shared_health_check(
                    model_list, details=True
                )
            )

        # Should return cached results, not call perform_health_check
        assert healthy == [{"model": "cached-model"}]
        assert unhealthy == []
        mock_perform.assert_not_called()

    @pytest.mark.asyncio
    async def test_perform_shared_health_check_with_lock_acquisition(
        self, shared_health_manager, mock_redis_cache
    ):
        """Test performing shared health check when acquiring lock"""
        # No cached results
        mock_redis_cache.async_get_cache.return_value = None
        # Lock acquisition succeeds
        mock_redis_cache.async_set_cache.return_value = True

        model_list = [
            {"model_name": "test-model", "litellm_params": {"model": "test-model"}}
        ]
        expected_healthy = [{"model": "test-model", "status": "healthy"}]
        expected_unhealthy = []

        with patch(
            "litellm.proxy.health_check_utils.shared_health_check_manager.perform_health_check"
        ) as mock_perform:
            mock_perform.return_value = (expected_healthy, expected_unhealthy, {})

            healthy, unhealthy, _ = (
                await shared_health_manager.perform_shared_health_check(
                    model_list, details=True
                )
            )

        # Should call perform_health_check and cache results
        mock_perform.assert_called_once_with(
            model_list=model_list,
            details=True,
            max_concurrency=None,
            health_check_skip_disabled_background_models=False,
        )
        assert healthy == expected_healthy
        assert unhealthy == expected_unhealthy

        # Should cache the results
        assert mock_redis_cache.async_set_cache.call_count >= 2  # Lock + cache

    @pytest.mark.asyncio
    async def test_perform_shared_health_check_lock_failed_then_cache(
        self, shared_health_manager, mock_redis_cache
    ):
        """Test performing shared health check when lock fails but cache becomes available during polling"""
        # First call: no cache, lock fails
        # Polling finds cache on first iteration
        mock_redis_cache.async_get_cache.side_effect = [
            None,  # No cache initially (get_cached_health_check_results)
            json.dumps(
                {  # Cache available on first poll iteration
                    "healthy_endpoints": [{"model": "cached-model"}],
                    "unhealthy_endpoints": [],
                    "healthy_count": 1,
                    "unhealthy_count": 0,
                    "timestamp": time.time() - 100,
                }
            ),
        ]
        mock_redis_cache.async_set_cache.return_value = False  # Lock acquisition fails

        model_list = [
            {"model_name": "test-model", "litellm_params": {"model": "test-model"}}
        ]

        with patch("asyncio.sleep") as mock_sleep:  # Mock sleep to avoid actual delay
            healthy, unhealthy, _ = (
                await shared_health_manager.perform_shared_health_check(
                    model_list, details=True
                )
            )

        # Should poll once (5s interval) and find cached results
        mock_sleep.assert_called_once_with(5)
        assert healthy == [{"model": "cached-model"}]
        assert unhealthy == []

    @pytest.mark.asyncio
    async def test_perform_shared_health_check_fallback(self, mock_redis_cache):
        """Test performing shared health check with fallback to local health check"""
        # Use short lock_ttl so the polling loop only runs 2 iterations
        manager = SharedHealthCheckManager(
            redis_cache=mock_redis_cache,
            health_check_ttl=300,
            lock_ttl=10,
        )

        # No cache ever, lock always held by another pod
        mock_redis_cache.async_get_cache.side_effect = [
            None,  # Initial cache check
            None,  # Iteration 1: cache check
            "other_pod",  # Iteration 1: lock check (still held)
            None,  # Iteration 2: cache check
            "other_pod",  # Iteration 2: lock check (still held)
        ]
        mock_redis_cache.async_set_cache.return_value = False  # Lock acquisition fails

        model_list = [
            {"model_name": "test-model", "litellm_params": {"model": "test-model"}}
        ]
        expected_healthy = [{"model": "test-model", "status": "healthy"}]
        expected_unhealthy = []

        with (
            patch("asyncio.sleep") as mock_sleep,
            patch(
                "litellm.proxy.health_check_utils.shared_health_check_manager.perform_health_check"
            ) as mock_perform,
        ):
            mock_perform.return_value = (expected_healthy, expected_unhealthy, {})

            healthy, unhealthy, _ = await manager.perform_shared_health_check(
                model_list, details=True
            )

        # Should poll twice (5s * 2 = 10s >= lock_ttl) then fall back
        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(5)
        mock_perform.assert_called_once_with(
            model_list=model_list,
            details=True,
            max_concurrency=None,
            health_check_skip_disabled_background_models=False,
        )
        assert healthy == expected_healthy
        assert unhealthy == expected_unhealthy

    @pytest.mark.asyncio
    async def test_perform_shared_health_check_early_exit_orphaned_lock(
        self, shared_health_manager, mock_redis_cache
    ):
        """Test that polling exits early when the lock disappears without a cache write (crash recovery)"""
        mock_redis_cache.async_get_cache.side_effect = [
            None,  # Initial cache check
            None,  # Iteration 1: cache check (still no cache)
            None,  # Iteration 1: lock check -> lock gone (holder crashed)
        ]
        mock_redis_cache.async_set_cache.return_value = False  # Lock acquisition fails

        model_list = [
            {"model_name": "test-model", "litellm_params": {"model": "test-model"}}
        ]
        expected_healthy = [{"model": "test-model", "status": "healthy"}]
        expected_unhealthy = []

        with (
            patch("asyncio.sleep") as mock_sleep,
            patch(
                "litellm.proxy.health_check_utils.shared_health_check_manager.perform_health_check"
            ) as mock_perform,
        ):
            mock_perform.return_value = (expected_healthy, expected_unhealthy, {})

            healthy, unhealthy, _ = (
                await shared_health_manager.perform_shared_health_check(
                    model_list, details=True
                )
            )

        # Should detect orphaned lock after 1 iteration and fall back immediately
        mock_sleep.assert_called_once_with(5)
        mock_perform.assert_called_once_with(
            model_list=model_list,
            details=True,
            max_concurrency=None,
            health_check_skip_disabled_background_models=False,
        )
        assert healthy == expected_healthy
        assert unhealthy == expected_unhealthy

    @pytest.mark.asyncio
    async def test_perform_shared_health_check_redis_error_during_polling(
        self, shared_health_manager, mock_redis_cache
    ):
        """Test that a transient Redis error during lock polling doesn't crash the loop"""
        cached_data = json.dumps(
            {
                "healthy_endpoints": [{"model": "cached-model"}],
                "unhealthy_endpoints": [],
                "healthy_count": 1,
                "unhealthy_count": 0,
                "timestamp": time.time() - 100,
            }
        )
        mock_redis_cache.async_get_cache.side_effect = [
            None,  # Initial cache check
            None,  # Iteration 1: cache check
            Exception("Redis connection lost"),  # Iteration 1: lock check errors
            cached_data,  # Iteration 2: cache check -> found!
        ]
        mock_redis_cache.async_set_cache.return_value = False  # Lock acquisition fails

        model_list = [
            {"model_name": "test-model", "litellm_params": {"model": "test-model"}}
        ]

        with patch("asyncio.sleep") as mock_sleep:
            healthy, unhealthy, _ = (
                await shared_health_manager.perform_shared_health_check(
                    model_list, details=True
                )
            )

        # Should survive the Redis error and find cache on iteration 2
        assert mock_sleep.call_count == 2
        assert healthy == [{"model": "cached-model"}]
        assert unhealthy == []

    @pytest.mark.asyncio
    async def test_perform_shared_health_check_no_redis_skips_polling(self):
        """Test that polling is skipped entirely when redis_cache is None"""
        manager = SharedHealthCheckManager(redis_cache=None)

        model_list = [
            {"model_name": "test-model", "litellm_params": {"model": "test-model"}}
        ]
        expected_healthy = [{"model": "test-model", "status": "healthy"}]
        expected_unhealthy = []

        with (
            patch("asyncio.sleep") as mock_sleep,
            patch(
                "litellm.proxy.health_check_utils.shared_health_check_manager.perform_health_check"
            ) as mock_perform,
        ):
            mock_perform.return_value = (expected_healthy, expected_unhealthy, {})

            healthy, unhealthy, _ = await manager.perform_shared_health_check(
                model_list, details=True
            )

        # Should NOT sleep at all — falls back to local health check immediately
        mock_sleep.assert_not_called()
        mock_perform.assert_called_once_with(
            model_list=model_list,
            details=True,
            max_concurrency=None,
            health_check_skip_disabled_background_models=False,
        )
        assert healthy == expected_healthy
        assert unhealthy == expected_unhealthy

    @pytest.mark.asyncio
    async def test_is_health_check_in_progress_true(
        self, shared_health_manager, mock_redis_cache
    ):
        """Test checking if health check is in progress when it is"""
        mock_redis_cache.async_get_cache.return_value = "other_pod_id"

        result = await shared_health_manager.is_health_check_in_progress()

        assert result is True

    @pytest.mark.asyncio
    async def test_is_health_check_in_progress_false(
        self, shared_health_manager, mock_redis_cache
    ):
        """Test checking if health check is in progress when it's not"""
        mock_redis_cache.async_get_cache.return_value = None

        result = await shared_health_manager.is_health_check_in_progress()

        assert result is False

    @pytest.mark.asyncio
    async def test_is_health_check_in_progress_own_lock(
        self, shared_health_manager, mock_redis_cache
    ):
        """Test checking if health check is in progress when we own the lock"""
        mock_redis_cache.async_get_cache.return_value = shared_health_manager.pod_id

        result = await shared_health_manager.is_health_check_in_progress()

        assert result is False

    @pytest.mark.asyncio
    async def test_get_health_check_status(
        self, shared_health_manager, mock_redis_cache
    ):
        """Test getting health check status"""
        current_time = time.time()
        cached_data = {
            "healthy_endpoints": [{"model": "test-model"}],
            "unhealthy_endpoints": [],
            "healthy_count": 1,
            "unhealthy_count": 0,
            "timestamp": current_time - 100,
            "checked_by": "test_pod",
        }

        mock_redis_cache.async_get_cache.side_effect = [
            "other_pod_id",  # Lock owner
            json.dumps(cached_data),  # Cached results
        ]

        status = await shared_health_manager.get_health_check_status()

        assert status["pod_id"] == shared_health_manager.pod_id
        assert status["redis_available"] is True
        assert status["lock_ttl"] == 60
        assert status["cache_ttl"] == 300
        assert status["lock_owner"] == "other_pod_id"
        assert status["lock_in_progress"] is True
        assert status["cache_available"] is True
        assert status["last_checked_by"] == "test_pod"
        assert "cache_age_seconds" in status

    @pytest.mark.asyncio
    async def test_get_health_check_status_no_redis(self):
        """Test getting health check status without Redis"""
        manager = SharedHealthCheckManager(redis_cache=None)

        status = await manager.get_health_check_status()

        assert status["pod_id"] == manager.pod_id
        assert status["redis_available"] is False
        assert status["lock_ttl"] == 60
        assert status["cache_ttl"] == 300
