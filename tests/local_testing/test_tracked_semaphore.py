"""
Unit tests for TrackedSemaphore.

Tests the semaphore wrapper that provides public queue depth tracking
for Prometheus metrics support.

GitHub Issue: https://github.com/BerriAI/litellm/issues/17764
"""

import asyncio
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath("../.."))

from litellm.router_utils.tracked_semaphore import TrackedSemaphore, SemaphoreStats


class TestTrackedSemaphoreInit:
    """Tests for TrackedSemaphore initialization."""

    def test_init_valid_max_concurrent(self):
        """Valid max_concurrent value should create semaphore."""
        semaphore = TrackedSemaphore(max_concurrent=5)
        assert semaphore.max_concurrent == 5
        assert semaphore.active == 0
        assert semaphore.queued == 0

    def test_init_min_valid_value(self):
        """max_concurrent=1 should be valid."""
        semaphore = TrackedSemaphore(max_concurrent=1)
        assert semaphore.max_concurrent == 1

    def test_init_zero_raises_value_error(self):
        """max_concurrent=0 should raise ValueError."""
        with pytest.raises(ValueError, match="max_concurrent must be at least 1"):
            TrackedSemaphore(max_concurrent=0)

    def test_init_negative_raises_value_error(self):
        """Negative max_concurrent should raise ValueError."""
        with pytest.raises(ValueError, match="max_concurrent must be at least 1"):
            TrackedSemaphore(max_concurrent=-1)


class TestTrackedSemaphoreAcquireRelease:
    """Tests for acquire/release behavior."""

    @pytest.mark.asyncio
    async def test_acquire_increments_active(self):
        """Acquiring semaphore should increment active count."""
        semaphore = TrackedSemaphore(max_concurrent=3)
        assert semaphore.active == 0

        await semaphore.acquire()
        assert semaphore.active == 1

        await semaphore.acquire()
        assert semaphore.active == 2

    @pytest.mark.asyncio
    async def test_release_decrements_active(self):
        """Releasing semaphore should decrement active count."""
        semaphore = TrackedSemaphore(max_concurrent=3)
        await semaphore.acquire()
        await semaphore.acquire()
        assert semaphore.active == 2

        semaphore.release()
        assert semaphore.active == 1

        semaphore.release()
        assert semaphore.active == 0

    @pytest.mark.asyncio
    async def test_queued_count_while_waiting(self):
        """Tasks waiting for semaphore should increment queued count."""
        semaphore = TrackedSemaphore(max_concurrent=1)

        # Acquire the only slot
        await semaphore.acquire()
        assert semaphore.active == 1
        assert semaphore.queued == 0

        # Start a task that will wait
        async def waiting_task():
            await semaphore.acquire()
            semaphore.release()

        task = asyncio.create_task(waiting_task())

        # Give the task time to start waiting
        await asyncio.sleep(0.01)

        assert semaphore.queued == 1, "Waiting task should be counted as queued"

        # Release to let the waiting task proceed
        semaphore.release()
        await task

        assert semaphore.queued == 0
        assert semaphore.active == 0


class TestTrackedSemaphoreContextManager:
    """Tests for async context manager behavior."""

    @pytest.mark.asyncio
    async def test_async_context_manager_basic(self):
        """Context manager should acquire on enter and release on exit."""
        semaphore = TrackedSemaphore(max_concurrent=3)

        async with semaphore:
            assert semaphore.active == 1

        assert semaphore.active == 0

    @pytest.mark.asyncio
    async def test_async_context_manager_releases_on_exception(self):
        """Context manager should release even if exception occurs."""
        semaphore = TrackedSemaphore(max_concurrent=3)

        with pytest.raises(ValueError):
            async with semaphore:
                assert semaphore.active == 1
                raise ValueError("Test exception")

        assert semaphore.active == 0

    @pytest.mark.asyncio
    async def test_async_context_manager_nested(self):
        """Multiple context managers should stack correctly."""
        semaphore = TrackedSemaphore(max_concurrent=3)

        async with semaphore:
            assert semaphore.active == 1
            async with semaphore:
                assert semaphore.active == 2
            assert semaphore.active == 1

        assert semaphore.active == 0


class TestTrackedSemaphoreCancellation:
    """Tests for task cancellation handling."""

    @pytest.mark.asyncio
    async def test_cancellation_decrements_queued(self):
        """Cancelled task should decrement queued count."""
        semaphore = TrackedSemaphore(max_concurrent=1)

        # Acquire the only slot
        await semaphore.acquire()

        # Start a task that will wait
        async def waiting_task():
            await semaphore.acquire()

        task = asyncio.create_task(waiting_task())

        # Give the task time to start waiting
        await asyncio.sleep(0.01)
        assert semaphore.queued == 1

        # Cancel the waiting task
        task.cancel()

        # Give time for cancellation to process
        try:
            await task
        except asyncio.CancelledError:
            pass

        await asyncio.sleep(0.01)
        assert semaphore.queued == 0, "Cancelled task should no longer be queued"

        # Clean up
        semaphore.release()


class TestTrackedSemaphoreStats:
    """Tests for stats property."""

    @pytest.mark.asyncio
    async def test_stats_property_returns_correct_values(self):
        """Stats property should return accurate SemaphoreStats."""
        semaphore = TrackedSemaphore(max_concurrent=5)

        stats = semaphore.stats
        assert isinstance(stats, SemaphoreStats)
        assert stats.max_concurrent == 5
        assert stats.active == 0
        assert stats.queued == 0
        assert stats.available == 5

        await semaphore.acquire()
        await semaphore.acquire()

        stats = semaphore.stats
        assert stats.active == 2
        assert stats.available == 3

    def test_stats_available_property(self):
        """SemaphoreStats.available should compute correctly."""
        stats = SemaphoreStats(max_concurrent=10, active=3, queued=2)
        assert stats.available == 7


class TestTrackedSemaphoreLocked:
    """Tests for locked() method."""

    @pytest.mark.asyncio
    async def test_locked_when_at_capacity(self):
        """locked() should return True when all slots are taken."""
        semaphore = TrackedSemaphore(max_concurrent=2)

        assert semaphore.locked() is False

        await semaphore.acquire()
        assert semaphore.locked() is False

        await semaphore.acquire()
        assert semaphore.locked() is True

        semaphore.release()
        assert semaphore.locked() is False

    @pytest.mark.asyncio
    async def test_locked_compatible_with_asyncio_semaphore(self):
        """locked() should behave like asyncio.Semaphore.locked()."""
        tracked = TrackedSemaphore(max_concurrent=1)
        stdlib = asyncio.Semaphore(1)

        # Both should start unlocked
        assert tracked.locked() == stdlib.locked()

        # Both should be locked after acquiring the only slot
        await tracked.acquire()
        await stdlib.acquire()
        assert tracked.locked() is True
        assert stdlib.locked() is True

        # Both should be unlocked after release
        tracked.release()
        stdlib.release()
        assert tracked.locked() is False
        assert stdlib.locked() is False


class TestTrackedSemaphoreConcurrency:
    """Tests for concurrent usage patterns."""

    @pytest.mark.asyncio
    async def test_multiple_concurrent_tasks(self):
        """Multiple tasks should correctly track active/queued counts."""
        semaphore = TrackedSemaphore(max_concurrent=2)
        results = []

        async def worker(worker_id: int, delay: float):
            async with semaphore:
                results.append(f"start_{worker_id}")
                await asyncio.sleep(delay)
                results.append(f"end_{worker_id}")

        # Start 4 tasks with semaphore limit of 2
        tasks = [
            asyncio.create_task(worker(i, 0.05))
            for i in range(4)
        ]

        # Give time for first batch to start
        await asyncio.sleep(0.01)
        assert semaphore.active == 2
        assert semaphore.queued == 2

        # Wait for all tasks to complete
        await asyncio.gather(*tasks)

        assert semaphore.active == 0
        assert semaphore.queued == 0
        assert len(results) == 8  # 4 starts + 4 ends
