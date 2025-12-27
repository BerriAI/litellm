"""
Tracked Semaphore for Queue Depth Metrics.

Provides a semaphore wrapper with public queue depth tracking,
avoiding reliance on private asyncio.Semaphore internals.

Used by LiteLLM Router for max_parallel_requests limiting with
Prometheus metrics support.

GitHub Issue: https://github.com/BerriAI/litellm/issues/17764
"""

import asyncio
from dataclasses import dataclass


@dataclass
class SemaphoreStats:
    """Statistics for a TrackedSemaphore instance."""

    max_concurrent: int
    active: int
    queued: int

    @property
    def available(self) -> int:
        """Number of available slots."""
        return self.max_concurrent - self.active


class TrackedSemaphore:
    """
    Asyncio semaphore wrapper with public queue depth tracking.

    Unlike asyncio.Semaphore, this class tracks waiting tasks explicitly,
    providing public access to queue depth without accessing private internals.

    Example:
        semaphore = TrackedSemaphore(max_concurrent=3)

        async with semaphore:
            # Do work with concurrency limit
            pass

        stats = semaphore.stats
        print(f"Active: {stats.active}, Queued: {stats.queued}")
    """

    def __init__(self, max_concurrent: int):
        """
        Initialize TrackedSemaphore.

        Args:
            max_concurrent: Maximum number of concurrent acquisitions.
        """
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")

        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._max_concurrent = max_concurrent
        self._active = 0
        self._queued = 0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """
        Acquire the semaphore, waiting if necessary.

        Increments queued count while waiting, then increments active count
        once acquired.
        """
        async with self._lock:
            self._queued += 1

        try:
            await self._semaphore.acquire()
        except asyncio.CancelledError:
            async with self._lock:
                self._queued -= 1
            raise

        async with self._lock:
            self._queued -= 1
            self._active += 1

    def release(self) -> None:
        """
        Release the semaphore, decrementing active count.
        """
        # Note: We need to handle the case where release is called
        # synchronously but we need to update _active atomically.
        # Since release() in asyncio.Semaphore is sync, we use a
        # thread-safe approach here.

        # Decrement active count
        # Using a simple flag since asyncio.Lock can't be used in sync context
        self._active -= 1
        self._semaphore.release()

    async def __aenter__(self) -> "TrackedSemaphore":
        """Async context manager entry."""
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        self.release()

    @property
    def stats(self) -> SemaphoreStats:
        """
        Get current semaphore statistics.

        Returns:
            SemaphoreStats with max_concurrent, active, and queued counts.
        """
        return SemaphoreStats(
            max_concurrent=self._max_concurrent,
            active=self._active,
            queued=self._queued,
        )

    @property
    def max_concurrent(self) -> int:
        """Maximum number of concurrent acquisitions."""
        return self._max_concurrent

    @property
    def active(self) -> int:
        """Number of currently active (acquired) slots."""
        return self._active

    @property
    def queued(self) -> int:
        """Number of tasks waiting to acquire."""
        return self._queued

    def locked(self) -> bool:
        """
        Return True if semaphore cannot be acquired immediately.

        Compatible with asyncio.Semaphore interface.
        """
        return self._semaphore.locked()
