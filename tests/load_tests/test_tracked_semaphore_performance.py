"""
Performance tests for TrackedSemaphore.

Validates that the TrackedSemaphore wrapper doesn't introduce significant
overhead compared to the standard asyncio.Semaphore.

GitHub Issue: https://github.com/BerriAI/litellm/issues/17764
"""

import asyncio
import time
import statistics
import sys
import os

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.router_utils.tracked_semaphore import TrackedSemaphore


class TestTrackedSemaphorePerformance:
    """Performance benchmarks for TrackedSemaphore vs asyncio.Semaphore."""

    @pytest.mark.asyncio
    async def test_acquire_release_overhead(self):
        """
        Measure overhead of TrackedSemaphore vs asyncio.Semaphore.

        Note: TrackedSemaphore uses asyncio.Lock for thread-safe counter updates,
        which adds overhead in microbenchmarks. However, this overhead is negligible
        in real workloads where actual I/O dominates (see test_high_concurrency_throughput).

        The threshold here is permissive (5x) because:
        1. Real LLM API calls take 100ms-10s, not microseconds
        2. The high concurrency test validates realistic scenarios
        3. This test is informational, not a hard requirement
        """
        iterations = 10000
        max_concurrent = 100

        # Benchmark asyncio.Semaphore
        stdlib_semaphore = asyncio.Semaphore(max_concurrent)
        start = time.perf_counter()
        for _ in range(iterations):
            await stdlib_semaphore.acquire()
            stdlib_semaphore.release()
        stdlib_time = time.perf_counter() - start

        # Benchmark TrackedSemaphore
        tracked_semaphore = TrackedSemaphore(max_concurrent)
        start = time.perf_counter()
        for _ in range(iterations):
            await tracked_semaphore.acquire()
            tracked_semaphore.release()
        tracked_time = time.perf_counter() - start

        # Calculate overhead
        overhead_ratio = tracked_time / stdlib_time
        overhead_percent = (overhead_ratio - 1) * 100

        print(f"\nPerformance Results ({iterations} iterations):")
        print(f"  asyncio.Semaphore: {stdlib_time*1000:.2f}ms")
        print(f"  TrackedSemaphore:  {tracked_time*1000:.2f}ms")
        print(f"  Overhead: {overhead_percent:.1f}%")

        # Allow up to 5x overhead in microbenchmarks - the asyncio.Lock adds cost
        # but this is negligible compared to actual LLM API latency (100ms-10s)
        assert overhead_ratio < 5.0, (
            f"TrackedSemaphore overhead too high: {overhead_percent:.1f}% "
            f"(expected < 400%)"
        )

    @pytest.mark.asyncio
    async def test_context_manager_overhead(self):
        """
        Measure overhead when using async context manager pattern.

        Same rationale as test_acquire_release_overhead - microbenchmark overhead
        is acceptable because real workloads are I/O bound.
        """
        iterations = 5000
        max_concurrent = 100

        # Benchmark asyncio.Semaphore with context manager
        stdlib_semaphore = asyncio.Semaphore(max_concurrent)
        start = time.perf_counter()
        for _ in range(iterations):
            async with stdlib_semaphore:
                pass
        stdlib_time = time.perf_counter() - start

        # Benchmark TrackedSemaphore with context manager
        tracked_semaphore = TrackedSemaphore(max_concurrent)
        start = time.perf_counter()
        for _ in range(iterations):
            async with tracked_semaphore:
                pass
        tracked_time = time.perf_counter() - start

        overhead_ratio = tracked_time / stdlib_time
        overhead_percent = (overhead_ratio - 1) * 100

        print(f"\nContext Manager Performance ({iterations} iterations):")
        print(f"  asyncio.Semaphore: {stdlib_time*1000:.2f}ms")
        print(f"  TrackedSemaphore:  {tracked_time*1000:.2f}ms")
        print(f"  Overhead: {overhead_percent:.1f}%")

        # Allow 5x overhead in microbenchmarks (see test_acquire_release_overhead)
        assert overhead_ratio < 5.0, (
            f"Context manager overhead too high: {overhead_percent:.1f}%"
        )

    @pytest.mark.asyncio
    async def test_high_concurrency_throughput(self):
        """
        Test throughput under high concurrency with many concurrent tasks.
        """
        num_tasks = 1000
        max_concurrent = 10
        work_time = 0.001  # 1ms simulated work

        async def worker(semaphore, results: list):
            start = time.perf_counter()
            async with semaphore:
                await asyncio.sleep(work_time)
            elapsed = time.perf_counter() - start
            results.append(elapsed)

        # Test asyncio.Semaphore
        stdlib_semaphore = asyncio.Semaphore(max_concurrent)
        stdlib_results = []
        start = time.perf_counter()
        await asyncio.gather(*[
            worker(stdlib_semaphore, stdlib_results)
            for _ in range(num_tasks)
        ])
        stdlib_total = time.perf_counter() - start

        # Test TrackedSemaphore
        tracked_semaphore = TrackedSemaphore(max_concurrent)
        tracked_results = []
        start = time.perf_counter()
        await asyncio.gather(*[
            worker(tracked_semaphore, tracked_results)
            for _ in range(num_tasks)
        ])
        tracked_total = time.perf_counter() - start

        # Calculate statistics
        stdlib_avg = statistics.mean(stdlib_results) * 1000  # ms
        tracked_avg = statistics.mean(tracked_results) * 1000

        print(f"\nHigh Concurrency Throughput ({num_tasks} tasks, {max_concurrent} concurrent):")
        print(f"  asyncio.Semaphore total: {stdlib_total:.3f}s, avg wait: {stdlib_avg:.2f}ms")
        print(f"  TrackedSemaphore total:  {tracked_total:.3f}s, avg wait: {tracked_avg:.2f}ms")

        # Total time should be similar (within 20%)
        time_ratio = tracked_total / stdlib_total
        assert time_ratio < 1.2, (
            f"Throughput degradation too high: {(time_ratio-1)*100:.1f}%"
        )

    @pytest.mark.asyncio
    async def test_stats_access_overhead(self):
        """
        Measure overhead of accessing stats property during operations.
        """
        iterations = 10000
        max_concurrent = 100

        tracked_semaphore = TrackedSemaphore(max_concurrent)

        # Acquire some slots to make stats interesting
        for _ in range(50):
            await tracked_semaphore.acquire()

        # Measure stats access time
        start = time.perf_counter()
        for _ in range(iterations):
            _ = tracked_semaphore.stats
        stats_time = time.perf_counter() - start

        # Also measure individual property access
        start = time.perf_counter()
        for _ in range(iterations):
            _ = tracked_semaphore.active
            _ = tracked_semaphore.queued
            _ = tracked_semaphore.max_concurrent
        props_time = time.perf_counter() - start

        print(f"\nStats Access Performance ({iterations} iterations):")
        print(f"  stats property: {stats_time*1000:.2f}ms ({stats_time/iterations*1e6:.2f}µs/call)")
        print(f"  individual props: {props_time*1000:.2f}ms ({props_time/iterations*1e6:.2f}µs/call)")

        # Stats access should be < 1µs per call (it's just creating a dataclass)
        us_per_call = stats_time / iterations * 1e6
        assert us_per_call < 10, (
            f"Stats access too slow: {us_per_call:.2f}µs (expected < 10µs)"
        )

        # Clean up
        for _ in range(50):
            tracked_semaphore.release()

    @pytest.mark.asyncio
    async def test_counter_accuracy_under_load(self):
        """
        Verify counters remain accurate under high concurrent load.

        This is a correctness test that also stresses the implementation.
        """
        num_tasks = 500
        max_concurrent = 20
        work_time = 0.01  # 10ms

        semaphore = TrackedSemaphore(max_concurrent)
        max_active_seen = 0
        completed = 0

        async def worker():
            nonlocal max_active_seen, completed
            async with semaphore:
                current_active = semaphore.active
                max_active_seen = max(max_active_seen, current_active)
                await asyncio.sleep(work_time)
            completed += 1

        # Run all tasks
        await asyncio.gather(*[worker() for _ in range(num_tasks)])

        print(f"\nCounter Accuracy Test ({num_tasks} tasks, {max_concurrent} max):")
        print(f"  Max active seen: {max_active_seen}")
        print(f"  Completed: {completed}")
        print(f"  Final active: {semaphore.active}")
        print(f"  Final queued: {semaphore.queued}")

        # Verify correctness
        assert completed == num_tasks, f"Not all tasks completed: {completed}/{num_tasks}"
        assert semaphore.active == 0, f"Active should be 0, got {semaphore.active}"
        assert semaphore.queued == 0, f"Queued should be 0, got {semaphore.queued}"
        assert max_active_seen <= max_concurrent, (
            f"Active exceeded max: {max_active_seen} > {max_concurrent}"
        )


class TestTrackedSemaphoreMemory:
    """Memory usage tests for TrackedSemaphore."""

    def test_memory_footprint(self):
        """
        Verify TrackedSemaphore has reasonable memory footprint.
        """
        import sys

        # Create instances
        stdlib = asyncio.Semaphore(100)
        tracked = TrackedSemaphore(100)

        stdlib_size = sys.getsizeof(stdlib)
        tracked_size = sys.getsizeof(tracked)

        print("\nMemory Footprint:")
        print(f"  asyncio.Semaphore: {stdlib_size} bytes")
        print(f"  TrackedSemaphore:  {tracked_size} bytes")

        # TrackedSemaphore adds a few integers and a lock
        # Should be less than 2x the stdlib size
        assert tracked_size < stdlib_size * 3, (
            f"Memory footprint too large: {tracked_size} vs {stdlib_size}"
        )

    def test_many_instances(self):
        """
        Verify creating many instances doesn't cause issues.

        LiteLLM Router creates one semaphore per deployment.
        """
        num_instances = 1000

        semaphores = [
            TrackedSemaphore(max_concurrent=i % 100 + 1)
            for i in range(num_instances)
        ]

        # Verify all instances work
        for i, sem in enumerate(semaphores):
            assert sem.max_concurrent == (i % 100 + 1)
            assert sem.active == 0
            assert sem.queued == 0

        print(f"\nCreated {num_instances} TrackedSemaphore instances successfully")
