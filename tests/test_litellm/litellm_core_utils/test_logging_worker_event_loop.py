"""
Tests for LoggingWorker event-loop rebinding safety.

Verifies that GLOBAL_LOGGING_WORKER handles event loop changes without
raising ``RuntimeError: Queue is bound to a different event loop``.
Covers the scenario in https://github.com/BerriAI/litellm/issues/14521:
pytest-parametrized async tests create a new event loop per test, but the
module-level singleton survives across tests.
"""

import asyncio
import atexit
import contextvars

import pytest

from litellm.litellm_core_utils.logging_worker import LoggingWorker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _noop_coro():
    """Coroutine that does nothing — used as a logging payload."""


def _make_fresh_worker() -> LoggingWorker:
    """Create a standalone worker (no atexit side-effects in tests)."""
    w = LoggingWorker(timeout=5.0, max_queue_size=100, concurrency=4)
    atexit.unregister(w._flush_on_exit)
    return w


# ---------------------------------------------------------------------------
# Tests — all run in their own event loop via pytest-asyncio
# ---------------------------------------------------------------------------


class TestEpochPreventsStaleAccess:
    """Worker loop exits when epoch is bumped (event loop changed)."""

    @pytest.mark.asyncio
    async def test_worker_exits_on_epoch_change(self):
        """Worker loop should stop when _epoch is incremented."""
        w = _make_fresh_worker()
        w.start()

        assert w._worker_task is not None
        assert not w._worker_task.done()

        # Let worker start and capture the current epoch
        await asyncio.sleep(0.01)

        # Bump epoch as _ensure_queue would on a new loop
        w._epoch += 1

        # Enqueue something so the worker wakes up from queue.get()
        w._queue.put_nowait({"coroutine": _noop_coro(), "context": contextvars.copy_context()})

        # Give the worker a chance to process and detect the epoch change
        for _ in range(20):
            if w._worker_task.done():
                break
            await asyncio.sleep(0.05)

        assert w._worker_task.done()

    @pytest.mark.asyncio
    async def test_ensure_queue_bumps_epoch_on_loop_change(self):
        """_ensure_queue should bump epoch when the loop is different."""
        w = _make_fresh_worker()
        w.start()
        old_epoch = w._epoch

        # Simulate a new loop by faking a different bound loop
        w._bound_loop = object()  # type: ignore[assignment]
        w._ensure_queue()

        assert w._epoch == old_epoch + 1

    @pytest.mark.asyncio
    async def test_epoch_not_bumped_on_same_loop(self):
        """_ensure_queue should NOT bump epoch when the loop is the same."""
        w = _make_fresh_worker()
        w.start()
        old_epoch = w._epoch

        w._ensure_queue()

        assert w._epoch == old_epoch


class TestCrossLoopIsolation:
    """Simulates two sequential event loops reusing the same LoggingWorker."""

    @pytest.mark.asyncio
    async def test_enqueue_on_new_loop_does_not_raise(self):
        """After loop change, enqueue should work without RuntimeError."""
        w = _make_fresh_worker()
        w.start()

        # Simulate loop change
        w._bound_loop = object()  # type: ignore[assignment]

        # This would previously raise "Queue is bound to a different event loop"
        w.ensure_initialized_and_enqueue(_noop_coro())

        # Worker should be running on the current loop
        assert w._worker_task is not None
        assert not w._worker_task.done()
        assert w._bound_loop is asyncio.get_running_loop()

        await w.stop()

    @pytest.mark.asyncio
    async def test_tasks_processed_after_loop_change(self):
        """Tasks enqueued after loop change should be processed."""
        w = _make_fresh_worker()

        processed = []

        async def record():
            processed.append(1)

        # Simulate loop change
        w._bound_loop = object()  # type: ignore[assignment]
        w._queue = asyncio.Queue(maxsize=100)  # stale queue on "old loop"

        w.ensure_initialized_and_enqueue(record())

        # Wait for processing
        await asyncio.sleep(0.2)

        assert len(processed) == 1

        await w.stop()


class TestWorkerLoopLocalCapture:
    """Worker loop uses local refs, not self._queue."""

    @pytest.mark.asyncio
    async def test_cancelled_worker_discards_own_queue(self):
        """When cancelled, the worker discards items from its own queue."""
        w = _make_fresh_worker()
        w.start()

        # Let worker start and reach await queue.get()
        await asyncio.sleep(0.01)

        original_queue = w._queue

        # Enqueue a task
        w.enqueue(_noop_coro())

        # Let the worker pick up the task
        await asyncio.sleep(0.01)

        # Replace self._queue (simulating _ensure_queue on new loop)
        new_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        w._queue = new_queue

        # Enqueue another task to the OLD queue (simulates in-flight item)
        original_queue.put_nowait({"coroutine": _noop_coro(), "context": contextvars.copy_context()})

        # Cancel the worker — it should discard from original_queue
        if w._worker_task is not None:
            w._worker_task.cancel()
            try:
                await w._worker_task
            except asyncio.CancelledError:
                pass

        # The original queue should be drained (coroutines closed)
        assert original_queue.empty()


class TestReset:
    """reset() clears all state for clean test fixture usage."""

    @pytest.mark.asyncio
    async def test_reset_clears_state(self):
        w = _make_fresh_worker()
        w.start()
        w.enqueue(_noop_coro())

        await w.reset()

        assert w._queue is None
        assert w._sem is None
        assert w._bound_loop is None
        assert w._worker_task is None

    @pytest.mark.asyncio
    async def test_reset_allows_restart(self):
        w = _make_fresh_worker()
        w.start()

        await w.reset()

        # Should be able to start again cleanly
        w.ensure_initialized_and_enqueue(_noop_coro())
        assert w._worker_task is not None
        assert not w._worker_task.done()

        await w.stop()

    @pytest.mark.asyncio
    async def test_reset_bumps_epoch(self):
        w = _make_fresh_worker()
        old_epoch = w._epoch
        w.start()

        await w.reset()

        assert w._epoch == old_epoch + 1


class TestMultipleLoopCycles:
    """Simulate N sequential loop changes on a single worker."""

    @pytest.mark.asyncio
    async def test_three_loop_cycles(self):
        """Worker should survive 3 consecutive loop rebinds."""
        w = _make_fresh_worker()

        for i in range(3):
            # Simulate new loop
            if i > 0:
                w._bound_loop = object()  # type: ignore[assignment]

            processed = []

            async def record(idx=i):
                processed.append(idx)

            w.ensure_initialized_and_enqueue(record())
            await asyncio.sleep(0.2)

            assert len(processed) == 1
            assert processed[0] == i

        await w.stop()
