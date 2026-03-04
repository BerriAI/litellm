"""
Tests for litellm.litellm_core_utils.task_registry
"""

import asyncio
import logging

import pytest

from litellm.litellm_core_utils.task_registry import TaskRegistry, tracked_create_task


@pytest.fixture(autouse=True)
def _reset_registry():
    """Reset the singleton before/after each test."""
    TaskRegistry.reset_instance()
    yield
    TaskRegistry.reset_instance()


@pytest.mark.asyncio
async def test_create_task_tracks_and_cleans_up():
    """Tasks are tracked while pending and removed when done."""
    registry = TaskRegistry.get_instance()
    assert registry.pending_count == 0

    event = asyncio.Event()

    async def _wait():
        await event.wait()

    task = registry.create_task(_wait(), name="test-task")
    assert registry.pending_count == 1
    assert task.get_name() == "test-task"

    event.set()
    await task
    # Allow done callback to fire
    await asyncio.sleep(0)
    assert registry.pending_count == 0


@pytest.mark.asyncio
async def test_done_callback_removes_completed_task():
    """Completed tasks are automatically removed from the registry."""
    registry = TaskRegistry.get_instance()

    async def _noop():
        return 42

    task = registry.create_task(_noop(), name="noop")
    await task
    await asyncio.sleep(0)
    assert registry.pending_count == 0


@pytest.mark.asyncio
async def test_failed_task_is_cleaned_up(caplog):
    """Tasks that raise exceptions are still removed from the registry."""
    registry = TaskRegistry.get_instance()

    async def _fail():
        raise ValueError("boom")

    with caplog.at_level(logging.DEBUG):
        task = registry.create_task(_fail(), name="fail-task")
        with pytest.raises(ValueError, match="boom"):
            await task
        await asyncio.sleep(0)

    assert registry.pending_count == 0


@pytest.mark.asyncio
async def test_shutdown_cancels_pending_tasks():
    """shutdown() cancels all pending tasks and clears the registry."""
    registry = TaskRegistry.get_instance()

    async def _block():
        await asyncio.sleep(3600)

    t1 = registry.create_task(_block(), name="block-1")
    t2 = registry.create_task(_block(), name="block-2")
    assert registry.pending_count == 2

    await registry.shutdown(timeout=1.0)
    assert registry.pending_count == 0
    assert t1.cancelled()
    assert t2.cancelled()


@pytest.mark.asyncio
async def test_shutdown_on_empty_registry():
    """shutdown() is a no-op when there are no tasks."""
    registry = TaskRegistry.get_instance()
    await registry.shutdown()
    assert registry.pending_count == 0


@pytest.mark.asyncio
async def test_leak_warning(caplog):
    """A warning is emitted when pending tasks exceed the threshold."""
    registry = TaskRegistry(max_pending_warning=2)
    blockers = []

    async def _block():
        await asyncio.sleep(3600)

    with caplog.at_level(logging.WARNING):
        for i in range(5):
            blockers.append(registry.create_task(_block(), name=f"t-{i}"))

    # Warning should fire only once despite multiple tasks exceeding threshold
    assert caplog.text.count("Possible task leak detected") == 1

    await registry.shutdown(timeout=1.0)


@pytest.mark.asyncio
async def test_shutdown_timeout_is_enforced(caplog):
    """shutdown() logs a warning and returns when timeout is exceeded."""
    registry = TaskRegistry.get_instance()

    async def _stubborn():
        """Task that catches cancellation and keeps running."""
        while True:
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                await asyncio.sleep(10)

    registry.create_task(_stubborn(), name="stubborn")
    await asyncio.sleep(0)

    with caplog.at_level(logging.WARNING):
        await registry.shutdown(timeout=0.05)

    assert "shutdown timed out after" in caplog.text
    assert registry.pending_count == 0


@pytest.mark.asyncio
async def test_tracked_create_task_convenience():
    """Module-level tracked_create_task uses the global singleton."""
    async def _noop():
        return 1

    task = tracked_create_task(_noop(), name="convenience")
    result = await task
    assert result == 1
    await asyncio.sleep(0)
    assert TaskRegistry.get_instance().pending_count == 0


@pytest.mark.asyncio
async def test_singleton_identity():
    """get_instance always returns the same object."""
    a = TaskRegistry.get_instance()
    b = TaskRegistry.get_instance()
    assert a is b
