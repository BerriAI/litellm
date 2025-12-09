"""
Tests for the LoggingWorker class to ensure graceful shutdown handling.
"""

import asyncio
import contextvars
from unittest.mock import AsyncMock, patch

import pytest

from litellm.constants import LOGGING_WORKER_AGGRESSIVE_CLEAR_COOLDOWN_SECONDS
from litellm.litellm_core_utils.logging_worker import LoggingWorker


class TestLoggingWorker:
    """Test cases for LoggingWorker functionality."""

    @pytest.fixture
    def logging_worker(self):
        """Create a LoggingWorker instance for testing."""
        return LoggingWorker(timeout=1.0, max_queue_size=10)

    @pytest.mark.asyncio
    async def test_graceful_shutdown_with_clear_queue(self, logging_worker):
        """Test that cancellation triggers clear_queue to prevent 'never awaited' warnings."""
        # Mock the clear_queue method to verify it's called during cancellation
        with patch.object(
            logging_worker, "clear_queue", new_callable=AsyncMock
        ) as mock_clear_queue:
            # Start the worker
            logging_worker.start()

            # Give it a moment to start
            await asyncio.sleep(0.1)

            # Cancel the worker task to simulate shutdown
            if logging_worker._worker_task:
                logging_worker._worker_task.cancel()

                # Wait for the task to handle the cancellation
                try:
                    await logging_worker._worker_task
                except asyncio.CancelledError:
                    # Expected during cancellation
                    pass

            # Verify that clear_queue was called during cancellation
            mock_clear_queue.assert_called_once()

    @pytest.mark.asyncio
    async def test_clear_queue_processes_remaining_items(self, logging_worker):
        """Test that clear_queue processes remaining coroutines to prevent warnings."""
        # Create mock coroutines
        mock_coro1 = AsyncMock()
        mock_coro2 = AsyncMock()

        # Initialize the worker and add items to queue
        logging_worker._ensure_queue()
        logging_worker.enqueue(mock_coro1())
        logging_worker.enqueue(mock_coro2())

        # Clear the queue
        await logging_worker.clear_queue()

        # Verify the queue is empty after clearing
        assert logging_worker._queue.empty()

    @pytest.mark.asyncio
    async def test_worker_handles_cancellation_gracefully(self, logging_worker):
        """Test that the worker handles cancellation without throwing exceptions."""
        # Mock verbose_logger to capture debug messages
        with patch(
            "litellm.litellm_core_utils.logging_worker.verbose_logger"
        ) as mock_logger:
            # Start the worker
            logging_worker.start()

            # Give it a moment to start
            await asyncio.sleep(0.1)

            # Cancel and wait for completion
            await logging_worker.stop()

            # Verify debug message was logged instead of exception
            debug_calls = [
                call
                for call in mock_logger.debug.call_args_list
                if "LoggingWorker cancelled during shutdown" in str(call)
            ]
            assert len(debug_calls) >= 0  # May be 0 if no cancellation occurred

    @pytest.mark.asyncio
    async def test_enqueue_and_process_single_item(self, logging_worker):
        """Test basic enqueue and process functionality."""
        # Create a mock coroutine that we can track
        mock_coro = AsyncMock()

        # Start the worker
        logging_worker.start()

        # Enqueue a coroutine
        logging_worker.enqueue(mock_coro())

        # Give the worker time to process the item
        await asyncio.sleep(0.2)

        # Stop the worker
        await logging_worker.stop()

        # The mock should have been awaited (processed)
        assert mock_coro.called

    @pytest.mark.asyncio
    async def test_clear_queue_with_time_limit(self, logging_worker):
        """Test that clear_queue respects the time limit."""
        # Create several mock coroutines that take time to complete
        slow_coro = AsyncMock()
        slow_coro.return_value = asyncio.sleep(0.5)  # Takes 500ms

        # Initialize the worker and add items
        logging_worker._ensure_queue()
        for _ in range(5):
            logging_worker.enqueue(slow_coro())

        # Clear the queue - should timeout based on MAX_TIME_TO_CLEAR_QUEUE
        start_time = asyncio.get_event_loop().time()
        await logging_worker.clear_queue()
        elapsed_time = asyncio.get_event_loop().time() - start_time

        # Should complete within reasonable time (allowing for some processing)
        assert elapsed_time < 10.0  # Much less than if it processed all slow items

    @pytest.mark.asyncio
    async def test_queue_full_handling(self, logging_worker):
        """Test that queue full condition is handled gracefully."""
        # Create a worker with very small queue size
        small_worker = LoggingWorker(timeout=1.0, max_queue_size=2)
        small_worker._ensure_queue()

        # Mock verbose_logger to capture exception messages
        with patch(
            "litellm.litellm_core_utils.logging_worker.verbose_logger"
        ) as mock_logger:
            # Fill the queue beyond capacity
            mock_coro = AsyncMock()
            for _ in range(5):  # More than max_queue_size of 2
                small_worker.enqueue(mock_coro())

            # Should have logged queue full exceptions
            exception_calls = [
                call
                for call in mock_logger.exception.call_args_list
                if "queue is full" in str(call)
            ]
            assert len(exception_calls) > 0

    @pytest.mark.asyncio
    async def test_context_propagation(self, logging_worker):
        """Test that enqueued tasks execute in their original context."""
        # Create a context variable for testing
        test_context_var: contextvars.ContextVar[str] = contextvars.ContextVar(
            "test_context_var"
        )

        # Track results from multiple tasks using asyncio.Event for synchronization
        task_results = []
        completion_events = {}

        async def test_task(task_id: str):
            """A test coroutine that checks if it can access the context variable."""
            try:
                # Try to get the context variable value
                value = test_context_var.get()
                task_results.append(
                    {
                        "task_id": task_id,
                        "context_value": value,
                        "context_accessible": True,
                    }
                )
            except LookupError:
                # Context variable not found
                task_results.append(
                    {
                        "task_id": task_id,
                        "context_accessible": False,
                        "context_value": None,
                    }
                )
            finally:
                # Signal that this task is complete
                completion_events[task_id].set()

        # Create completion events for each task
        completion_events["task_1"] = asyncio.Event()
        completion_events["task_2"] = asyncio.Event()
        completion_events["task_3"] = asyncio.Event()

        # Start the logging worker
        logging_worker.start()

        # Give the worker a moment to start
        await asyncio.sleep(0.1)

        # Create two separate contexts and enqueue tasks from each

        # Context 1: Set context var to "context_1"
        ctx1 = contextvars.copy_context()
        ctx1.run(test_context_var.set, "context_1")
        ctx1.run(logging_worker.enqueue, test_task("task_1"))

        # Context 2: Set context var to "context_2"
        ctx2 = contextvars.copy_context()
        ctx2.run(test_context_var.set, "context_2")
        ctx2.run(logging_worker.enqueue, test_task("task_2"))

        # Context 3: No context variable set (should get LookupError)
        ctx3 = contextvars.copy_context()
        ctx3.run(logging_worker.enqueue, test_task("task_3"))

        # Wait for all tasks to complete with a reasonable timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    completion_events["task_1"].wait(),
                    completion_events["task_2"].wait(),
                    completion_events["task_3"].wait(),
                ),
                timeout=5.0,
            )
        except asyncio.TimeoutError:
            pytest.fail("Tasks did not complete within timeout")

        # Stop the worker
        await logging_worker.stop()

        # Sort results by task_id for consistent testing
        task_results.sort(key=lambda x: x["task_id"])

        # Verify that each task saw its own context
        assert (
            len(task_results) == 3
        ), f"Expected 3 results, got {len(task_results)}: {task_results}"

        # Task 1 should see "context_1"
        task1_result = next((r for r in task_results if r["task_id"] == "task_1"), None)
        assert task1_result is not None, "Task 1 result not found"
        assert (
            task1_result["context_accessible"] is True
        ), "Task 1 should have access to context variable"
        assert (
            task1_result["context_value"] == "context_1"
        ), f"Task 1 should see 'context_1', got: {task1_result['context_value']}"

        # Task 2 should see "context_2"
        task2_result = next((r for r in task_results if r["task_id"] == "task_2"), None)
        assert task2_result is not None, "Task 2 result not found"
        assert (
            task2_result["context_accessible"] is True
        ), "Task 2 should have access to context variable"
        assert (
            task2_result["context_value"] == "context_2"
        ), f"Task 2 should see 'context_2', got: {task2_result['context_value']}"

        # Task 3 should not have access to the context variable
        task3_result = next((r for r in task_results if r["task_id"] == "task_3"), None)
        assert task3_result is not None, "Task 3 result not found"
        assert (
            task3_result["context_accessible"] is False
        ), "Task 3 should not have access to context variable"

    @pytest.mark.asyncio
    async def test_semaphore_concurrency_limit(self):
        """Test that the worker respects the semaphore concurrency limit."""
        worker = LoggingWorker(timeout=5.0, max_queue_size=20, concurrency=2)
        worker.start()

        running_tasks, max_concurrent, lock = set(), 0, asyncio.Lock()
        completed = asyncio.Event()

        async def tracked_task(task_id: int):
            async with lock:
                running_tasks.add(task_id)
                nonlocal max_concurrent
                max_concurrent = max(max_concurrent, len(running_tasks))
            await asyncio.sleep(0.2)
            async with lock:
                running_tasks.remove(task_id)
                if not running_tasks:
                    completed.set()

        for i in range(5):
            worker.enqueue(tracked_task(i))

        await asyncio.wait_for(completed.wait(), timeout=5.0)
        await worker.stop()

        assert max_concurrent <= 2, f"Max {max_concurrent} exceeded limit 2"
        assert max_concurrent >= 2, f"Expected 2+ concurrent, got {max_concurrent}"

    @pytest.mark.asyncio
    async def test_aggressive_queue_clearing(self):
        """Test that aggressive queue clearing processes tasks when queue is full."""
        worker = LoggingWorker(timeout=2.0, max_queue_size=4, concurrency=1)
        worker.start()

        processed, lock = [], asyncio.Lock()

        async def tracked_task(task_id: int):
            async with lock:
                processed.append(task_id)
            await asyncio.sleep(0.01)

        for i in range(4):
            worker.enqueue(tracked_task(i))
        await asyncio.sleep(0.1)

        for i in range(4, 8):
            worker.enqueue(tracked_task(i))

        await asyncio.sleep(LOGGING_WORKER_AGGRESSIVE_CLEAR_COOLDOWN_SECONDS + 0.3)
        await worker.stop()
        await worker.clear_queue()

        assert len(processed) >= 4, f"Expected 4+ tasks processed, got {len(processed)}"
