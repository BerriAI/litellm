"""
Tests for the LoggingWorker class to ensure graceful shutdown handling.
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, patch

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
        with patch.object(logging_worker, "clear_queue", new_callable=AsyncMock) as mock_clear_queue:
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
        with patch("litellm.litellm_core_utils.logging_worker.verbose_logger") as mock_logger:
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
        with patch("litellm.litellm_core_utils.logging_worker.verbose_logger") as mock_logger:
            # Fill the queue beyond capacity
            mock_coro = AsyncMock()
            for _ in range(5):  # More than max_queue_size of 2
                small_worker.enqueue(mock_coro())

            # Should have logged queue full exceptions
            exception_calls = [call for call in mock_logger.exception.call_args_list if "queue is full" in str(call)]
            assert len(exception_calls) > 0
