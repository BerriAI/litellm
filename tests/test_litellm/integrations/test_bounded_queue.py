"""
Test BoundedQueue OOM protection through CustomBatchLogger (realistic integration test).
"""

import asyncio
import unittest

from litellm.integrations.custom_batch_logger import CustomBatchLogger


class TestBoundedQueueWithCustomBatchLogger(unittest.TestCase):
    """Test BoundedQueue behavior through CustomBatchLogger."""

    def test_force_flush_on_queue_overflow(self):
        """Test that force_flush strategy triggers flush when queue is full."""
        async def run_test():
            flush_called = []
            
            async def mock_async_send_batch(*args, **kwargs):
                """Track that flush was called."""
                flush_called.append(True)
            
            # Create logger with small queue for testing
            logger = CustomBatchLogger(batch_size=10, flush_interval=60)
            logger.async_send_batch = mock_async_send_batch
            logger.flush_lock = asyncio.Lock()
            
            # Override max_size to small value for testing
            original_max_size = logger.log_queue.max_size
            logger.log_queue.max_size = 3
            
            # Fill queue to max
            logger.log_queue.append("item1")
            logger.log_queue.append("item2")
            logger.log_queue.append("item3")
            self.assertEqual(len(logger.log_queue), 3)
            
            # Add 4th item - should trigger force_flush (default strategy)
            logger.log_queue.append("item4")
            
            # Item should be added immediately (flush is async)
            self.assertEqual(len(logger.log_queue), 4)
            
            # Wait for async flush task to run
            await asyncio.sleep(0.1)
            
            
            # Flush should have been called
            self.assertTrue(len(flush_called) > 0)
            
            # Queue should be cleared after flush
            self.assertEqual(len(logger.log_queue), 0)
            
            # Restore original max_size
            logger.log_queue.max_size = original_max_size
        
        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
