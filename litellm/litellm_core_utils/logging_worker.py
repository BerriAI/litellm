import asyncio
import contextlib
from typing import Coroutine, Optional

from litellm._logging import verbose_logger


class LoggingWorker:
    """
    A simple, async logging worker that processes log coroutines in the background.
    Designed to be best-effort with bounded queues to prevent backpressure.
    """
    MAX_QUEUE_SIZE = 50_000
    MAX_TIMEOUT = 20.0
    
    def __init__(self, timeout: float = MAX_TIMEOUT, max_queue_size: int = MAX_QUEUE_SIZE):
        self.timeout = timeout
        self.max_queue_size = max_queue_size
        self._queue: Optional[asyncio.Queue] = None
        self._worker_task: Optional[asyncio.Task] = None
    
    def _ensure_queue(self) -> None:
        """Initialize the queue if it doesn't exist."""
        if self._queue is None:
            self._queue = asyncio.Queue(maxsize=self.max_queue_size)
    
    def start(self) -> None:
        """Start the logging worker. Idempotent - safe to call multiple times."""
        self._ensure_queue()
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker_loop())
    
    async def _worker_loop(self) -> None:
        """Main worker loop that processes log coroutines sequentially."""
        try:
            if self._queue is None:
                return
            
            while True:
                # Process one coroutine at a time to keep event loop load predictable
                coroutine = await self._queue.get()
                try:
                    await asyncio.wait_for(coroutine, timeout=self.timeout)
                except Exception as e:
                    verbose_logger.exception(f"LoggingWorker error: {e}")
                    pass
                finally:
                    self._queue.task_done()
                    
        except asyncio.CancelledError as e:
            verbose_logger.exception(f"LoggingWorker cancelled: {e}")
            pass
    
    def enqueue(self, coroutine: Coroutine) -> None:
        """
        Add a coroutine to the logging queue. 
        Hot path: never blocks, drops logs if queue is full.
        """
        if self._queue is None:
            return
        
        try:
            self._queue.put_nowait(coroutine)
            print("LoggingWorker current queue size==>", self._queue.qsize())
        except asyncio.QueueFull as e:
            verbose_logger.exception(f"LoggingWorker queue is full: {e}")
            # Drop logs on overload to protect request throughput
            pass
    
    def ensure_initialized_and_enqueue(self, async_coroutine: Coroutine):
        """
        Ensure the logging worker is initialized and enqueue the coroutine.
        """
        self.start()
        self.enqueue(async_coroutine)
    
    async def stop(self) -> None:
        """Stop the logging worker and clean up resources."""
        if self._worker_task:
            self._worker_task.cancel()
            with contextlib.suppress(Exception):
                await self._worker_task
            self._worker_task = None
    
    async def flush(self) -> None:
        """Flush the logging queue."""
        if self._queue is None:
            return
        while not self._queue.empty():
            await self._queue.join()


# Global instance for backward compatibility
GLOBAL_LOGGING_WORKER = LoggingWorker()

