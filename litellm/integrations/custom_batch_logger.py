"""
Custom Logger that handles batching logic 

Use this if you want your logs to be stored in memory and flushed periodically.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Callable, Generic, List, Optional, TypeVar, cast

import litellm
from litellm._logging import verbose_logger
from litellm.constants import DEFAULT_MAX_QUEUE_SIZE, DEFAULT_QUEUE_OVERFLOW_STRATEGY
from litellm.integrations.custom_logger import CustomLogger

# Type of items stored in the log queue. Subclasses use CustomBatchLogger[TheirPayloadType].
T = TypeVar("T")


class CustomBatchLogger(CustomLogger, Generic[T]):
    """
    Base class for batch loggers. Subclass with a type parameter for typed queues:
    e.g. class LangsmithLogger(CustomBatchLogger[LangsmithQueueObject])
    """

    log_queue: BoundedQueue[T]

    def __init__(
        self,
        flush_lock: Optional[asyncio.Lock] = None,
        batch_size: Optional[int] = None,
        flush_interval: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Args:
            flush_lock (Optional[asyncio.Lock], optional): Lock to use when flushing the queue. Defaults to None. Only used for custom loggers that do batching
            max_queue_size (Optional[int], optional): Maximum queue size to prevent OOM. Defaults to DEFAULT_MAX_QUEUE_SIZE (1000). Set to None for unlimited (not recommended). Can also be set via DEFAULT_MAX_QUEUE_SIZE env var.
            queue_overflow_strategy (str): What to do when queue is full: "drop_oldest" (default), "drop_newest", "reject", or "force_flush". Can also be set via DEFAULT_QUEUE_OVERFLOW_STRATEGY env var.
        """
        
        # Use BoundedQueue instead of plain list - all existing append() calls automatically get protection
        # Pass flush callback for "force_flush" strategy. Cast for generic T (runtime is same class).
        self.log_queue = cast(
            BoundedQueue[T],
            BoundedQueue(
                max_size=DEFAULT_MAX_QUEUE_SIZE,
                overflow_strategy=DEFAULT_QUEUE_OVERFLOW_STRATEGY,
                flush_callback=lambda: asyncio.create_task(self.flush_queue()),
            ),
        )
        self.flush_interval = flush_interval or litellm.DEFAULT_FLUSH_INTERVAL_SECONDS
        self.batch_size: int = batch_size or litellm.DEFAULT_BATCH_SIZE
        self.last_flush_time = time.time()
        self.flush_lock = flush_lock

        super().__init__(**kwargs)

    async def periodic_flush(self):
        while True:
            await asyncio.sleep(self.flush_interval)
            verbose_logger.debug(
                f"CustomLogger periodic flush after {self.flush_interval} seconds"
            )
            await self.flush_queue()

    async def flush_queue(self):
        if self.flush_lock is None:
            return

        async with self.flush_lock:
            if self.log_queue:
                verbose_logger.debug(
                    "CustomLogger: Flushing batch of %s events", len(self.log_queue)
                )
                await self.async_send_batch()
                self.log_queue.clear()
                self.last_flush_time = time.time()

    async def async_send_batch(self, *args, **kwargs):
        pass


class BoundedQueue(Generic[T]):
    """
    A list-like queue with optional size limit to prevent OOM.
    Generic over the item type so subclasses get typed append/iterate.
    """

    def __init__(
        self,
        max_size: Optional[int] = None,
        overflow_strategy: str = "force_flush",
        flush_callback: Optional[Callable[[], Any]] = None,
    ):
        """
        Args:
            max_size: Maximum queue size. None = unlimited (backward compatible)
            overflow_strategy: What to do when queue is full:
                - "drop_oldest": Remove oldest item (FIFO)
                - "drop_newest": Reject new item (don't add)
                - "reject": Raise RuntimeError (fail fast)
                - "force_flush": Trigger flush callback to send logs, then add item - default
            flush_callback: Optional callback function to trigger flush (required for "force_flush" strategy)
        """
        self._items: List[T] = []
        self.max_size = max_size
        self.overflow_strategy = overflow_strategy
        self._dropped_count = 0
        self._flush_callback = flush_callback
        
        # Map strategy names to handler methods
        self._strategy_handlers = {
            "drop_oldest": self._handle_drop_oldest,
            "drop_newest": self._handle_drop_newest,
            "reject": self._handle_reject,
            "force_flush": self._handle_force_flush,
        }
        self._handler = self._strategy_handlers.get(overflow_strategy, self._handle_force_flush)
    
    def _handle_drop_oldest(self, item: T) -> None:
        """Handle overflow by removing oldest item (FIFO)."""
        self._items.pop(0)
        self._items.append(item)
        self._dropped_count += 1
        if self._dropped_count % 100 == 0:  # Log every 100 drops to avoid spam
            verbose_logger.warning(
                f"CustomBatchLogger: Queue full ({self.max_size}), dropped {self._dropped_count} oldest logs"
            )
    
    def _handle_drop_newest(self, item: T) -> None:
        """Handle overflow by rejecting new item."""
        self._dropped_count += 1
        if self._dropped_count % 100 == 0:
            verbose_logger.warning(
                f"CustomBatchLogger: Queue full ({self.max_size}), rejected {self._dropped_count} new logs"
            )
    
    def _handle_reject(self, item: T) -> None:
        """Handle overflow by raising RuntimeError."""
        raise RuntimeError(
            f"CustomBatchLogger: Queue full ({self.max_size}). "
            f"Cannot add more logs. Consider increasing max_queue_size or fixing flush issues."
        )
    
    def _handle_force_flush(self, item: T) -> None:
        """Handle overflow by triggering flush callback, then adding item."""
        if self._flush_callback is None:
            # Fallback to drop_oldest if no flush callback provided
            verbose_logger.warning(
                "CustomBatchLogger: force_flush strategy requires flush_callback, falling back to drop_oldest"
            )
            self._handle_drop_oldest(item)
            return
        
        # Trigger flush to make space
        try:
            self._flush_callback()
            verbose_logger.debug(
                f"CustomBatchLogger: Queue full ({self.max_size}), triggered force flush to prevent log loss"
            )
        except Exception as e:
            verbose_logger.warning(
                f"CustomBatchLogger: Error during force flush: {e}, falling back to drop_oldest"
            )
            self._handle_drop_oldest(item)
            return
        
        # Add item after flush is triggered (flush is async, but we've scheduled it)
        self._items.append(item)
    
    def append(self, item: T) -> None:
        """Append item with OOM protection."""
        # No limit - allow unlimited growth (backward compatible)
        if self.max_size is None:
            self._items.append(item)
            return
        
        # Check if queue is full
        if len(self._items) >= self.max_size:
            self._handler(item)
        else:
            # Queue has space - add normally
            self._items.append(item)
    
    def clear(self):
        """Clear all items."""
        self._items.clear()
    
    def __len__(self):
        """Return queue length."""
        return len(self._items)
    
    def __bool__(self):
        """Check if queue is non-empty."""
        return bool(self._items)
    
    def __iter__(self):
        """Iterate over items."""
        return iter(self._items)
    
    def __getitem__(self, index: int) -> T:
        """Get item by index."""
        return self._items[index]

    def __setitem__(self, index: int, value: T) -> None:
        """Set item by index."""
        self._items[index] = value
    
    def __delitem__(self, index):
        """Delete item by index."""
        del self._items[index]
    
    def __contains__(self, item):
        """Check if item is in queue."""
        return item in self._items
    
    def __repr__(self):
        """String representation."""
        return f"BoundedQueue(max_size={self.max_size}, size={len(self._items)}, dropped={self._dropped_count})"
    
    def extend(self, items: List[T]) -> None:
        """Extend queue with items."""
        for item in items:
            self.append(item)

    def pop(self, index: int = -1) -> T:
        """Pop item from queue."""
        return self._items.pop(index)

    def remove(self, item: T) -> None:
        """Remove item from queue."""
        self._items.remove(item)

    def index(self, item: T) -> int:
        """Get index of item."""
        return self._items.index(item)

    def count(self, item: T) -> int:
        """Count occurrences of item."""
        return self._items.count(item)
