"""@private"""

import atexit
import logging
from datetime import datetime
from queue import Empty, Queue
from threading import Thread
from typing import Dict, List, Optional, Set

from langfuse.model import PromptClient

DEFAULT_PROMPT_CACHE_TTL_SECONDS = 60

DEFAULT_PROMPT_CACHE_REFRESH_WORKERS = 1


class PromptCacheItem:
    def __init__(self, prompt: PromptClient, ttl_seconds: int):
        self.value = prompt
        self._expiry = ttl_seconds + self.get_epoch_seconds()

    def is_expired(self) -> bool:
        return self.get_epoch_seconds() > self._expiry

    @staticmethod
    def get_epoch_seconds() -> int:
        return int(datetime.now().timestamp())


class PromptCacheRefreshConsumer(Thread):
    _log = logging.getLogger("langfuse")
    _queue: Queue
    _identifier: int
    running: bool = True

    def __init__(self, queue: Queue, identifier: int):
        super().__init__()
        self.daemon = True
        self._queue = queue
        self._identifier = identifier

    def run(self):
        while self.running:
            try:
                task = self._queue.get(timeout=1)
                self._log.debug(
                    f"PromptCacheRefreshConsumer processing task, {self._identifier}"
                )
                try:
                    task()
                # Task failed, but we still consider it processed
                except Exception as e:
                    self._log.warning(
                        f"PromptCacheRefreshConsumer encountered an error, cache was not refreshed: {self._identifier}, {e}"
                    )

                self._queue.task_done()
            except Empty:
                pass

    def pause(self):
        """Pause the consumer."""
        self.running = False


class PromptCacheTaskManager(object):
    _log = logging.getLogger("langfuse")
    _consumers: List[PromptCacheRefreshConsumer]
    _threads: int
    _queue: Queue
    _processing_keys: Set[str]

    def __init__(self, threads: int = 1):
        self._queue = Queue()
        self._consumers = []
        self._threads = threads
        self._processing_keys = set()

        for i in range(self._threads):
            consumer = PromptCacheRefreshConsumer(self._queue, i)
            consumer.start()
            self._consumers.append(consumer)

        atexit.register(self.shutdown)

    def add_task(self, key: str, task):
        if key not in self._processing_keys:
            self._log.debug(f"Adding prompt cache refresh task for key: {key}")
            self._processing_keys.add(key)
            wrapped_task = self._wrap_task(key, task)
            self._queue.put((wrapped_task))
        else:
            self._log.debug(
                f"Prompt cache refresh task already submitted for key: {key}"
            )

    def active_tasks(self) -> int:
        return len(self._processing_keys)

    def _wrap_task(self, key: str, task):
        def wrapped():
            self._log.debug(f"Refreshing prompt cache for key: {key}")
            try:
                task()
            finally:
                self._processing_keys.remove(key)
                self._log.debug(f"Refreshed prompt cache for key: {key}")

        return wrapped

    def shutdown(self):
        self._log.debug(
            f"Shutting down prompt refresh task manager, {len(self._consumers)} consumers,..."
        )

        atexit.unregister(self.shutdown)

        for consumer in self._consumers:
            consumer.pause()

        for consumer in self._consumers:
            try:
                consumer.join()
            except RuntimeError:
                # consumer thread has not started
                pass

        self._log.debug("Shutdown of prompt refresh task manager completed.")


class PromptCache:
    _cache: Dict[str, PromptCacheItem]

    _task_manager: PromptCacheTaskManager
    """Task manager for refreshing cache"""

    _log = logging.getLogger("langfuse")

    def __init__(
        self, max_prompt_refresh_workers: int = DEFAULT_PROMPT_CACHE_REFRESH_WORKERS
    ):
        self._cache = {}
        self._task_manager = PromptCacheTaskManager(threads=max_prompt_refresh_workers)
        self._log.debug("Prompt cache initialized.")

    def get(self, key: str) -> Optional[PromptCacheItem]:
        return self._cache.get(key, None)

    def set(self, key: str, value: PromptClient, ttl_seconds: Optional[int]):
        if ttl_seconds is None:
            ttl_seconds = DEFAULT_PROMPT_CACHE_TTL_SECONDS

        self._cache[key] = PromptCacheItem(value, ttl_seconds)

    def invalidate(self, prompt_name: str):
        """Invalidate all cached prompts with the given prompt name."""
        for key in list(self._cache):
            if key.startswith(prompt_name):
                del self._cache[key]

    def add_refresh_prompt_task(self, key: str, fetch_func):
        self._log.debug(f"Submitting refresh task for key: {key}")
        self._task_manager.add_task(key, fetch_func)

    @staticmethod
    def generate_cache_key(
        name: str, *, version: Optional[int], label: Optional[str]
    ) -> str:
        parts = [name]

        if version is not None:
            parts.append(f"version:{version}")

        elif label is not None:
            parts.append(f"label:{label}")

        else:
            # Default to production labeled prompt
            parts.append("label:production")

        return "-".join(parts)
