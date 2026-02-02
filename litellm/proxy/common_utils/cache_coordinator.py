"""
Event-driven cache coordinator to prevent cache stampede.

Use this when many requests can miss the same cache key at once (e.g. after
expiry or restart). Without coordination, they would all run the expensive
load (DB query, API call) in parallel and overload the backend.

This module ensures only one request performs the load; the rest wait for a
signal and then read the freshly cached value. Reuse it for any cache-aside
pattern: global spend, feature flags, config, or other shared read-through data.
"""

import asyncio
import time
from typing import Any, Awaitable, Callable, Optional, Protocol, TypeVar

from litellm._logging import verbose_proxy_logger

T = TypeVar("T")


class AsyncCacheProtocol(Protocol):
    """Protocol for cache backends used by EventDrivenCacheCoordinator."""

    async def async_get_cache(self, key: str, **kwargs: Any) -> Any:
        ...

    async def async_set_cache(self, key: str, value: Any, **kwargs: Any) -> Any:
        ...


class EventDrivenCacheCoordinator:
    """
    Coordinates a single in-flight load per logical resource to prevent cache stampede.

    Pattern:
    - First request: loads data (e.g. DB query), caches it, then signals waiters.
    - Other requests: wait for the signal, then read from cache.

    Create one instance per resource (e.g. one for global spend, one for feature flags).
    """

    def __init__(self, log_prefix: str = "[CACHE]"):
        self._lock = asyncio.Lock()
        self._event: Optional[asyncio.Event] = None
        self._query_in_progress = False
        self._log_prefix = log_prefix

    async def _get_cached(
        self, cache_key: str, cache: AsyncCacheProtocol
    ) -> Optional[Any]:
        """Return value from cache if present, else None."""
        return await cache.async_get_cache(key=cache_key)

    def _log_cache_hit(self, value: T) -> None:
        if self._log_prefix:
            verbose_proxy_logger.debug(
                "%s Cache hit, value: %s", self._log_prefix, value
            )

    def _log_cache_miss(self) -> None:
        if self._log_prefix:
            verbose_proxy_logger.debug("%s Cache miss", self._log_prefix)

    async def _claim_role(self) -> Optional[asyncio.Event]:
        """
        Under lock: return event to wait on if load is in progress, else set us as loader and return None.
        """
        async with self._lock:
            if self._query_in_progress and self._event is not None:
                if self._log_prefix:
                    verbose_proxy_logger.debug(
                        "%s Load in flight, waiting for signal", self._log_prefix
                    )
                return self._event
            self._query_in_progress = True
            self._event = asyncio.Event()
            if self._log_prefix:
                verbose_proxy_logger.debug(
                    "%s Starting load (will signal others when done)",
                    self._log_prefix,
                )
            return None

    async def _wait_for_signal_and_get(
        self,
        event: asyncio.Event,
        cache_key: str,
        cache: AsyncCacheProtocol,
    ) -> Optional[T]:
        """Wait for loader to finish, then read from cache."""
        await event.wait()
        if self._log_prefix:
            verbose_proxy_logger.debug(
                "%s Signal received, reading from cache", self._log_prefix
            )
        value: Optional[T] = await cache.async_get_cache(key=cache_key)
        if value is not None and self._log_prefix:
            verbose_proxy_logger.debug(
                "%s Cache filled by other request, value: %s",
                self._log_prefix,
                value,
            )
        elif value is None and self._log_prefix:
            verbose_proxy_logger.debug(
                "%s Signal received but cache still empty", self._log_prefix
            )
        return value

    async def _load_and_cache(
        self,
        cache_key: str,
        cache: AsyncCacheProtocol,
        load_fn: Callable[[], Awaitable[T]],
    ) -> Optional[T]:
        """Double-check cache, run load_fn, set cache, return value. Caller must call _signal_done in finally."""
        value = await cache.async_get_cache(key=cache_key)
        if value is not None:
            if self._log_prefix:
                verbose_proxy_logger.debug(
                    "%s Cache filled while acquiring lock, value: %s",
                    self._log_prefix,
                    value,
                )
            return value

        if self._log_prefix:
            verbose_proxy_logger.debug("%s Running load", self._log_prefix)
        start = time.perf_counter()
        value = await load_fn()
        elapsed_ms = (time.perf_counter() - start) * 1000
        if self._log_prefix:
            verbose_proxy_logger.debug(
                "%s Load completed in %.2fms, result: %s",
                self._log_prefix,
                elapsed_ms,
                value,
            )

        await cache.async_set_cache(key=cache_key, value=value)
        if self._log_prefix:
            verbose_proxy_logger.debug("%s Result cached", self._log_prefix)
        return value

    async def _signal_done(self) -> None:
        """Reset loader state and signal all waiters."""
        async with self._lock:
            self._query_in_progress = False
            if self._event is not None:
                if self._log_prefix:
                    verbose_proxy_logger.debug(
                        "%s Signaling all waiting requests", self._log_prefix
                    )
                self._event.set()
                self._event = None

    async def get_or_load(
        self,
        cache_key: str,
        cache: AsyncCacheProtocol,
        load_fn: Callable[[], Awaitable[T]],
    ) -> Optional[T]:
        """
        Return cached value or load it once and signal waiters.

        - cache_key: Key to read/write in the cache.
        - cache: Object with async_get_cache(key) and async_set_cache(key, value).
        - load_fn: Async callable that performs the load (e.g. DB query). No args.
                   Return value is cached and returned. If it raises, waiters are
                   still signaled so they can retry or handle empty cache.

        Returns the value from cache or from load_fn, or None if load failed or
        cache was still empty after waiting.
        """
        value = await self._get_cached(cache_key, cache)
        if value is not None:
            self._log_cache_hit(value)
            return value

        self._log_cache_miss()
        event_to_wait = await self._claim_role()

        if event_to_wait is not None:
            return await self._wait_for_signal_and_get(
                event_to_wait, cache_key, cache
            )

        try:
            result = await self._load_and_cache(cache_key, cache, load_fn)
            return result
        finally:
            await self._signal_done()
