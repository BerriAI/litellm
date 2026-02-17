"""
Add the event loop to the cache key, to prevent event loop closed errors.
"""

import asyncio

from litellm._logging import verbose_logger

from .in_memory_cache import InMemoryCache


def _close_http_client_on_evict(client) -> None:
    """
    Callback invoked when an HTTP client is evicted from the LLMClientCache.

    Ensures the underlying connection pool is closed deterministically instead
    of relying on __del__ / garbage collection, which is unreliable for async
    resources and can cause connection pool leaks under high load.
    """
    close_fn = getattr(client, "close", None)
    if close_fn is None:
        return

    if asyncio.iscoroutinefunction(close_fn):
        # Schedule async close on the running event loop if available
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(close_fn())
        except RuntimeError:
            # No running loop – best-effort sync fallback
            pass
    else:
        try:
            close_fn()
        except Exception:
            pass


class LLMClientCache(InMemoryCache):
    def __init__(self, **kwargs):
        super().__init__(on_evict=_close_http_client_on_evict, **kwargs)

    def update_cache_key_with_event_loop(self, key):
        """
        Add the event loop to the cache key, to prevent event loop closed errors.
        If none, use the key as is.
        """
        try:
            event_loop = asyncio.get_running_loop()
            stringified_event_loop = str(id(event_loop))
            return f"{key}-{stringified_event_loop}"
        except RuntimeError:  # handle no current running event loop
            return key

    def set_cache(self, key, value, **kwargs):
        key = self.update_cache_key_with_event_loop(key)
        return super().set_cache(key, value, **kwargs)

    async def async_set_cache(self, key, value, **kwargs):
        key = self.update_cache_key_with_event_loop(key)
        return await super().async_set_cache(key, value, **kwargs)

    def get_cache(self, key, **kwargs):
        key = self.update_cache_key_with_event_loop(key)

        return super().get_cache(key, **kwargs)

    async def async_get_cache(self, key, **kwargs):
        key = self.update_cache_key_with_event_loop(key)

        return await super().async_get_cache(key, **kwargs)
