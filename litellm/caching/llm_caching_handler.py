"""
Add the event loop to the cache key, to prevent event loop closed errors.
"""

import asyncio

from .in_memory_cache import InMemoryCache


class LLMClientCache(InMemoryCache):
    """Cache for LLM HTTP clients (OpenAI, Azure, httpx, etc.).

    IMPORTANT: This cache intentionally does NOT close clients on eviction.
    Evicted clients may still be in use by in-flight requests. Closing them
    eagerly causes ``RuntimeError: Cannot send a request, as the client has
    been closed.`` errors in production after the TTL (1 hour) expires.

    Clients that are no longer referenced will be garbage-collected normally.
    For explicit shutdown cleanup, use ``close_litellm_async_clients()``.
    """

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
