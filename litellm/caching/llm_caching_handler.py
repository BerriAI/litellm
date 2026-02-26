"""
Add the event loop to the cache key, to prevent event loop closed errors.
"""

import asyncio

from .in_memory_cache import InMemoryCache


class LLMClientCache(InMemoryCache):
    def _remove_key(self, key: str) -> None:
        """Remove key from cache WITHOUT closing the client.

        Callers of get_async_httpx_client / _get_httpx_client hold direct
        references to the returned client objects (e.g. litellm.module_level_aclient,
        streaming handlers, passthrough request handlers).  If we close the
        underlying httpx client here, those callers will receive:

            RuntimeError: Cannot send a request, as the client has been closed.

        Instead we simply drop the cache's own reference.  When all external
        references are released the normal Python GC cycle will finalize the
        transport / connection-pool resources.
        """
        super()._remove_key(key)

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
