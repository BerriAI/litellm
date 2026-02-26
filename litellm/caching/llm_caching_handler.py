"""
Add the event loop to the cache key, to prevent event loop closed errors.
"""

import asyncio

from .in_memory_cache import InMemoryCache


class LLMClientCache(InMemoryCache):
    def _remove_key(self, key: str) -> None:
        """
        Remove the key from cache WITHOUT closing the client.

        Closing clients on eviction is unsafe because other parts of the code
        may still hold references to the evicted client (e.g.,
        litellm.module_level_aclient stored in the module __dict__, or
        in-flight requests that obtained the client before eviction).

        Closing such clients causes RuntimeError("Cannot send a request, as
        the client has been closed.") for all subsequent or in-flight users
        of that client reference.

        Client cleanup is handled by:
        - AsyncHTTPHandler.__del__ / HTTPHandler.__del__ (GC-triggered)
        - atexit handler registered by register_async_client_cleanup()
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
