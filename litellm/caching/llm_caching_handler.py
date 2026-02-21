"""
Add the event loop to the cache key, to prevent event loop closed errors.
"""

import asyncio

from .in_memory_cache import InMemoryCache


class LLMClientCache(InMemoryCache):
    def _remove_key(self, key: str) -> None:
        """Close async clients before evicting them to prevent connection pool leaks."""
        value = self.cache_dict.get(key)
        super()._remove_key(key)
        if value is not None:
            close_fn = getattr(value, "aclose", None) or getattr(
                value, "close", None
            )
            if close_fn and asyncio.iscoroutinefunction(close_fn):
                try:
                    asyncio.get_running_loop().create_task(close_fn())
                except RuntimeError:
                    pass
            elif close_fn and callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass

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
