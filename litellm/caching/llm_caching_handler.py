"""
Add the event loop to the cache key, to prevent event loop closed errors.
"""

import asyncio
from typing import Final

from .evicted_client_closer import EvictedClientCloser, default_evicted_client_closer
from .in_memory_cache import InMemoryCache


class LLMClientCache(InMemoryCache):
    """Cache for LLM HTTP clients (OpenAI, Azure, httpx, etc.).

    An evicted client is never closed on the spot: a request handed the client
    just before eviction is still using it, and closing it there raises
    ``RuntimeError: Cannot send a request, as the client has been closed.``

    Nor can eviction be left to rely on garbage collection. The SDK clients are
    reference cycles, so an evicted client and its open TCP connections survive
    until a generational collection runs. Instead a client litellm created is
    handed to ``EvictedClientCloser``, which closes it once a grace window has
    passed. Clients the caller supplied are left untouched.
    """

    def __init__(
        self,
        max_size_in_memory: int | None = 200,
        default_ttl: int | None = 600,
        max_size_per_item: int | None = 1024,
        evicted_client_closer: EvictedClientCloser | None = None,
    ) -> None:
        super().__init__(
            max_size_in_memory=max_size_in_memory,
            default_ttl=default_ttl,
            max_size_per_item=max_size_per_item,
        )
        self.evicted_client_closer = evicted_client_closer or default_evicted_client_closer

    def _remove_key(self, key: str) -> None:
        evicted: Final[object] = self.cache_dict.get(key)
        super()._remove_key(key)
        self.evicted_client_closer.schedule(evicted)
        self.evicted_client_closer.reap()

    def update_cache_key_with_event_loop(self, key):
        """
        Add the event loop to the cache key, to prevent event loop closed errors.
        If none, use the key as is.
        """
        try:
            event_loop: Final = asyncio.get_running_loop()
            stringified_event_loop: Final = str(id(event_loop))
            return f"{key}-{stringified_event_loop}"
        except RuntimeError:  # handle no current running event loop
            return key

    def set_cache(self, key: str, value: object, litellm_owned_client: bool = False, **kwargs):
        """``litellm_owned_client`` marks a client litellm built, so it may be closed once evicted."""
        if litellm_owned_client:
            self.evicted_client_closer.mark_owned(value)
        key = self.update_cache_key_with_event_loop(key)
        return super().set_cache(key, value, **kwargs)

    async def async_set_cache(self, key: str, value: object, litellm_owned_client: bool = False, **kwargs):
        if litellm_owned_client:
            self.evicted_client_closer.mark_owned(value)
        key = self.update_cache_key_with_event_loop(key)
        return await super().async_set_cache(key, value, **kwargs)

    def get_cache(self, key, **kwargs):
        key = self.update_cache_key_with_event_loop(key)
        self.evicted_client_closer.reap()

        return super().get_cache(key, **kwargs)

    async def async_get_cache(self, key, **kwargs):
        key = self.update_cache_key_with_event_loop(key)

        return await super().async_get_cache(key, **kwargs)
