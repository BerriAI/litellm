"""
Add the event loop to the cache key, to prevent event loop closed errors.
"""

import asyncio
import contextlib
import inspect
from typing import Final

from .evicted_client_closer import (
    EvictedClientCloser,
    close_function,
    default_evicted_client_closer,
    running_loop_token,
)
from .in_memory_cache import InMemoryCache


def _loop_scoped(key: str, token: int | None) -> str:
    return f"{key}-{token}" if token is not None else key


async def _aclose_quietly(client: object) -> None:
    close_fn: Final = close_function(client)
    if close_fn is None:
        return
    with contextlib.suppress(Exception):
        closing: Final = close_fn()
        if inspect.isawaitable(closing):
            await closing


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

    A loop that is about to close calls ``aclose_clients_for_running_loop``,
    since nothing cached under it can be used afterwards.
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
        self._loop_token_by_key: dict[str, int] = {}  # mutable-ok: tracks which loop each scoped key belongs to

    def _cached(self, key: str) -> object:
        return self.cache_dict.get(key)

    def _drop(self, key: str) -> object:
        client: Final = self._cached(key)
        self._loop_token_by_key.pop(key, None)
        InMemoryCache._remove_key(self, key)
        return client

    def _remove_key(self, key: str) -> None:
        self.evicted_client_closer.schedule(self._drop(key))
        self.evicted_client_closer.reap()

    def update_cache_key_with_event_loop(self, key: str) -> str:
        """
        Add the event loop to the cache key, to prevent event loop closed errors.
        If none, use the key as is.
        """
        return _loop_scoped(key, running_loop_token())

    def _scoped_key(self, key: str, loop_bound: bool) -> str:
        return self.update_cache_key_with_event_loop(key) if loop_bound else key

    def _key_for_set(self, key: str, value: object, litellm_owned_client: bool, loop_bound: bool) -> str:
        if litellm_owned_client:
            self.evicted_client_closer.mark_owned(value)
        token: Final = running_loop_token() if loop_bound else None
        scoped: Final = _loop_scoped(key, token)
        if token is not None:
            self._loop_token_by_key[scoped] = token
        return scoped

    def set_cache(self, key: str, value: object, litellm_owned_client: bool = False, loop_bound: bool = True, **kwargs):
        """``litellm_owned_client`` marks a client litellm built, so it may be closed once evicted."""
        return super().set_cache(self._key_for_set(key, value, litellm_owned_client, loop_bound), value, **kwargs)

    async def async_set_cache(self, key: str, value: object, litellm_owned_client: bool = False, **kwargs):
        scoped: Final = self._key_for_set(key, value, litellm_owned_client, loop_bound=True)
        return await super().async_set_cache(scoped, value, **kwargs)

    async def aclose_clients_for_running_loop(self) -> None:
        """Drop every client cached under the running loop and close the litellm-owned ones."""
        token: Final = running_loop_token()
        if token is None:
            return
        keys: Final = tuple(key for key, owner in tuple(self._loop_token_by_key.items()) if owner == token)
        owned: Final = tuple(
            client for client in (self._drop(key) for key in keys) if self.evicted_client_closer.is_owned(client)
        )
        if owned:
            _ = await asyncio.gather(*(_aclose_quietly(client) for client in owned))

    def get_cache(self, key: str, loop_bound: bool = True, **kwargs):
        scoped_key: Final = self._scoped_key(key, loop_bound)
        self.evicted_client_closer.reap()

        return super().get_cache(scoped_key, **kwargs)

    async def async_get_cache(self, key: str, **kwargs):
        key = self.update_cache_key_with_event_loop(key)

        return await super().async_get_cache(key, **kwargs)
