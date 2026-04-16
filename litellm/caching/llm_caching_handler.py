"""
Add the event loop to the cache key, to prevent event loop closed errors.
"""

import asyncio
from typing import Any, Optional

from .in_memory_cache import InMemoryCache


def _client_is_closed(value: Any) -> bool:
    """Return True if the cached HTTP client has already been closed.

    A cached httpx / OpenAI SDK client becomes unusable once its underlying
    httpx connection pool is closed — subsequent ``send()`` calls raise
    ``RuntimeError: Cannot send a request, as the client has been closed``.
    That state can be reached without the cache doing anything itself:

      - a test or caller closes the client explicitly,
      - the OpenAI SDK wraps httpx and one of its methods closes the pool,
      - GC finalizes an AsyncHTTPHandler whose ``__del__`` schedules a close.

    After that, handing the same cached instance back out of the cache
    produces the "client has been closed" failure at request time.

    Check a few common shapes because different callers cache different
    wrappers:

      - ``AsyncHTTPHandler`` / ``HTTPHandler``: wraps httpx on ``.client``.
      - ``AsyncAzureOpenAI`` / ``AsyncOpenAI``: wraps httpx on ``._client``.
      - raw ``httpx.AsyncClient`` / ``httpx.Client``: exposes ``.is_closed``.
      - ``BaseLLMAIOHTTPHandler``: aiohttp-backed, so ``is_closed`` is not
        defined; treat as open.
    """
    for candidate in (
        value,
        getattr(value, "_client", None),
        getattr(value, "client", None),
    ):
        if candidate is None:
            continue
        try:
            if getattr(candidate, "is_closed", False):
                return True
        except Exception:
            # Defensive — an exotic wrapper shouldn't crash the cache read.
            continue
    return False


class LLMClientCache(InMemoryCache):
    """Cache for LLM HTTP clients (OpenAI, Azure, httpx, etc.).

    IMPORTANT: This cache intentionally does NOT close clients on eviction.
    Evicted clients may still be in use by in-flight requests. Closing them
    eagerly causes ``RuntimeError: Cannot send a request, as the client has
    been closed.`` errors in production after the TTL (1 hour) expires.

    Clients that are no longer referenced will be garbage-collected normally.
    For explicit shutdown cleanup, use ``close_litellm_async_clients()``.

    On READ, however, we do check whether the cached client is already closed
    (``is_closed`` is True on the inner httpx client). If so we evict the
    stale entry and return ``None`` — the caller will then build a fresh
    client via its normal factory path. Without this, a closed client can
    sit in the cache and be handed back to unrelated callers, producing
    mysterious "Cannot send a request, as the client has been closed"
    failures that are order-dependent and hard to reproduce.
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

    def _get_and_drop_if_closed(self, key: str, value: Any) -> Optional[Any]:
        """Return ``value`` unless its underlying HTTP client is closed.

        When we detect a closed client we evict it so subsequent reads for
        the same key don't keep hitting the same broken entry.
        """
        if value is None:
            return value
        if _client_is_closed(value):
            self._remove_key(key)
            return None
        return value

    def set_cache(self, key, value, **kwargs):
        key = self.update_cache_key_with_event_loop(key)
        return super().set_cache(key, value, **kwargs)

    async def async_set_cache(self, key, value, **kwargs):
        key = self.update_cache_key_with_event_loop(key)
        return await super().async_set_cache(key, value, **kwargs)

    def get_cache(self, key, **kwargs):
        resolved_key = self.update_cache_key_with_event_loop(key)
        value = super().get_cache(resolved_key, **kwargs)
        return self._get_and_drop_if_closed(resolved_key, value)

    async def async_get_cache(self, key, **kwargs):
        resolved_key = self.update_cache_key_with_event_loop(key)
        value = await super().async_get_cache(resolved_key, **kwargs)
        return self._get_and_drop_if_closed(resolved_key, value)
