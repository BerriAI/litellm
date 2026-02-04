"""
Add the event loop to the cache key, to prevent event loop closed errors.

Also ensures proper cleanup of HTTP clients when they expire from the cache
to prevent idle database/HTTP connections from accumulating.

Relevant Issue: https://github.com/BerriAI/litellm/issues/19921
"""

import asyncio
from typing import TYPE_CHECKING

from litellm._logging import verbose_logger

from .in_memory_cache import InMemoryCache

if TYPE_CHECKING:
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler


class LLMClientCache(InMemoryCache):
    """
    Cache for LLM HTTP clients (httpx AsyncClient/Client).

    Ensures proper cleanup of HTTP clients when they expire from the cache
    to prevent idle connections from accumulating.

    Without proper cleanup, expired httpx clients leave connections open/idle
    until they timeout on the server side, which can exhaust connection pools.
    """

    def _remove_key(self, key: str) -> None:
        """
        Override _remove_key to ensure proper cleanup of HTTP clients.

        When httpx clients expire from the cache, we need to close them
        to release their underlying connections. Without this, connections
        remain idle until they timeout on the server side.

        This prevents the "idle connections" issue reported in:
        https://github.com/BerriAI/litellm/issues/19921
        """
        if key not in self.cache_dict:
            return super()._remove_key(key)

        cached_value = self.cache_dict[key]

        # Check if it's an HTTP handler that needs cleanup
        try:
            from litellm.llms.custom_httpx.http_handler import (
                AsyncHTTPHandler,
                HTTPHandler,
            )

            if isinstance(cached_value, AsyncHTTPHandler):
                # AsyncHTTPHandler wraps httpx.AsyncClient
                if hasattr(cached_value, "client") and cached_value.client is not None:
                    try:
                        # Schedule the async close in the event loop
                        loop = asyncio.get_running_loop()
                        loop.create_task(self._async_close_client(cached_value))
                        verbose_logger.debug(
                            f"LLMClientCache: Scheduled async close for expired AsyncHTTPHandler (key={key})"
                        )
                    except RuntimeError:
                        # No running event loop - try sync close if possible
                        verbose_logger.debug(
                            f"LLMClientCache: No event loop for async close, client will be garbage collected (key={key})"
                        )

            elif isinstance(cached_value, HTTPHandler):
                # HTTPHandler wraps httpx.Client (sync)
                if hasattr(cached_value, "client") and cached_value.client is not None:
                    try:
                        cached_value.close()
                        verbose_logger.debug(
                            f"LLMClientCache: Closed expired HTTPHandler (key={key})"
                        )
                    except Exception as e:
                        verbose_logger.debug(
                            f"LLMClientCache: Error closing HTTPHandler (key={key}): {e}"
                        )

        except ImportError:
            # http_handler module not available - skip cleanup
            pass
        except Exception as e:
            verbose_logger.debug(
                f"LLMClientCache: Error during client cleanup (key={key}): {e}"
            )

        # Call parent class to remove key from cache
        return super()._remove_key(key)

    async def _async_close_client(self, handler: "AsyncHTTPHandler") -> None:
        """
        Safely close an AsyncHTTPHandler's underlying httpx.AsyncClient.
        """
        try:
            await handler.close()
        except Exception as e:
            verbose_logger.debug(f"LLMClientCache: Error in async client close: {e}")

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
