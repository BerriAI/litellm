"""
Utility functions for cleaning up async HTTP clients to prevent resource leaks.
"""

import asyncio
from collections.abc import Iterable
from typing import Final, Protocol


class _CachedClients(Protocol):
    def values(self) -> Iterable[object]: ...


class _ReportsClosed(Protocol):
    @property
    def is_closed(self) -> object: ...


async def close_litellm_async_clients():
    """
    Close all cached async HTTP clients to prevent resource leaks.

    This function iterates through all cached clients in litellm's in-memory cache
    and closes any aiohttp client sessions that are still open. Also closes the
    global base_llm_aiohttp_handler instance (issue #12443).
    """
    # Import here to avoid circular import
    import litellm
    from litellm.litellm_core_utils.core_helpers import attribute_or_none, has_attributes, require_attributes
    from litellm.llms.custom_httpx.aiohttp_handler import BaseLLMAIOHTTPHandler
    from litellm.llms.custom_httpx.stream_close import SupportsAclose

    cache_dict: Final[object] = getattr(litellm.in_memory_llm_clients_cache, "cache_dict", {})

    for handler in require_attributes(cache_dict, _CachedClients, "values").values():
        # Handle BaseLLMAIOHTTPHandler instances (aiohttp_openai provider)
        if isinstance(handler, BaseLLMAIOHTTPHandler) and hasattr(handler, "close"):
            try:
                await handler.close()
            except Exception:
                # Silently ignore errors during cleanup
                pass

        # Handle AsyncHTTPHandler instances (used by Gemini and other providers)
        elif hasattr(handler, "_client") or hasattr(handler, "client"):
            client = (
                attribute_or_none(handler, "_client")
                if hasattr(handler, "_client")
                else attribute_or_none(handler, "client")
            )
            transport = attribute_or_none(client, "_transport")
            # Check if the httpx client has an aiohttp transport
            if has_attributes(transport, SupportsAclose, "aclose"):
                try:
                    await transport.aclose()
                except Exception:
                    # Silently ignore errors during cleanup
                    pass
            # Also close the httpx client itself
            if (
                has_attributes(client, SupportsAclose, "aclose")
                and not require_attributes(client, _ReportsClosed, "is_closed").is_closed
            ):
                try:
                    await client.aclose()
                except Exception:
                    # Silently ignore errors during cleanup
                    pass

        # Handle any other objects with aclose method
        elif has_attributes(handler, SupportsAclose, "aclose"):
            try:
                await handler.aclose()
            except Exception:
                # Silently ignore errors during cleanup
                pass

    # Close the global base_llm_aiohttp_handler instance (issue #12443)
    # This is used by Gemini and other providers that use aiohttp
    if hasattr(litellm, "base_llm_aiohttp_handler"):
        base_handler: Final = getattr(litellm, "base_llm_aiohttp_handler", None)
        if isinstance(base_handler, BaseLLMAIOHTTPHandler) and hasattr(base_handler, "close"):
            try:
                await base_handler.close()
            except Exception:
                # Silently ignore errors during cleanup
                pass


def register_async_client_cleanup():
    """
    Register the async client cleanup function to run at exit.

    This ensures that all async HTTP clients are properly closed when the program exits.
    """
    import atexit

    def cleanup_wrapper():
        """
        Cleanup wrapper that creates a fresh event loop for atexit cleanup.

        At exit time, the main event loop is often already closed. Creating a new
        event loop ensures cleanup runs successfully (fixes issue #12443).
        """
        try:
            # Always create a fresh event loop at exit time
            # Don't use get_event_loop() - it may be closed or unavailable
            loop: Final = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(close_litellm_async_clients())
            finally:
                # Clean up the loop we created
                loop.close()
        except Exception:
            # Silently ignore errors during cleanup to avoid exit handler failures
            pass

    atexit.register(cleanup_wrapper)
