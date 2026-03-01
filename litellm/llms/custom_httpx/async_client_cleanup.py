"""
Utility functions for cleaning up async HTTP clients to prevent resource leaks.

Provides two cleanup strategies:
1. ``close_litellm_async_clients()`` — async cleanup, used when an event loop
   is available (call via ``await litellm.aclose()``).
2. ``_close_aiohttp_sessions_sync()`` — synchronous fallback that directly
   closes aiohttp connectors and marks sessions as closed.  Used by the
   ``atexit`` handler where the original event loop is already shut down.

The atexit handler alone is NOT sufficient for ``asyncio.run()`` usage because
``asyncio.run()`` closes the event loop *before* atexit handlers execute,
causing aiohttp to emit "Unclosed client session" warnings.  Users running
inside ``asyncio.run()`` should call ``await litellm.aclose()`` before
returning from their top-level coroutine.

See: https://github.com/BerriAI/litellm/issues/13251
"""

import asyncio
from typing import Any, List


def _collect_aiohttp_sessions() -> List[Any]:
    """
    Walk the LLM client cache and collect all live aiohttp ClientSession objects.

    Returns a list of ``aiohttp.ClientSession`` instances (may be empty).
    """
    import litellm

    sessions: List[Any] = []

    try:
        from aiohttp import ClientSession
    except ImportError:
        return sessions

    cache = getattr(litellm, "in_memory_llm_clients_cache", None)
    if cache is None:
        return sessions

    cache_dict = getattr(cache, "cache_dict", {})

    for handler in cache_dict.values():
        _extract_sessions_from_handler(handler, sessions, ClientSession)

    base_handler = getattr(litellm, "base_llm_aiohttp_handler", None)
    if base_handler is not None:
        _extract_sessions_from_handler(base_handler, sessions, ClientSession)

    return sessions


def _extract_sessions_from_handler(handler: Any, sessions: list, session_cls: type) -> None:
    """Extract aiohttp ClientSession objects from a handler (any type)."""
    # BaseLLMAIOHTTPHandler — has .client_session
    cs = getattr(handler, "client_session", None)
    if isinstance(cs, session_cls) and not cs.closed:
        sessions.append(cs)

    # AsyncHTTPHandler — has .client (httpx.AsyncClient) with ._transport
    httpx_client = getattr(handler, "client", None)
    if httpx_client is not None:
        transport = getattr(httpx_client, "_transport", None)
        if transport is not None:
            inner = getattr(transport, "client", None)
            if isinstance(inner, session_cls) and not inner.closed:
                sessions.append(inner)


def _close_aiohttp_sessions_sync() -> None:
    """
    Close aiohttp sessions **synchronously** by closing their underlying
    TCP connectors and marking them as closed.

    ``aiohttp.BaseConnector._close()`` is synchronous and safe to call from
    any thread or event loop state.  This avoids the need for a running
    event loop, which is unavailable in atexit handlers after
    ``asyncio.run()`` has returned.
    """
    for session in _collect_aiohttp_sessions():
        try:
            connector = getattr(session, "_connector", None)
            if connector is not None and not getattr(connector, "closed", True):
                # _close() is the synchronous core of connector.close().
                # It sets _closed=True, closes transports, and clears pools.
                # The returned waiters (for SSL shutdown) are intentionally
                # not awaited — acceptable during process shutdown.
                _close_fn = getattr(connector, "_close", None)
                if _close_fn is not None:
                    _close_fn()
                else:
                    # Fallback: directly set closed flag
                    connector._closed = True
            # Detach connector so session.closed returns True
            session._connector = None
        except Exception:
            pass


async def close_litellm_async_clients():
    """
    Close all cached async HTTP clients to prevent resource leaks.

    This function iterates through all cached clients in litellm's in-memory cache
    and closes any aiohttp client sessions that are still open. Also closes the
    global base_llm_aiohttp_handler instance (issue #12443).
    """
    # Import here to avoid circular import
    import litellm
    from litellm.llms.custom_httpx.aiohttp_handler import BaseLLMAIOHTTPHandler

    cache = getattr(litellm, "in_memory_llm_clients_cache", None)
    if cache is None:
        return

    cache_dict = getattr(cache, "cache_dict", {})

    for handler in list(cache_dict.values()):
        # Handle BaseLLMAIOHTTPHandler instances (aiohttp_openai provider)
        if isinstance(handler, BaseLLMAIOHTTPHandler) and hasattr(handler, "close"):
            try:
                await handler.close()
            except Exception:
                pass

        # Handle AsyncHTTPHandler instances (used by Gemini and other providers)
        elif hasattr(handler, 'client'):
            client = handler.client
            # Check if the httpx client has an aiohttp transport
            if hasattr(client, '_transport') and hasattr(client._transport, 'aclose'):
                try:
                    await client._transport.aclose()
                except Exception:
                    pass
            # Also close the httpx client itself
            if hasattr(client, 'aclose') and not client.is_closed:
                try:
                    await client.aclose()
                except Exception:
                    pass

        # Handle any other objects with aclose method
        elif hasattr(handler, 'aclose'):
            try:
                await handler.aclose()
            except Exception:
                pass

    # Close the global base_llm_aiohttp_handler instance (issue #12443)
    if hasattr(litellm, 'base_llm_aiohttp_handler'):
        base_handler = getattr(litellm, 'base_llm_aiohttp_handler', None)
        if isinstance(base_handler, BaseLLMAIOHTTPHandler) and hasattr(base_handler, 'close'):
            try:
                await base_handler.close()
            except Exception:
                pass


def register_async_client_cleanup():
    """
    Register the async client cleanup function to run at exit.

    This ensures that all async HTTP clients are properly closed when the program exits.
    """
    import atexit

    def cleanup_wrapper():
        """
        Cleanup wrapper that closes aiohttp sessions at exit time.

        Uses synchronous connector cleanup first (reliable, no event loop needed),
        then attempts async cleanup as a fallback for non-aiohttp transports.
        """
        # Synchronous cleanup — close aiohttp connectors directly.
        # This is the primary strategy because it works even when the
        # original event loop is already closed (the common case after
        # asyncio.run() returns).
        try:
            _close_aiohttp_sessions_sync()
        except Exception:
            pass

        # Async fallback — catches non-aiohttp transports and httpx clients.
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(close_litellm_async_clients())
            finally:
                loop.close()
        except Exception:
            pass

    atexit.register(cleanup_wrapper)
