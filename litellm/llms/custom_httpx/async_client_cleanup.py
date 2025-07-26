"""
Utility functions for cleaning up async HTTP clients to prevent resource leaks.
"""
import asyncio


async def close_litellm_async_clients():
    """
    Close all cached async HTTP clients to prevent resource leaks.

    This function iterates through all cached clients in litellm's in-memory cache
    and closes any aiohttp client sessions that are still open.
    """
    # Import here to avoid circular import
    import litellm
    from litellm.llms.custom_httpx.aiohttp_handler import BaseLLMAIOHTTPHandler

    cache_dict = getattr(litellm.in_memory_llm_clients_cache, "cache_dict", {})

    for key, handler in cache_dict.items():
        # Handle BaseLLMAIOHTTPHandler instances (aiohttp_openai provider)
        if isinstance(handler, BaseLLMAIOHTTPHandler) and hasattr(handler, "close"):
            try:
                await handler.close()
            except Exception:
                # Silently ignore errors during cleanup
                pass
        
        # Handle AsyncHTTPHandler instances (used by Gemini and other providers)
        elif hasattr(handler, 'client'):
            client = handler.client
            # Check if the httpx client has an aiohttp transport
            if hasattr(client, '_transport') and hasattr(client._transport, 'aclose'):
                try:
                    await client._transport.aclose()
                except Exception:
                    # Silently ignore errors during cleanup
                    pass
            # Also close the httpx client itself
            if hasattr(client, 'aclose') and not client.is_closed:
                try:
                    await client.aclose()
                except Exception:
                    # Silently ignore errors during cleanup
                    pass
        
        # Handle any other objects with aclose method
        elif hasattr(handler, 'aclose'):
            try:
                await handler.aclose()
            except Exception:
                # Silently ignore errors during cleanup
                pass


def _sync_close_clients():
    """
    Synchronous version of client cleanup for use in atexit handlers.
    
    This function attempts to close clients synchronously where possible,
    avoiding the issues with running async code in atexit handlers.
    """
    # Import here to avoid circular import
    import litellm
    from litellm.llms.custom_httpx.aiohttp_handler import BaseLLMAIOHTTPHandler

    cache_dict = getattr(litellm.in_memory_llm_clients_cache, "cache_dict", {})

    for key, handler in cache_dict.items():
        try:
            # Handle BaseLLMAIOHTTPHandler instances (aiohttp_openai provider)
            if isinstance(handler, BaseLLMAIOHTTPHandler):
                # For aiohttp handlers, we can try to close synchronously if possible
                if hasattr(handler, '_session') and handler._session and not handler._session.closed:
                    # Try to close the session synchronously
                    # Note: This might not work in all cases, but it's better than nothing
                    try:
                        if hasattr(handler._session, 'close'):
                            # aiohttp sessions have a close() method that can be called synchronously
                            # but it returns a coroutine, so we can't use it here
                            pass
                    except Exception:
                        pass
            
            # Handle AsyncHTTPHandler instances (used by Gemini and other providers)
            elif hasattr(handler, 'client'):
                client = handler.client
                # For httpx clients, try to close synchronously if possible
                if hasattr(client, 'close') and not client.is_closed:
                    try:
                        client.close()  # httpx has a synchronous close method
                    except Exception:
                        pass
            
            # For other handlers, we can't safely close them synchronously
            # so we just skip them to avoid warnings
            
        except Exception:
            # Silently ignore errors during cleanup
            pass


def register_async_client_cleanup():
    """
    Register the async client cleanup function to run at exit.

    This ensures that all async HTTP clients are properly closed when the program exits.
    Note: Uses synchronous cleanup in atexit handler to avoid issues with async code.
    """
    import atexit

    def cleanup_wrapper():
        try:
            # Use synchronous cleanup to avoid async issues in atexit handlers
            _sync_close_clients()
        except Exception:
            # Silently ignore errors during cleanup
            pass

    atexit.register(cleanup_wrapper)
