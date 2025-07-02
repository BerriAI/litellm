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

    for key, client in cache_dict.items():
        if isinstance(client, BaseLLMAIOHTTPHandler) and hasattr(client, "close"):
            try:
                await client.close()
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
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule the cleanup coroutine
                loop.create_task(close_litellm_async_clients())
            else:
                # Run the cleanup coroutine
                loop.run_until_complete(close_litellm_async_clients())
        except Exception:
            # If we can't get an event loop or it's already closed, try creating a new one
            try:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(close_litellm_async_clients())
                loop.close()
            except Exception:
                # Silently ignore errors during cleanup
                pass

    atexit.register(cleanup_wrapper)
