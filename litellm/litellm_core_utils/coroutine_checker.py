# CoroutineChecker utility for checking if functions/callables are coroutines or coroutine functions

import inspect
from typing import Any
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.constants import (
    COROUTINE_CHECKER_MAX_SIZE_IN_MEMORY,
    COROUTINE_CHECKER_DEFAULT_TTL,
    COROUTINE_CHECKER_MAX_SIZE_PER_ITEM,
)


class CoroutineChecker:
    """Utility class for checking coroutine status of functions and callables.
    
    Provides cached checking for better performance when repeatedly checking
    the same callables.
    """
    
    def __init__(self):
        self._cache = InMemoryCache(
            max_size_in_memory=COROUTINE_CHECKER_MAX_SIZE_IN_MEMORY,
            default_ttl=COROUTINE_CHECKER_DEFAULT_TTL,
            max_size_per_item=COROUTINE_CHECKER_MAX_SIZE_PER_ITEM
        )
    
    def is_async_callable(self, callback: Any) -> bool:
        """Fast, cached check for whether a callback is an async function."""
        try:
            cache_key = str(callback)
        except Exception:
            cache_key = f"callback_{id(callback)}"
        
        cached_result = self._cache.get_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        target = callback
        if not inspect.isfunction(target) and not inspect.ismethod(target):
            try:
                call_attr = getattr(target, "__call__", None)
                if call_attr is not None:
                    target = call_attr
            except Exception:
                pass

        try:
            result = inspect.iscoroutinefunction(target)
        except Exception:
            result = False

        self._cache.set_cache(cache_key, result)
        return result
    

# Global instance for backward compatibility and convenience
coroutine_checker = CoroutineChecker()
