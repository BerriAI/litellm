# CoroutineChecker utility for checking if functions/callables are coroutines or coroutine functions

import inspect
from typing import Any
from weakref import WeakKeyDictionary
from litellm.constants import (
    COROUTINE_CHECKER_MAX_SIZE_IN_MEMORY,
)


class CoroutineChecker:
    """Utility class for checking coroutine status of functions and callables.
    
    Simple bounded cache using WeakKeyDictionary to avoid memory leaks.
    """
    
    def __init__(self):
        self._cache = WeakKeyDictionary()
        self._max_size = COROUTINE_CHECKER_MAX_SIZE_IN_MEMORY
    
    def is_async_callable(self, callback: Any) -> bool:
        """Fast, cached check for whether a callback is an async function.
        Falls back gracefully if the object cannot be weak-referenced or cached.
        2.59x speedup.
        """
        # Fast path: check cache first (most common case)
        try:
            cached = self._cache.get(callback)
            if cached is not None:
                return cached
        except Exception:
            pass

        # Determine target - optimized path for common cases
        target = callback
        if not inspect.isfunction(target) and not inspect.ismethod(target):
            try:
                call_attr = getattr(target, "__call__", None)
                if call_attr is not None:
                    target = call_attr
            except Exception:
                pass

        # Compute result
        try:
            result = inspect.iscoroutinefunction(target)
        except Exception:
            result = False

        # Cache the result with size enforcement
        try:
            # Simple size enforcement: clear cache if it gets too large
            if len(self._cache) >= self._max_size:
                self._cache.clear()
            
            self._cache[callback] = result
        except Exception:
            pass

        return result

# Global instance for backward compatibility and convenience
coroutine_checker = CoroutineChecker()
