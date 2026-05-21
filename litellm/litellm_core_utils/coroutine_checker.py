# CoroutineChecker utility for checking if functions/callables are coroutines or coroutine functions

import inspect
from typing import Any, Dict
from weakref import WeakKeyDictionary
from litellm.constants import (
    COROUTINE_CHECKER_MAX_SIZE_IN_MEMORY,
)


class CoroutineChecker:
    """Utility class for checking coroutine status of functions and callables.

    Uses WeakKeyDictionary for callable objects and a bounded dict for hashable
    objects (e.g. strings) that cannot be weak-referenced.
    """

    def __init__(self):
        self._cache = WeakKeyDictionary()
        self._hashable_cache: Dict[Any, bool] = {}
        self._max_size = COROUTINE_CHECKER_MAX_SIZE_IN_MEMORY

    def _get_hashable_cached(self, callback: Any) -> Any:
        try:
            hash(callback)
        except TypeError:
            return None
        return self._hashable_cache.get(callback)

    def _set_hashable_cached(self, callback: Any, result: bool) -> None:
        try:
            hash(callback)
        except TypeError:
            return

        if len(self._hashable_cache) >= self._max_size:
            self._hashable_cache.clear()
        self._hashable_cache[callback] = result

    def is_async_callable(self, callback: Any) -> bool:
        """Fast, cached check for whether a callback is an async function.
        Falls back gracefully if the object cannot be weak-referenced or cached.
        """
        # String callback names (e.g. "langfuse") are never async callables.
        if isinstance(callback, str):
            return False

        hashable_cached = self._get_hashable_cached(callback)
        if hashable_cached is not None:
            return hashable_cached

        # Fast path: check weak-ref cache first (most common case)
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
        self._set_hashable_cached(callback, result)
        try:
            if len(self._cache) >= self._max_size:
                self._cache.clear()

            self._cache[callback] = result
        except Exception:
            pass

        return result


# Global instance for backward compatibility and convenience
coroutine_checker = CoroutineChecker()
