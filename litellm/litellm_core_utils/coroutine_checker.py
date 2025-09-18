# CoroutineChecker utility for checking if functions/callables are coroutines or coroutine functions

import inspect
from typing import Any
from weakref import WeakKeyDictionary
from litellm.constants import COROUTINE_CHECKER_MAX_SIZE_IN_MEMORY


class CoroutineChecker:
    """Utility for checking coroutine status with a bounded weakref cache."""

    def __init__(self):
        self._cache = WeakKeyDictionary()
        self._max_size = COROUTINE_CHECKER_MAX_SIZE_IN_MEMORY

    def is_async_callable(self, callback: Any) -> bool:
        """Return True if `callback` is an async function/callable, else False."""

        if not callable(callback):  # Fast path for non-callables
            return False

        # Check cache
        try:
            if callback in self._cache:
                return self._cache[callback]
        except Exception:
            return False  # Weird/unhashable callables

        # Resolve to the actual target (function/method or __call__)
        target = getattr(callback, "__call__", callback)
        try:
            result = inspect.iscoroutinefunction(target)
        except Exception:
            result = False

        # Cache result (bounded)
        try:
            if len(self._cache) >= self._max_size:
                self._cache.clear()
            self._cache[callback] = result
        except Exception:
            pass  # Skip caching if not possible

        return result


# Global instance for convenience
coroutine_checker = CoroutineChecker()
