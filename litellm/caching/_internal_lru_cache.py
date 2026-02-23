from functools import lru_cache
from typing import Callable, Optional, TypeVar

T = TypeVar("T")


def lru_cache_wrapper(
    maxsize: Optional[int] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Wrapper for lru_cache that caches success and exceptions
    """

    def decorator(f: Callable[..., T]) -> Callable[..., T]:
        @lru_cache(maxsize=maxsize)
        def wrapper(*args, **kwargs):
            try:
                return ("success", f(*args, **kwargs))
            except Exception as e:
                return ("error", e)

        def wrapped(*args, **kwargs):
            result = wrapper(*args, **kwargs)
            if result[0] == "error":
                raise result[1]
            return result[1]

        return wrapped

    return decorator
