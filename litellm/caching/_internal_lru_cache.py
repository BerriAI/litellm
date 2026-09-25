from collections.abc import Callable
from functools import lru_cache
from typing import Final, TypeVar

T = TypeVar("T")


def lru_cache_wrapper(
    maxsize: int | None = None,
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
            result: Final = wrapper(*args, **kwargs)
            if result[0] == "error":
                raise result[1]
            return result[1]

        return wrapped

    return decorator
