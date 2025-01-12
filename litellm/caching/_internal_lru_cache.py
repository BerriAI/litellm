from functools import lru_cache
from typing import Any, Callable, TypeVar, cast

from typing_extensions import ParamSpec

P = ParamSpec("P")
T = TypeVar("T", bound=Callable[..., Any])


def typed_lru_cache(maxsize: int) -> Callable[[T], T]:
    """
    Decorator to cache the result of a function with a configurable maximum size.
    Skips caching if any arguments are not hashable.

    Args:
        maxsize (int): Maximum size of the cache. Defaults to 128.
    """

    def decorator(f: T) -> T:
        cached_f = lru_cache(maxsize=maxsize, typed=True)(f)

        def wrapper(*args, **kwargs):
            try:
                return cached_f(*args, **kwargs)
            except TypeError:
                return f(*args, **kwargs)

        return cast(T, wrapper)

    return decorator
