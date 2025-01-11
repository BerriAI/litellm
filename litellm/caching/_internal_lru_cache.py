from functools import lru_cache
from typing import Any, Callable, ParamSpec, TypeVar, cast

P = ParamSpec("P")
T = TypeVar("T", bound=Callable[..., Any])


def typed_lru_cache(maxsize: int) -> Callable[[T], T]:
    """
    Decorator to cache the result of a function with a configurable maximum size.
    Args:
        maxsize (int): Maximum size of the cache. Defaults to 128.
    """

    def decorator(f: T) -> T:
        return cast(T, lru_cache(maxsize=maxsize)(f))

    return decorator
