from functools import lru_cache
from typing import Any, Callable, ParamSpec, TypeVar, cast

P = ParamSpec("P")
T = TypeVar("T", bound=Callable[..., Any])


def typed_lru_cache(f: T) -> T:
    """
    Decorator to cache the result of a function with a maximum size of 128.
    """
    return cast(T, lru_cache(maxsize=128)(f))
