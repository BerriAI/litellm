from functools import lru_cache, wraps
from typing import Callable, TypeVar, cast

RT = TypeVar("RT")  # Return type


def typed_lru_cache(maxsize: int = 128) -> Callable:
    def decorator(func: Callable[..., RT]) -> Callable[..., RT]:
        wrapped = lru_cache(maxsize=maxsize)(func)
        return cast(Callable[..., RT], wraps(func)(wrapped))

    return decorator
