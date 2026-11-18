"""While set, a cache entry refilled from the database fails closed and is read from the writer."""

from contextvars import ContextVar, Token
from typing import Final

_strict: Final[ContextVar[bool]] = ContextVar("strict_cache_refill", default=False)


def strict_cache_refill() -> bool:
    return _strict.get()


def require_strict_cache_refill() -> Token[bool]:
    return _strict.set(True)


def release_strict_cache_refill(token: Token[bool]) -> None:
    _strict.reset(token)
