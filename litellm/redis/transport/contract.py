import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Literal, Protocol, TypeAlias, TypeVar, overload

Arg: TypeAlias = str | bytes | int | float

_T: Final = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class RedisFailure:
    kind: Literal["transport", "server", "decode", "breaker_open"]
    message: str
    timeout: bool = False


@dataclass(frozen=True, slots=True)
class Get:
    key: str


@dataclass(frozen=True, slots=True)
class Set:
    key: str
    value: Arg
    ttl: timedelta | None
    nx: bool = False


@dataclass(frozen=True, slots=True)
class Delete:
    key: str


Command: TypeAlias = Get | Set | Delete


class Guard(Protocol):
    def __call__(self, name: str, call: Callable[[], Awaitable[_T]], /) -> Awaitable[_T | RedisFailure]: ...


def ttl_ms(ttl: timedelta) -> int:
    return max(1, math.ceil(ttl / timedelta(milliseconds=1)))


class RedisTransport(Protocol):
    @overload
    async def execute(self, command: Get) -> bytes | None | RedisFailure: ...

    @overload
    async def execute(self, command: Set) -> bool | RedisFailure: ...

    @overload
    async def execute(self, command: Delete) -> int | RedisFailure: ...
