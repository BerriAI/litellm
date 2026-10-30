import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Literal, Protocol, TypeAlias, TypeVar

Arg: TypeAlias = str | bytes | int | float
EvalReply: TypeAlias = int | bytes | None | tuple["EvalReply", ...]

_T: Final = TypeVar("_T")


class Admit(Protocol):
    def __call__(self, call: Callable[[], Awaitable[_T]], /) -> Awaitable[_T]: ...


@dataclass(frozen=True, slots=True)
class RedisFailure:
    kind: Literal["transport", "server", "decode", "breaker_open"]
    message: str
    timeout: bool = False


def ttl_ms(ttl: timedelta) -> int:
    return max(1, math.ceil(ttl / timedelta(milliseconds=1)))


class RedisTransport(Protocol):
    async def get(self, key: str) -> bytes | None | RedisFailure: ...

    async def mget(self, keys: Sequence[str]) -> tuple[bytes | None, ...] | RedisFailure: ...

    async def set(self, key: str, value: Arg, *, ttl: timedelta | None, nx: bool = False) -> bool | RedisFailure: ...

    async def delete(self, *keys: str) -> int | RedisFailure: ...

    async def eval(self, script: str, keys: Sequence[str], args: Sequence[Arg]) -> EvalReply | RedisFailure: ...
