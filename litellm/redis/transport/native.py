from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Final, TypeVar, assert_never, overload

from litellm.redis.transport.contract import Arg, Command, Delete, Get, Guard, RedisFailure, Set, ttl_ms
from litellm.rust_bridge._native import NativeRedisClient, NativeRedisFailure

_R: Final = TypeVar("_R")


def _encode(value: Arg) -> bytes:
    match value:
        case bytes():
            return value
        case str():
            return value.encode()
        case float():
            return repr(value).encode()
        case int():
            return str(value).encode()


def _failure(failure: NativeRedisFailure) -> RedisFailure:
    return RedisFailure(kind=failure.kind, message=failure.message, timeout=failure.timeout)


async def _reply(pending: Awaitable[_R | NativeRedisFailure]) -> _R | RedisFailure:
    reply: Final = await pending
    return _failure(reply) if isinstance(reply, NativeRedisFailure) else reply


@dataclass(frozen=True, slots=True)
class NativeRedisTransport:
    client: NativeRedisClient
    guard: Guard

    @overload
    async def execute(self, command: Get) -> bytes | None | RedisFailure: ...

    @overload
    async def execute(self, command: Set) -> bool | RedisFailure: ...

    @overload
    async def execute(self, command: Delete) -> int | RedisFailure: ...

    async def execute(self, command: Command) -> bytes | None | bool | int | RedisFailure:
        match command:
            case Get(key=key):
                return await self.guard("get", lambda: _reply(self.client.get(key)))
            case Set(key=key, value=value, ttl=ttl, nx=nx):
                if ttl is None:
                    return RedisFailure(kind="server", message="the native client requires a TTL on every write")
                return await self.guard(
                    "set", lambda: _reply(self.client.set(key, _encode(value), ttl_ms=ttl_ms(ttl), nx=nx))
                )
            case Delete(key=key):
                return await self.guard("delete", lambda: _reply(self.client.delete(key)))
            case _:
                assert_never(command)
