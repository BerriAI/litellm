from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Final, Protocol, assert_never, overload

from litellm.redis.transport.contract import Arg, Command, Delete, Get, Guard, RedisFailure, Set, ttl_ms


class AsyncRedisCommands(Protocol):
    def get(self, name: str) -> Awaitable[object]: ...

    def set(self, name: str, value: Arg, *, nx: bool, px: int | None) -> Awaitable[object]: ...

    def delete(self, *names: str) -> Awaitable[int]: ...


def _decode_failure(value: object) -> RedisFailure:
    return RedisFailure(kind="decode", message=f"unexpected Redis reply of type {type(value).__name__}")


def _bytes_reply(reply: object) -> bytes | None | RedisFailure:
    if reply is None or isinstance(reply, (bytes, RedisFailure)):
        return reply
    if isinstance(reply, str):
        return reply.encode()
    return _decode_failure(reply)


@dataclass(frozen=True, slots=True)
class RedisPyTransport:
    client: Callable[[], AsyncRedisCommands]
    namespace: str | None
    guard: Guard

    def _key(self, key: str) -> str:
        if not self.namespace or key.startswith(f"{self.namespace}:"):
            return key
        return f"{self.namespace}:{key}"

    @overload
    async def execute(self, command: Get) -> bytes | None | RedisFailure: ...

    @overload
    async def execute(self, command: Set) -> bool | RedisFailure: ...

    @overload
    async def execute(self, command: Delete) -> int | RedisFailure: ...

    async def execute(self, command: Command) -> bytes | None | bool | int | RedisFailure:
        match command:
            case Get(key=key):
                return _bytes_reply(await self.guard("get", lambda: self.client().get(self._key(key))))
            case Set(key=key, value=value, ttl=ttl, nx=nx):
                px: Final = None if ttl is None else ttl_ms(ttl)
                written: Final = await self.guard("set", lambda: self.client().set(self._key(key), value, nx=nx, px=px))
                return written if isinstance(written, RedisFailure) else written is not None and written is not False
            case Delete(key=key):
                return await self.guard("delete", lambda: self.client().delete(self._key(key)))
            case _:
                assert_never(command)
