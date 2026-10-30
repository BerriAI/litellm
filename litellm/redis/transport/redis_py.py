import hashlib
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Protocol, TypeVar

from pydantic import TypeAdapter, ValidationError
from redis.exceptions import NoScriptError

from litellm.caching.redis_cache import RedisCircuitBreaker
from litellm.redis.transport.breaker import run_under_breaker
from litellm.redis.transport.contract import Admit, Arg, EvalReply, RedisFailure, ttl_ms
from litellm.redis.transport.events import RedisServiceEvents

_T: Final = TypeVar("_T")

_ARRAY: Final = TypeAdapter(tuple[object, ...])


class AsyncRedisCommands(Protocol):
    def get(self, name: str) -> Awaitable[object]: ...

    def mget(self, keys: Sequence[str]) -> Awaitable[object]: ...

    def set(self, name: str, value: Arg, *, nx: bool, px: int | None) -> Awaitable[object]: ...

    def delete(self, *names: str) -> Awaitable[int]: ...

    def eval(self, script: str, numkeys: int, *keys_and_args: Arg) -> Awaitable[object]: ...

    def evalsha(self, sha: str, numkeys: int, *keys_and_args: Arg) -> Awaitable[object]: ...


@dataclass(frozen=True, slots=True)
class RedisPyTransport:
    client: Callable[[], AsyncRedisCommands]
    namespace: str | None
    breaker: RedisCircuitBreaker
    admit: Admit
    events: RedisServiceEvents

    async def get(self, key: str) -> bytes | None | RedisFailure:
        return _bytes_reply(await self._run("get", lambda: self.client().get(self._key(key))))

    async def mget(self, keys: Sequence[str]) -> tuple[bytes | None, ...] | RedisFailure:
        namespaced: Final = tuple(self._key(k) for k in keys)
        return _mget_reply(await self._run("mget", lambda: self.client().mget(namespaced)), len(namespaced))

    async def set(self, key: str, value: Arg, *, ttl: timedelta | None, nx: bool = False) -> bool | RedisFailure:
        px: Final = None if ttl is None else ttl_ms(ttl)
        return _bool_reply(await self._run("set", lambda: self.client().set(self._key(key), value, px=px, nx=nx)))

    async def delete(self, *keys: str) -> int | RedisFailure:
        namespaced: Final = tuple(self._key(k) for k in keys)
        return _int_reply(await self._run("delete", lambda: self.client().delete(*namespaced)))

    async def eval(self, script: str, keys: Sequence[str], args: Sequence[Arg]) -> EvalReply | RedisFailure:
        sha: Final = hashlib.sha1(script.encode(), usedforsecurity=False).hexdigest()
        namespaced: Final = tuple(self._key(k) for k in keys)
        return _eval_reply(
            await self._run("eval", lambda: _evalsha_or_eval(self.client(), script, sha, namespaced, args))
        )

    def _key(self, key: str) -> str:
        if not self.namespace or key.startswith(f"{self.namespace}:"):
            return key
        return f"{self.namespace}:{key}"

    async def _run(self, name: str, call: Callable[[], Awaitable[_T]]) -> _T | RedisFailure:
        return await self.events.observe(name, lambda: run_under_breaker(self.breaker, name, self.admit, call))


async def _evalsha_or_eval(
    client: AsyncRedisCommands, script: str, sha: str, keys: Sequence[str], args: Sequence[Arg]
) -> object:
    try:
        return await client.evalsha(sha, len(keys), *keys, *args)
    except NoScriptError:
        return await client.eval(script, len(keys), *keys, *args)


def _decode_failure(reply: object) -> RedisFailure:
    return RedisFailure(kind="decode", message=f"unexpected Redis reply of type {type(reply).__name__}")


def _optional_bytes(reply: object) -> bytes | None | RedisFailure:
    if reply is None or isinstance(reply, bytes):
        return reply
    if isinstance(reply, str):
        return reply.encode()
    return _decode_failure(reply)


def _bytes_reply(reply: object) -> bytes | None | RedisFailure:
    return reply if isinstance(reply, RedisFailure) else _optional_bytes(reply)


def _array(reply: object) -> tuple[object, ...] | None:
    if not isinstance(reply, list):
        return None
    try:
        return _ARRAY.validate_python(reply)
    except ValidationError:
        return None


def _mget_reply(reply: object, expected: int) -> tuple[bytes | None, ...] | RedisFailure:
    if isinstance(reply, RedisFailure):
        return reply
    array: Final = _array(reply)
    if array is None or len(array) != expected:
        return _decode_failure(reply)
    values: Final = tuple(_optional_bytes(value) for value in array)
    failure: Final = next((value for value in values if isinstance(value, RedisFailure)), None)
    return failure if failure is not None else tuple(value for value in values if not isinstance(value, RedisFailure))


def _bool_reply(reply: object) -> bool | RedisFailure:
    if isinstance(reply, RedisFailure):
        return reply
    return reply is not None and reply is not False


def _int_reply(reply: object) -> int | RedisFailure:
    if isinstance(reply, RedisFailure):
        return reply
    return reply if isinstance(reply, int) and not isinstance(reply, bool) else _decode_failure(reply)


def _eval_reply(reply: object) -> EvalReply | RedisFailure:
    if isinstance(reply, RedisFailure) or reply is None or isinstance(reply, bytes):
        return reply
    if isinstance(reply, bool):
        return _decode_failure(reply)
    if isinstance(reply, int):
        return reply
    if isinstance(reply, str):
        return reply.encode()
    array: Final = _array(reply)
    if array is None:
        return _decode_failure(reply)
    items: Final = tuple(_eval_reply(item) for item in array)
    failure: Final = next((item for item in items if isinstance(item, RedisFailure)), None)
    return failure if failure is not None else tuple(item for item in items if not isinstance(item, RedisFailure))
