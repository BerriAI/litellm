import asyncio
import hashlib
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from types import MappingProxyType
from typing import Final, Generic, TypeVar

from pydantic import TypeAdapter, ValidationError
from redis.exceptions import NoScriptError

from litellm.caching.redis_cache import RedisCircuitBreaker
from litellm.redis.transport.batch import BatchResult, PipelineBackend, PipelineBatch
from litellm.redis.transport.breaker import failure_from, run_under_breaker
from litellm.redis.transport.client import AsyncPubSub, AsyncPubSubClient, AsyncRedisCommands, RedisPipeline
from litellm.redis.transport.contract import (
    Admit,
    Arg,
    EvalReply,
    InfoValue,
    Popped,
    PubSubMessage,
    RedisFailure,
    ttl_ms,
)
from litellm.redis.transport.events import RedisServiceEvents

_T: Final = TypeVar("_T")

_ARRAY: Final = TypeAdapter(tuple[object, ...])
_INFO: Final[TypeAdapter[Mapping[str, InfoValue]]] = TypeAdapter(Mapping[str, InfoValue])
_CLIENTS: Final = TypeAdapter(tuple[Mapping[str, str], ...])
_MESSAGE: Final = TypeAdapter(Mapping[str, object])


def is_missing_script(failure: Exception) -> bool:
    return isinstance(failure, NoScriptError)


@dataclass(frozen=True, slots=True)
class RedisPyTransport:
    client: Callable[[], AsyncRedisCommands]
    namespace: str | None
    breaker: RedisCircuitBreaker
    admit: Admit
    events: RedisServiceEvents
    multi_get: Callable[[Sequence[str]], Awaitable[object]]
    server_version: Callable[[], int]
    pubsub_client: Callable[[], AsyncPubSubClient]
    request_engine: Callable[[], PipelineBatch | None]
    post_call_engine: Callable[[], PipelineBatch | None]

    async def get(self, key: str) -> bytes | None | RedisFailure:
        return _bytes_reply(await self._run("get", lambda: self.client().get(self._key(key))))

    async def mget(self, keys: Sequence[str]) -> tuple[bytes | None, ...] | RedisFailure:
        namespaced: Final = tuple(self._key(k) for k in keys)
        return _mget_reply(await self._run("mget", lambda: self.multi_get(namespaced)), len(namespaced))

    async def set(self, key: str, value: Arg, *, ttl: timedelta | None, nx: bool = False) -> bool | RedisFailure:
        px: Final = None if ttl is None else ttl_ms(ttl)
        return _bool_reply(await self._run("set", lambda: self.client().set(self._key(key), value, px=px, nx=nx)))

    async def set_seconds(
        self, key: str, value: Arg, *, ttl: timedelta | None, nx: bool = False
    ) -> bool | RedisFailure:
        return _set_reply(
            await self._run("set_seconds", lambda: self.client().set(self._key(key), value, ex=ttl, nx=nx))
        )

    async def delete(self, *keys: str) -> int | RedisFailure:
        namespaced: Final = tuple(self._key(k) for k in keys)
        return _int_reply(await self._run("delete", lambda: self.client().delete(*namespaced)))

    async def incrby(self, key: str, amount: int) -> int | RedisFailure:
        return _int_reply(await self._run("incrby", lambda: self.client().incrby(self._key(key), amount)))

    async def incrbyfloat(self, key: str, amount: float) -> float | RedisFailure:
        return _float_reply(await self._run("incrbyfloat", lambda: self.client().incrbyfloat(self._key(key), amount)))

    async def expire(self, key: str, ttl: timedelta) -> bool | RedisFailure:
        return _bool_reply(await self._run("expire", lambda: self.client().pexpire(self._key(key), ttl_ms(ttl))))

    async def expire_seconds(self, key: str, ttl: timedelta) -> bool | RedisFailure:
        return _expire_reply(
            await self._run("expire_seconds", lambda: self.client().expire(self._key(key), int(ttl.total_seconds())))
        )

    async def ttl(self, key: str) -> int | RedisFailure:
        return _int_reply(await self._run("ttl", lambda: self.client().ttl(self._key(key))))

    async def rpush(self, key: str, *values: Arg) -> int | RedisFailure:
        return _int_reply(await self._run("rpush", lambda: self.client().rpush(self._key(key), *values)))

    async def lpop(self, key: str, count: int | None = None) -> Popped | RedisFailure:
        name: Final = self._key(key)
        if count is None or self.server_version() >= 7:
            return _popped_reply(await self._run("lpop", lambda: self.client().lpop(name, count)))
        return _popped_each(await self._run("lpop", lambda: _lpop_each_alone(self.client(), name, count)))

    async def ltrim(self, key: str, start: int, stop: int) -> bool | RedisFailure:
        return _bool_reply(await self._run("ltrim", lambda: self.client().ltrim(self._key(key), start, stop)))

    async def sadd(self, key: str, *members: Arg) -> int | RedisFailure:
        return _int_reply(await self._run("sadd", lambda: self.client().sadd(self._key(key), *members)))

    async def scan(self, prefix: str, *, limit: int) -> tuple[bytes, ...] | RedisFailure:
        match: Final = f"{self._key(prefix)}*"
        return _keys_reply(await self._run("scan", lambda: _scan(self.client(), match, limit)))

    async def eval(self, script: str, keys: Sequence[str], args: Sequence[Arg]) -> EvalReply | RedisFailure:
        sha: Final = _sha(script)
        namespaced: Final = tuple(self._key(k) for k in keys)
        return _eval_reply(
            await self._run("eval", lambda: _evalsha_or_eval(self.client(), script, sha, namespaced, args))
        )

    def batch(
        self, *, transaction: bool = False, raise_on_error: bool = False, name: str = "redis_batch"
    ) -> "RedisPyCommandBatch":
        return RedisPyCommandBatch(
            self,
            PipelineBatch(
                self._backend(runs_alone=False), name=name, transaction=transaction, raise_on_error=raise_on_error
            ),
        )

    def request_batch(self) -> "RedisPyCommandBatch | None":
        engine: Final = self.request_engine()
        return None if engine is None else RedisPyCommandBatch(self, engine)

    def post_call_batch(self) -> "RedisPyCommandBatch | None":
        engine: Final = self.post_call_engine()
        return None if engine is None else RedisPyCommandBatch(self, engine)

    async def publish(self, channel: str, message: str | bytes) -> int | RedisFailure:
        return _int_reply(
            await self._run_unguarded("publish", lambda: self.pubsub_client().publish(self._key(channel), message))
        )

    async def publish_all(self, channel: str, messages: Sequence[str | bytes]) -> tuple[int, ...] | RedisFailure:
        name: Final = self._key(channel)
        return _ints_reply(
            await self._run_unguarded("publish", lambda: _publish_each(self.pubsub_client(), name, messages)),
            len(messages),
        )

    async def subscribe(self, *channels: str) -> "RedisPySubscription | RedisFailure":
        names: Final = tuple(self._key(channel) for channel in channels)
        return await self._run_unguarded("subscribe", lambda: _subscribe(self.pubsub_client(), names))

    async def ping(self) -> bool | RedisFailure:
        return _bool_reply(await self._run_unguarded("ping", lambda: self.client().ping()))

    async def info(self) -> Mapping[str, InfoValue] | RedisFailure:
        return _info_reply(await self._run_unguarded("info", lambda: self.client().info()))

    async def client_list(self) -> tuple[Mapping[str, str], ...] | RedisFailure:
        return _clients_reply(await self._run_unguarded("client_list", lambda: self.client().client_list()))

    async def flushall(self) -> bool | RedisFailure:
        return _bool_reply(await self._run_unguarded("flushall", lambda: self.client().flushall()))

    def _key(self, key: str) -> str:
        if not self.namespace or key.startswith(f"{self.namespace}:"):
            return key
        return f"{self.namespace}:{key}"

    def _backend(self, *, runs_alone: bool) -> PipelineBackend:
        return PipelineBackend(
            client=self.client,
            breaker=self.breaker,
            admit=self.admit,
            service_logging=self.events.service_logging,
            runs_alone=runs_alone,
        )

    async def _run(self, name: str, call: Callable[[], Awaitable[_T]]) -> _T | RedisFailure:
        return await self.events.observe(name, lambda: run_under_breaker(self.breaker, name, self.admit, call))

    async def _run_unguarded(self, name: str, call: Callable[[], Awaitable[_T]]) -> _T | RedisFailure:
        return await self.events.observe(name, lambda: _without_breaker(call))


@dataclass(frozen=True, slots=True)
class _Command(Generic[_T]):
    queue: Callable[[RedisPipeline], int]
    decode: Callable[[Sequence[object]], _T | RedisFailure]
    alone: Callable[[], Awaitable[_T | RedisFailure]]

    def enqueue(self, pipe: RedisPipeline) -> int:
        return self.queue(pipe)

    def resolve(self, replies: Sequence[object]) -> _T | RedisFailure:
        return self.decode(replies)

    def recover(self, error: Exception) -> _T | RedisFailure:
        failure: Final = failure_from(error)
        if failure is None:
            raise error
        return failure

    def run_alone(self) -> Awaitable[_T | RedisFailure]:
        return self.alone()


@dataclass(frozen=True, slots=True)
class RedisPyCommandBatch:
    transport: RedisPyTransport
    engine: PipelineBatch

    def get(self, key: str) -> BatchResult[bytes | None | RedisFailure]:
        name: Final = self.transport._key(key)  # pyright: ignore[reportPrivateUsage]  # the batch is the transport's own
        reply: Final = self._declare(
            lambda pipe: _one(pipe.get(name)), lambda replies: _bytes_reply(replies[0]), lambda: self.transport.get(key)
        )
        reply.on_settled(lambda future: self._note_read((key,), future, lambda value: (value,)))
        return reply

    def mget(self, keys: Sequence[str]) -> BatchResult[tuple[bytes | None, ...] | RedisFailure]:
        logical: Final = tuple(keys)
        names: Final = tuple(self.transport._key(k) for k in logical)  # pyright: ignore[reportPrivateUsage]  # the batch is the transport's own
        reply: Final = self._declare(
            lambda pipe: _one(pipe.mget(names)),
            lambda replies: _mget_reply(replies[0], len(names)),
            lambda: self.transport.mget(logical),
        )
        reply.on_settled(lambda future: self._note_read(logical, future, lambda values: values))
        return reply

    def set(self, key: str, value: Arg, *, ttl: timedelta | None, nx: bool = False) -> BatchResult[bool | RedisFailure]:
        name: Final = self.transport._key(key)  # pyright: ignore[reportPrivateUsage]  # the batch is the transport's own
        px: Final = None if ttl is None else ttl_ms(ttl)
        self.engine.note_present(key)
        return self._declare(
            lambda pipe: _one(pipe.set(name, value, px=px, nx=nx)),
            lambda replies: _bool_reply(replies[0]),
            lambda: self.transport.set(key, value, ttl=ttl, nx=nx),
        )

    def delete(self, *keys: str) -> BatchResult[int | RedisFailure]:
        names: Final = tuple(self.transport._key(k) for k in keys)  # pyright: ignore[reportPrivateUsage]  # the batch is the transport's own
        self.engine.note_missing(keys)
        return self._declare(
            lambda pipe: _one(pipe.delete(*names)),
            lambda replies: _int_reply(replies[0]),
            lambda: self.transport.delete(*keys),
        )

    def set_seconds(
        self, key: str, value: Arg, *, ttl: timedelta | None, nx: bool = False
    ) -> BatchResult[bool | RedisFailure]:
        name: Final = self.transport._key(key)  # pyright: ignore[reportPrivateUsage]  # the batch is the transport's own
        self.engine.note_present(key)
        return self._declare(
            lambda pipe: _one(pipe.set(name, value, ex=ttl, nx=nx)),
            lambda replies: _set_reply(replies[0]),
            lambda: self.transport.set_seconds(key, value, ttl=ttl, nx=nx),
        )

    def incrby(self, key: str, amount: int) -> BatchResult[int | RedisFailure]:
        name: Final = self.transport._key(key)  # pyright: ignore[reportPrivateUsage]  # the batch is the transport's own
        self.engine.note_present(key)
        return self._declare(
            lambda pipe: _one(pipe.incrby(name, amount)),
            lambda replies: _int_reply(replies[0]),
            lambda: self.transport.incrby(key, amount),
        )

    def incrbyfloat(self, key: str, amount: float) -> BatchResult[float | RedisFailure]:
        name: Final = self.transport._key(key)  # pyright: ignore[reportPrivateUsage]  # the batch is the transport's own
        self.engine.note_present(key)
        return self._declare(
            lambda pipe: _one(pipe.incrbyfloat(name, amount)),
            lambda replies: _float_reply(replies[0]),
            lambda: self.transport.incrbyfloat(key, amount),
        )

    def expire(self, key: str, ttl: timedelta) -> BatchResult[bool | RedisFailure]:
        name: Final = self.transport._key(key)  # pyright: ignore[reportPrivateUsage]  # the batch is the transport's own
        return self._declare(
            lambda pipe: _one(pipe.pexpire(name, ttl_ms(ttl))),
            lambda replies: _bool_reply(replies[0]),
            lambda: self.transport.expire(key, ttl),
        )

    def ttl(self, key: str) -> BatchResult[int | RedisFailure]:
        name: Final = self.transport._key(key)  # pyright: ignore[reportPrivateUsage]  # the batch is the transport's own
        return self._declare(
            lambda pipe: _one(pipe.ttl(name)), lambda replies: _int_reply(replies[0]), lambda: self.transport.ttl(key)
        )

    def expire_seconds(self, key: str, ttl: timedelta) -> BatchResult[bool | RedisFailure]:
        name: Final = self.transport._key(key)  # pyright: ignore[reportPrivateUsage]  # the batch is the transport's own
        return self._declare(
            lambda pipe: _one(pipe.expire(name, ttl)),
            lambda replies: _expire_reply(replies[0]),
            lambda: self.transport.expire_seconds(key, ttl),
        )

    def rpush(self, key: str, *values: Arg) -> BatchResult[int | RedisFailure]:
        name: Final = self.transport._key(key)  # pyright: ignore[reportPrivateUsage]  # the batch is the transport's own
        self.engine.note_present(key)
        return self._declare(
            lambda pipe: _one(pipe.rpush(name, *values)),
            lambda replies: _int_reply(replies[0]),
            lambda: self.transport.rpush(key, *values),
        )

    def lpop(self, key: str, count: int | None = None) -> BatchResult[Popped | RedisFailure]:
        name: Final = self.transport._key(key)  # pyright: ignore[reportPrivateUsage]  # the batch is the transport's own
        if count is None or self.transport.server_version() >= 7:
            return self._declare(
                lambda pipe: _one(pipe.lpop(name, count)),
                lambda replies: _popped_reply(replies[0]),
                lambda: self.transport.lpop(key, count),
            )
        return self._declare(
            lambda pipe: _queue_lpop_each(pipe, name, count),
            _popped_each,
            lambda: self.transport.lpop(key, count),
        )

    def ltrim(self, key: str, start: int, stop: int) -> BatchResult[bool | RedisFailure]:
        name: Final = self.transport._key(key)  # pyright: ignore[reportPrivateUsage]  # the batch is the transport's own
        return self._declare(
            lambda pipe: _one(pipe.ltrim(name, start, stop)),
            lambda replies: _bool_reply(replies[0]),
            lambda: self.transport.ltrim(key, start, stop),
        )

    def sadd(self, key: str, *members: Arg) -> BatchResult[int | RedisFailure]:
        name: Final = self.transport._key(key)  # pyright: ignore[reportPrivateUsage]  # the batch is the transport's own
        self.engine.note_present(key)
        return self._declare(
            lambda pipe: _one(pipe.sadd(name, *members)),
            lambda replies: _int_reply(replies[0]),
            lambda: self.transport.sadd(key, *members),
        )

    def eval(self, script: str, keys: Sequence[str], args: Sequence[Arg]) -> BatchResult[EvalReply | RedisFailure]:
        names: Final = tuple(self.transport._key(k) for k in keys)  # pyright: ignore[reportPrivateUsage]  # the batch is the transport's own
        sha: Final = _sha(script)
        transaction: Final = self.engine.transaction
        return self._declare(
            lambda pipe: _one(
                pipe.eval(script, len(names), *names, *args)
                if transaction
                else pipe.evalsha(sha, len(names), *names, *args)
            ),
            lambda replies: _eval_reply(replies[0]),
            lambda: self.transport.eval(script, keys, args),
        )

    async def flush(self) -> None:
        await self.engine.flush()

    def _declare(
        self,
        queue: Callable[[RedisPipeline], int],
        decode: Callable[[Sequence[object]], _T | RedisFailure],
        alone: Callable[[], Awaitable[_T | RedisFailure]],
    ) -> BatchResult[_T | RedisFailure]:
        return self.engine.declare(_Command(queue, decode, alone))

    def _note_read(
        self,
        keys: Sequence[str],
        future: "asyncio.Future[_T | RedisFailure]",
        values: Callable[[_T], Sequence[bytes | None]],
    ) -> None:
        if future.cancelled() or future.exception() is not None:
            return
        result: Final = future.result()
        if isinstance(result, RedisFailure):
            return
        self.engine.note_missing(key for key, value in zip(keys, values(result)) if value is None)


@dataclass(frozen=True, slots=True)
class RedisPySubscription:
    pubsub: AsyncPubSub

    async def get_message(self, *, timeout: float) -> PubSubMessage | None | RedisFailure:
        return _message_reply(
            await _without_breaker(lambda: self.pubsub.get_message(ignore_subscribe_messages=True, timeout=timeout))
        )

    async def close(self) -> None | RedisFailure:
        closed: Final = await _without_breaker(self.pubsub.reset)
        return closed if isinstance(closed, RedisFailure) else None


def _one(_queued: object) -> int:
    return 1


def _sha(script: str) -> str:
    return hashlib.sha1(script.encode(), usedforsecurity=False).hexdigest()


def _queue_lpop_each(pipe: RedisPipeline, name: str, count: int) -> int:
    for _ in range(count):
        pipe.lpop(name)
    return count


async def _lpop_each_alone(client: AsyncRedisCommands, name: str, count: int) -> Sequence[object]:
    async with client.pipeline(transaction=False) as pipe:
        _queue_lpop_each(pipe, name, count)
        return await pipe.execute()


async def _publish_each(client: AsyncPubSubClient, channel: str, messages: Sequence[str | bytes]) -> Sequence[object]:
    async with client.pipeline(transaction=False) as pipe:
        for message in messages:
            pipe.publish(channel, message)
        return await pipe.execute()


async def _subscribe(client: AsyncPubSubClient, channels: Sequence[str]) -> RedisPySubscription:
    pubsub: Final = client.pubsub()
    await pubsub.subscribe(*channels)
    return RedisPySubscription(pubsub)


async def _scan(client: AsyncRedisCommands, match: str, limit: int) -> tuple[object, ...]:
    keys: Final[list[object]] = []  # mutable-ok: filled while the cursor walks
    async for key in client.scan_iter(match=match, count=limit):
        keys.append(key)
        if len(keys) >= limit:
            break
    return tuple(keys)


async def _without_breaker(call: Callable[[], Awaitable[_T]]) -> _T | RedisFailure:
    try:
        return await call()
    except Exception as e:
        failure: Final = failure_from(e)
        if failure is None:
            raise
        return failure


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
    if not isinstance(reply, (list, tuple)):
        return None
    try:
        return _ARRAY.validate_python(reply)
    except ValidationError:
        return None


def _all_bytes(values: Sequence[object]) -> tuple[bytes, ...] | RedisFailure:
    decoded: Final = tuple(_optional_bytes(value) for value in values)
    failure: Final = next((value for value in decoded if isinstance(value, RedisFailure)), None)
    if failure is not None:
        return failure
    present: Final = tuple(value for value in decoded if isinstance(value, bytes))
    return present if len(present) == len(decoded) else _decode_failure(values)


def _mget_reply(reply: object, expected: int) -> tuple[bytes | None, ...] | RedisFailure:
    if isinstance(reply, RedisFailure):
        return reply
    array: Final = _array(reply)
    if array is None or len(array) != expected:
        return _decode_failure(reply)
    values: Final = tuple(_optional_bytes(value) for value in array)
    failure: Final = next((value for value in values if isinstance(value, RedisFailure)), None)
    return failure if failure is not None else tuple(value for value in values if not isinstance(value, RedisFailure))


def _keys_reply(reply: object) -> tuple[bytes, ...] | RedisFailure:
    if isinstance(reply, RedisFailure):
        return reply
    array: Final = _array(reply)
    return _decode_failure(reply) if array is None else _all_bytes(array)


def _popped_reply(reply: object) -> Popped | RedisFailure:
    if isinstance(reply, RedisFailure) or reply is None:
        return reply
    array: Final = _array(reply)
    if array is None:
        return _bytes_reply(reply)
    return _all_bytes(array)


def _popped_each(replies: object) -> Popped | RedisFailure:
    if isinstance(replies, RedisFailure):
        return replies
    array: Final = _array(replies)
    if array is None:
        return _decode_failure(replies)
    popped: Final = _all_bytes(tuple(value for value in array if value is not None))
    return None if popped == () else popped


def _bool_reply(reply: object) -> bool | RedisFailure:
    if isinstance(reply, RedisFailure):
        return reply
    return reply is not None and reply is not False and reply != 0


def _expire_reply(reply: object) -> bool | RedisFailure:
    if isinstance(reply, (RedisFailure, bool)):
        return reply
    if type(reply) is int and reply in (0, 1):
        return reply == 1
    return _decode_failure(reply)


def _set_reply(reply: object) -> bool | RedisFailure:
    if isinstance(reply, RedisFailure):
        return reply
    if reply is None or reply is False:
        return False
    if reply is True or reply in (b"OK", "OK"):
        return True
    return _decode_failure(reply)


def _int_reply(reply: object) -> int | RedisFailure:
    if isinstance(reply, RedisFailure):
        return reply
    return reply if isinstance(reply, int) and not isinstance(reply, bool) else _decode_failure(reply)


def _ints_reply(reply: object, expected: int) -> tuple[int, ...] | RedisFailure:
    if isinstance(reply, RedisFailure):
        return reply
    array: Final = _array(reply)
    if array is None or len(array) != expected:
        return _decode_failure(reply)
    counts: Final = tuple(_int_reply(value) for value in array)
    failure: Final = next((value for value in counts if isinstance(value, RedisFailure)), None)
    return failure if failure is not None else tuple(value for value in counts if not isinstance(value, RedisFailure))


def _float_reply(reply: object) -> float | RedisFailure:
    if isinstance(reply, RedisFailure):
        return reply
    if isinstance(reply, bool) or not isinstance(reply, (int, float, str, bytes)):
        return _decode_failure(reply)
    try:
        return float(reply)
    except ValueError:
        return _decode_failure(reply)


def _info_reply(reply: object) -> Mapping[str, InfoValue] | RedisFailure:
    if isinstance(reply, RedisFailure):
        return reply
    try:
        return MappingProxyType(_INFO.validate_python(reply))
    except ValidationError:
        return _decode_failure(reply)


def _clients_reply(reply: object) -> tuple[Mapping[str, str], ...] | RedisFailure:
    if isinstance(reply, RedisFailure):
        return reply
    try:
        return tuple(MappingProxyType(client) for client in _CLIENTS.validate_python(reply))
    except ValidationError:
        return _decode_failure(reply)


def _message_reply(reply: object) -> PubSubMessage | None | RedisFailure:
    if isinstance(reply, RedisFailure) or reply is None:
        return reply
    try:
        message: Final = _MESSAGE.validate_python(reply)
    except ValidationError:
        return _decode_failure(reply)
    channel: Final = _optional_bytes(message.get("channel"))
    data: Final = _optional_bytes(message.get("data"))
    if not isinstance(channel, bytes) or not isinstance(data, bytes):
        return _decode_failure(reply)
    return PubSubMessage(channel=channel, data=data)


def _eval_reply(reply: object) -> EvalReply | RedisFailure:
    frames: Final[list[tuple[Iterator[object], list[EvalReply]]]] = [  # mutable-ok: linear-time traversal stack
        (iter((reply,)), [])
    ]
    while frames:
        pending, values = frames[-1]
        for item in pending:
            if (array := _array(item)) is not None:
                frames.append((iter(array), []))
                break
            if isinstance(scalar := _eval_scalar(item), RedisFailure):
                return scalar
            values.append(scalar)
        else:
            frames.pop()
            if not frames:
                return values[0]
            frames[-1][1].append(tuple(values))
    return _decode_failure(reply)


def _eval_scalar(reply: object) -> int | bytes | None | RedisFailure:
    if isinstance(reply, RedisFailure) or reply is None or isinstance(reply, bytes):
        return reply
    if isinstance(reply, bool):
        return _decode_failure(reply)
    if isinstance(reply, int):
        return reply
    if isinstance(reply, str):
        return reply.encode()
    return _decode_failure(reply)
