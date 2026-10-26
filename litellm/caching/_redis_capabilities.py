from __future__ import annotations

import asyncio
import warnings
from collections.abc import Awaitable, Callable, Generator, Mapping, Sequence, Sized
from contextlib import suppress
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Final,
    Generic,
    Literal,
    Protocol,
    TypeAlias,
    TypeVar,
    cast,  # noqa: TID251  # redis transport protocols only; all replies are validated
)

from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm.types.caching import (
    CacheFailure,
    CacheOperation,
    CachePoolInfo,
    CachePublish,
    CacheQueueUpdate,
    CacheSubscription,
    CapacityExceeded,
    CounterUpdate,
    LimitedIncrement,
    RedisPipelineRpushOperation,
    WindowCount,
)

if TYPE_CHECKING:
    from litellm.caching.redis_batch import BatchResult, RedisBatch
    from litellm.caching.redis_cache import RedisCache

_T: Final = TypeVar("_T")
_Arg: TypeAlias = str | bytes | int | float
_OBJECTS: Final = TypeAdapter(tuple[object, ...])
_INTS: Final = TypeAdapter(tuple[int, ...])
_FLOATS: Final = TypeAdapter(tuple[float, ...])
_INT: Final = TypeAdapter(int)
_FLOAT: Final = TypeAdapter(float)
_TEXT: Final = TypeAdapter(str)
_OPTIONAL_TEXT: Final = TypeAdapter[str | None](str | None)
_PAIR: Final = TypeAdapter(tuple[int, int])
_JSON: Final = TypeAdapter[JsonValue](JsonValue)
_PAYLOAD: Final = TypeAdapter[str | bytes](str | bytes)
_OPTIONAL_INT: Final = TypeAdapter[int | None](int | None)
_CACHE_VALUES: Final = TypeAdapter(tuple[str | bytes | None, ...])


def provisional(name: str) -> None:
    warnings.warn(
        f"RedisCache.{name} is experimental and may change in a future release",
        FutureWarning,
        stacklevel=3,
    )


def failure(error: BaseException) -> CacheFailure:
    from litellm.caching.redis_cache import RedisCircuitBreakerOpenError, is_redis_timeout_failure

    reason: Final = (
        "circuit_open"
        if isinstance(error, RedisCircuitBreakerOpenError)
        else "timeout" if is_redis_timeout_failure(error) else "other"
    )
    return CacheFailure(
        kind="unavailable", message=f"Cache operation failed ({type(error).__name__}): {error}", reason=reason
    )


def invalid_reply() -> CacheFailure:
    return CacheFailure(kind="invalid_response", message="Cache returned an invalid operation response")


def validated(decode: Callable[[object], _T | CacheFailure], reply: object) -> _T | CacheFailure:
    try:
        return decode(reply)
    except ValidationError as error:
        return CacheFailure(kind="invalid_response", message=str(error), validation_error=error)
    except (ValueError, TypeError, UnicodeError, SyntaxError):
        return invalid_reply()


class _Operation(Generic[_T]):
    def __init__(
        self,
        run: Callable[[], Awaitable[object]],
        decode: Callable[[object], _T | CacheFailure],
        batched: BatchResult[object] | None,
    ) -> None:
        self._run = run
        self._decode = decode
        self._batched = batched
        self._task: asyncio.Task[_T | CacheFailure] | None = None

    @property
    def done(self) -> bool:
        provisional("operation.done")
        return self._batched.done if self._batched is not None else self._task is not None and self._task.done()

    def __await__(self) -> Generator[object, None, _T | CacheFailure]:
        return self._wait().__await__()

    async def _execute(self) -> _T | CacheFailure:
        try:
            reply: Final = await self._run() if self._batched is None else await self._batched
        except Exception as error:  # noqa: BLE001  # backend failures are values; cancellation propagates
            return failure(error)
        return validated(self._decode, reply)

    def _ensure_task(self) -> asyncio.Task[_T | CacheFailure]:
        if self._task is None:
            self._task = asyncio.create_task(self._execute())
        return self._task

    async def _wait(self) -> _T | CacheFailure:
        return await self._ensure_task()

    def on_settled(self, hook: Callable[[asyncio.Future[_T | CacheFailure]], Awaitable[None] | None]) -> None:
        provisional("operation.on_settled")

        async def notify(raw: asyncio.Future[object]) -> None:
            result: Final[asyncio.Future[_T | CacheFailure]] = asyncio.get_running_loop().create_future()
            if raw.cancelled():
                result.cancel()
            elif (error := raw.exception()) is not None:
                result.set_result(failure(error))
            else:
                result.set_result(validated(self._decode, raw.result()))
            follow_up: Final = hook(result)
            if follow_up is not None:
                await follow_up

        if self._batched is not None:
            self._batched.on_settled(notify)
            return

        def completed(result: asyncio.Future[_T | CacheFailure]) -> None:
            follow_up: Final = hook(result)
            if follow_up is not None:
                asyncio.ensure_future(follow_up)

        self._ensure_task().add_done_callback(completed)


def operation(
    cache: RedisCache,
    source: str,
    keys: Sequence[str],
    args: Sequence[_Arg],
    decode: Callable[[object], _T | CacheFailure],
    batch: RedisBatch | None,
) -> CacheOperation[_T]:
    from litellm.caching.redis_batch import _Script  # pyright: ignore[reportPrivateUsage]  # private cache transport

    runner: Final = cache._script_runner(source)  # pyright: ignore[reportPrivateUsage]  # private cache implementation
    batched: Final = (
        None if batch is None else batch._declare(_Script(cache, source, runner, keys, args))  # pyright: ignore[reportPrivateUsage]  # private cache transport
    )
    return _Operation(lambda: runner(keys=keys, args=args), decode, batched)


def delete_record(cache: RedisCache, key: str) -> CacheOperation[bool]:
    return _Operation(lambda: cache.async_delete_cache(key), boolean, None)


def enqueue_updates(cache: RedisCache, updates: Sequence[CacheQueueUpdate]) -> CacheOperation[tuple[int, ...]]:
    commands: Final = tuple(RedisPipelineRpushOperation(key=item.key, values=item.values) for item in updates)
    return _Operation(lambda: cache.async_rpush_pipeline(commands), lambda reply: counts(reply, len(commands)), None)


def integer(reply: object) -> int:
    return _INT.validate_python(reply)


def number(reply: object) -> float:
    return _FLOAT.validate_python(reply)


def acknowledged(reply: object) -> None | CacheFailure:
    return None if integer(reply) == 1 else invalid_reply()


def boolean(reply: object) -> bool | CacheFailure:
    value: Final = integer(reply)
    return bool(value) if value in (0, 1) else invalid_reply()


def acquired(reply: object) -> Literal["acquired", "held"] | CacheFailure:
    if reply is None or reply is False:
        return "held"
    return "acquired" if reply in (True, "OK", b"OK") else invalid_reply()


def text(reply: object) -> str | None:
    return _OPTIONAL_TEXT.validate_python(reply)


def pin(reply: object) -> JsonValue | CacheFailure:
    try:
        value: Final = _TEXT.validate_python(reply.decode("utf-8") if isinstance(reply, bytes) else reply)
    except UnicodeError as error:
        return failure(error)
    try:
        return _JSON.validate_json(value)
    except ValidationError:
        return value


def block_ttls(reply: object) -> tuple[int, int]:
    return _PAIR.validate_python(reply)


def limited(reply: object) -> LimitedIncrement | CacheFailure:
    accepted, count = _PAIR.validate_python(reply)
    return LimitedIncrement(bool(accepted), count) if accepted in (0, 1) else invalid_reply()


def counts(reply: object, size: int) -> tuple[int, ...] | CacheFailure:
    values: Final = _INTS.validate_python(reply)
    return values if len(values) == size else invalid_reply()


def totals(reply: object, size: int) -> tuple[float, ...] | CacheFailure:
    values: Final = _FLOATS.validate_python(reply)
    return values if len(values) == size else invalid_reply()


def optional_counts(reply: object, size: int) -> tuple[int | None, ...] | CacheFailure:
    values: Final = _OBJECTS.validate_python(reply)
    if len(values) != size:
        return invalid_reply()
    return tuple(None if value is None or value is False else integer(value) for value in values)


def _window_start(reply: object) -> str:
    number(reply)
    return _TEXT.validate_python(reply) if isinstance(reply, (str, bytes)) else str(reply)


def window_counts(reply: object, size: int) -> tuple[WindowCount, ...] | CacheFailure:
    values: Final = _OBJECTS.validate_python(reply)
    if len(values) != size * 2:
        return invalid_reply()
    return tuple(WindowCount(_window_start(values[i]), integer(values[i + 1])) for i in range(0, len(values), 2))


def _exceeded(values: tuple[int, ...], size: int) -> CapacityExceeded | CacheFailure:
    if len(values) != 4 or not 1 <= values[1] <= size:
        return invalid_reply()
    return CapacityExceeded(index=values[1] - 1, count=values[2], limit=values[3])


def reservation(reply: object, size: int) -> tuple[WindowCount, ...] | CapacityExceeded | CacheFailure:
    values: Final = _OBJECTS.validate_python(reply)
    if not values:
        return invalid_reply()
    if integer(values[0]) == 1:
        return _exceeded(_INTS.validate_python(values), size)
    if integer(values[0]) != 0 or len(values) != 1 + size * 2:
        return invalid_reply()
    return tuple(WindowCount(_window_start(values[i + 1]), integer(values[i])) for i in range(1, len(values), 2))


def slots(reply: object, size: int) -> tuple[int, ...] | CapacityExceeded | CacheFailure:
    values: Final = _INTS.validate_python(reply)
    if not values:
        return invalid_reply()
    if values[0] == 1:
        return _exceeded(values, size)
    return values[1:] if values[0] == 0 and len(values) == size + 1 else invalid_reply()


def guarded(reply: object, size: int) -> tuple[CounterUpdate, ...] | CacheFailure:
    values: Final = _INTS.validate_python(reply)
    if len(values) != size * 2 or any(values[i] not in (0, 1) for i in range(0, len(values), 2)):
        return invalid_reply()
    return tuple(CounterUpdate(bool(values[i]), values[i + 1]) for i in range(0, len(values), 2))


class _RegisteredScript(Protocol):
    def __call__(self, keys: Sequence[str], args: Sequence[_Arg], client: object = None) -> Awaitable[object]: ...


class _ScriptClient(Protocol):
    def register_script(self, source: str) -> _RegisteredScript: ...


class _ClusterScripts(Protocol):
    def script_load(self, source: str) -> str | Awaitable[str]: ...
    async def evalsha(self, sha: str, numkeys: int, *args: _Arg) -> object: ...


def script_executor(cache: RedisCache, source: str) -> _RegisteredScript:
    transport: Final = cache._get_async_client()  # pyright: ignore[reportPrivateUsage]  # private cache transport
    if hasattr(transport, "register_script"):
        registered: Final = cast(_ScriptClient, transport).register_script(source)  # cast-ok: redis transport only

        async def standalone(keys: Sequence[str], args: Sequence[_Arg], client: object = None) -> object:
            names: Final = tuple(cache.check_and_fix_namespace(key) for key in keys)
            return await registered(keys=names, args=args, client=client)

        return standalone
    if not hasattr(transport, "script_load"):
        raise ValueError("Redis client does not support Lua script registration")
    cluster: Final = cast(_ClusterScripts, transport)  # cast-ok: RedisCluster omits register_script
    initial_load: Final = cluster.script_load(source)
    load_task: Final = asyncio.ensure_future(initial_load) if isinstance(initial_load, Awaitable) else None

    async def clustered(keys: Sequence[str], args: Sequence[_Arg], client: object = None) -> object:
        from redis.exceptions import NoScriptError

        names: Final = tuple(cache.check_and_fix_namespace(key) for key in keys)
        sha: Final = _TEXT.validate_python(await load_task if load_task is not None else initial_load)
        try:
            return await cluster.evalsha(sha, len(names), *names, *args)
        except NoScriptError:
            reloaded: Final = cluster.script_load(source)
            retry_sha: Final = _TEXT.validate_python(await reloaded if isinstance(reloaded, Awaitable) else reloaded)
            return await cluster.evalsha(retry_sha, len(names), *names, *args)

    return clustered


class _Pubsub(Protocol):
    async def subscribe(self, channel: str) -> object: ...
    async def get_message(self, *, ignore_subscribe_messages: bool, timeout: float) -> object: ...
    async def aclose(self) -> None: ...


class _PublishPipeline(Protocol):
    def publish(self, channel: str, message: str) -> object: ...
    async def execute(self) -> object: ...
    async def __aenter__(self) -> _PublishPipeline: ...
    async def __aexit__(self, *args: object) -> None: ...


class _PubsubClient(Protocol):
    async def publish(self, channel: str, message: str) -> object: ...
    def pipeline(self, *, transaction: bool) -> _PublishPipeline: ...
    def pubsub(self) -> _Pubsub: ...


class _Subscription:
    def __init__(self, pubsub: _Pubsub) -> None:
        self._pubsub: Final = pubsub
        self._closed = False

    async def receive(self, timeout_seconds: float = 0.0) -> str | bytes | None | CacheFailure:
        provisional("subscription.receive")
        if self._closed:
            return CacheFailure("unavailable", "Subscription is closed")
        try:
            reply: Final = await self._pubsub.get_message(ignore_subscribe_messages=True, timeout=timeout_seconds)
            if reply is None:
                return None
            message: Final = TypeAdapter(dict[str, object]).validate_python(reply)
            if message.get("type") != "message":
                return None
            return _PAYLOAD.validate_python(message.get("data"))
        except (ValueError, TypeError, UnicodeError):
            return invalid_reply()
        except Exception as error:  # noqa: BLE001  # backend exceptions must stay behind the cache facade
            return failure(error)

    async def close(self) -> None | CacheFailure:
        provisional("subscription.close")
        if self._closed:
            return None
        self._closed = True
        try:
            await self._pubsub.aclose()
        except Exception as error:  # noqa: BLE001  # backend exceptions must stay behind the cache facade
            return failure(error)
        return None


def _pubsub_client(cache: RedisCache) -> _PubsubClient:
    return cast(  # cast-ok: cache-owned redis client; wire responses are validated separately
        _PubsubClient,
        cache._get_pubsub_client(),  # pyright: ignore[reportPrivateUsage]  # private cache transport
    )


async def publish(cache: RedisCache, channel: str, message: str) -> int | CacheFailure:
    try:
        reply: Final = await _pubsub_client(cache).publish(cache.check_and_fix_namespace(channel), message)
        return validated(integer, reply)
    except Exception as error:  # noqa: BLE001  # backend exceptions must stay behind the cache facade
        return failure(error)


async def publish_many(cache: RedisCache, messages: Sequence[CachePublish]) -> tuple[int, ...] | CacheFailure:
    try:
        async with _pubsub_client(cache).pipeline(transaction=False) as pipe:
            for item in messages:
                pipe.publish(cache.check_and_fix_namespace(item.channel), item.message)
            reply: Final = await pipe.execute()
        return validated(lambda value: counts(value, len(messages)), reply)
    except Exception as error:  # noqa: BLE001  # backend exceptions must stay behind the cache facade
        return failure(error)


async def subscribe(cache: RedisCache, channel: str) -> CacheSubscription | CacheFailure:
    try:
        pubsub: Final = _pubsub_client(cache).pubsub()
    except Exception as error:  # noqa: BLE001  # construction is part of the backend boundary
        return failure(error)
    try:
        await pubsub.subscribe(cache.check_and_fix_namespace(channel))
    except BaseException as error:  # noqa: BLE001  # cancellation must close a partially subscribed connection
        with suppress(Exception):
            await pubsub.aclose()
        if isinstance(error, Exception):
            return failure(error)
        raise
    return _Subscription(pubsub)


class _SyncClusterValues(Protocol):
    def mget_nonatomic(self, *, keys: Sequence[str]) -> object: ...


def cluster_values_sync(cache: RedisCache, keys: Sequence[str]) -> tuple[str | bytes | None, ...]:
    client: Final = cast(_SyncClusterValues, cache.redis_client)  # cast-ok: cache-owned cluster transport
    return _CACHE_VALUES.validate_python(client.mget_nonatomic(keys=keys))


class _ClusterValues(Protocol):
    async def mget_nonatomic(self, *, keys: Sequence[str]) -> object: ...


async def cluster_values(cache: RedisCache, keys: Sequence[str]) -> tuple[str | bytes | None, ...]:
    client: Final = cast(  # cast-ok: RedisClusterCache configures a cluster client; validate its wire response
        _ClusterValues,
        cache._get_async_client(),  # pyright: ignore[reportPrivateUsage]  # cache-owned cluster transport
    )
    return _CACHE_VALUES.validate_python(await client.mget_nonatomic(keys=keys))


async def read_values(cache: RedisCache, keys: Sequence[str]) -> Mapping[str, object] | CacheFailure:
    try:
        namespaced: Final = [cache.check_and_fix_namespace(key) for key in keys]
        from litellm.caching.redis_cache import (
            _run_under_circuit_breaker,  # pyright: ignore[reportPrivateUsage]  # private cache health boundary
        )

        replies: Final = await _run_under_circuit_breaker(
            cache._circuit_breaker,  # pyright: ignore[reportPrivateUsage]  # same breaker as legacy MGET
            "async_read_values",
            lambda: cache._admit_async_call(  # pyright: ignore[reportPrivateUsage]  # preserve cluster connection admission
                lambda: cache._async_run_redis_mget_operation(keys=namespaced)  # pyright: ignore[reportPrivateUsage]  # topology-specific transport
            ),
        )
        values: Final = _CACHE_VALUES.validate_python(replies)
        if len(values) != len(keys):
            return invalid_reply()
        return MappingProxyType(
            {
                key: cache._get_cache_logic(value)  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]  # legacy cache decoder owns response shapes
                for key, value in zip(keys, values)
            }
        )
    except (ValueError, TypeError, UnicodeError, SyntaxError):
        return invalid_reply()
    except Exception as error:  # noqa: BLE001  # backend exceptions must stay behind the cache facade
        return failure(error)


def pool_info(cache: RedisCache) -> CachePoolInfo | CacheFailure | None:
    try:
        client: Final = TypeAdapter(object).validate_python(cache.redis_client)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # legacy transport narrowed to object
        pool: Final[object] = getattr(client, "connection_pool", None)
        if pool is None:
            return None
        maximum: Final = _OPTIONAL_INT.validate_python(getattr(pool, "max_connections", None))
        available: Final[object] = getattr(pool, "_available_connections", None)
        in_use: Final[object] = getattr(pool, "_in_use_connections", None)
        connection_class: Final[object] = getattr(pool, "connection_class", None)
        return CachePoolInfo(
            available_connections=len(available) if isinstance(available, Sized) else None,
            in_use_connections=len(in_use) if isinstance(in_use, Sized) else None,
            max_connections=maximum,
            connection_class=_OPTIONAL_TEXT.validate_python(getattr(connection_class, "__name__", None)),
        )
    except (ValueError, TypeError, UnicodeError):
        return invalid_reply()
    except Exception as error:  # noqa: BLE001  # optional diagnostics cannot leak redis transport exceptions
        return failure(error)
