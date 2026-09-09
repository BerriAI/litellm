"""A Redis for tests that answers like Redis and fails on cue.

``FaultInjectingRedis`` is ``fakeredis.FakeAsyncRedis`` with a per-command schedule of faults.
Each command name maps to an iterator; every call to that command takes the next item and, if
it is an exception, raises it instead of running. Exhausted or unlisted commands fall through to
``default`` (another iterator), and then to running normally against the in-memory store. The exceptions are the real ``redis.exceptions`` types, so the
circuit breaker classifies them exactly as it would a real client's.

``FaultInjectingRedisCache`` is a ``RedisCache`` whose async client is that fake, so the
breaker, namespacing and TTL handling all run unchanged. ``timeout_min_duration`` defaults to
zero so a burst of injected timeouts opens the breaker without waiting out the real 5 s gate.

    cache = FaultInjectingRedisCache(FaultInjectingRedis({"INCRBYFLOAT": always(RedisTimeoutError("injected"))}))
    cache = FaultInjectingRedisCache(FaultInjectingRedis({"GET": sequence(RedisConnectionError("down"), None)}))
    cache = FaultInjectingRedisCache(FaultInjectingRedis({}, default=always(RedisTimeoutError("slow"))))
"""

from collections.abc import Iterator, Mapping
from itertools import repeat
from typing import Final

import fakeredis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from litellm.caching.redis_cache import RedisCache, RedisCircuitBreaker
from litellm.constants import (
    REDIS_CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    REDIS_CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
)

Fault = RedisTimeoutError | RedisConnectionError | None


def always(fault: Fault) -> Iterator[Fault]:
    return repeat(fault)


def sequence(*faults: Fault) -> Iterator[Fault]:
    return iter(faults)


class FaultInjectingRedis(fakeredis.FakeAsyncRedis):
    def __init__(self, faults: Mapping[str, Iterator[Fault]], default: Iterator[Fault] | None = None) -> None:
        super().__init__()
        self._faults: Final[Mapping[str, Iterator[Fault]]] = dict(faults)
        self._default: Final[Iterator[Fault]] = default if default is not None else iter(())
        self.injected: tuple[str, ...] = ()

    async def execute_command(self, *args: object, **options: object) -> object:
        command: Final = str(args[0]).upper()
        fault: Final = next(self._faults.get(command, self._default), None)
        if fault is None:
            return await super().execute_command(*args, **options)
        self.injected = (*self.injected, command)
        raise fault


class FaultInjectingRedisCache(RedisCache):
    def __init__(self, client: FaultInjectingRedis, timeout_min_duration: float = 0.0) -> None:
        super().__init__(host="127.0.0.1", port=1, socket_timeout=0.01)
        self._client: Final = client
        self._circuit_breaker = RedisCircuitBreaker(
            failure_threshold=REDIS_CIRCUIT_BREAKER_FAILURE_THRESHOLD,
            recovery_timeout=REDIS_CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
            timeout_min_duration=timeout_min_duration,
        )

    def _setup_health_pings(self) -> None:
        return

    def init_async_client(self) -> FaultInjectingRedis:
        self.redis_async_client = self._client
        return self._client
