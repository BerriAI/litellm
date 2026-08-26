"""
Redis Cache implementation

Has 4 primary methods:
    - set_cache
    - get_cache
    - async_set_cache
    - async_get_cache
"""

import ast
import asyncio
import functools
import hashlib
import inspect
import json
import time
from collections.abc import Awaitable, Callable, Sequence
from contextvars import ContextVar
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Final, TypeVar, cast

import litellm
from litellm._logging import print_verbose, verbose_logger
from litellm.constants import (
    DEFAULT_REDIS_MAJOR_VERSION,
    REDIS_CIRCUIT_BREAKER_ENABLED,
    REDIS_CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    REDIS_CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
)
from litellm.litellm_core_utils.core_helpers import _get_parent_otel_span_from_kwargs
from litellm.litellm_core_utils.coroutine_checker import coroutine_checker
from litellm.types.caching import (
    RedisPipelineIncrementOperation,
    RedisPipelineLpopOperation,
    RedisPipelineRpushOperation,
)
from litellm.types.services import ServiceTypes

from .base_cache import BaseCache

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span
    from redis.asyncio import Redis, RedisCluster
    from redis.asyncio.client import Pipeline
    from redis.asyncio.cluster import ClusterPipeline

    pipeline = Pipeline
    cluster_pipeline = ClusterPipeline
    async_redis_client = Redis
    async_redis_cluster_client = RedisCluster
    Span = _Span
else:
    pipeline = Any
    cluster_pipeline = Any
    async_redis_client = Any
    async_redis_cluster_client = Any
    Span = Any


def _get_call_stack_info(num_frames: int = 2) -> str:
    """
    Get the function names from the previous 1-2 functions in the call stack.

    Args:
        num_frames: Number of previous frames to include (default: 2)

    Returns:
        A string with format "current_function <- caller_function [<- grandparent_function]"
    """
    try:
        current_frame: Final = inspect.currentframe()
        if current_frame is None:
            return "unknown"

        # Skip this function and the immediate caller (which sets call_type)
        f_back: Final = current_frame.f_back
        if f_back is None:
            return "unknown"
        frame = f_back.f_back
        if frame is None:
            return "unknown"
        function_names: Final = []

        for _ in range(num_frames):
            if frame is None:
                break
            func_name = frame.f_code.co_name
            function_names.append(func_name)
            frame = frame.f_back

        if not function_names:
            return "unknown"

        return " <- ".join(function_names)
    except Exception:
        return "unknown"


class RedisCircuitBreaker:
    """
    Tracks Redis health for a RedisCache instance.

    States:
      CLOSED    - normal, Redis is called
      OPEN      - Redis is down, raise immediately (no network call)
      HALF_OPEN - recovery probe: allow one request through

    Transitions:
      CLOSED    -> OPEN      after failure_threshold consecutive failures
      OPEN      -> HALF_OPEN after recovery_timeout seconds
      HALF_OPEN -> CLOSED    on success
      HALF_OPEN -> OPEN      on failure (resets timer)
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int,
        recovery_timeout: int,
        enabled: bool = True,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.enabled = enabled
        self._failure_count = 0
        self._opened_at: float | None = None
        self._state = self.CLOSED

    def is_open(self) -> bool:
        """Returns True if Redis calls should be skipped."""
        if not self.enabled:
            return False
        if self._state == self.HALF_OPEN:
            # Probe already in flight — fast-fail all concurrent requests.
            # Only the one call that caused the OPEN→HALF_OPEN transition
            # (which returned False) is the designated probe.
            return True
        if self._state == self.OPEN:
            if time.time() - (self._opened_at or 0) > self.recovery_timeout:
                self._state = self.HALF_OPEN
                return False  # this caller is the designated probe
            return True
        return False

    def record_failure(self) -> None:
        if not self.enabled:
            return
        self._failure_count += 1
        self._opened_at = time.time()
        if self._failure_count >= self.failure_threshold:
            if self._state != self.OPEN:
                verbose_logger.warning(
                    "Redis circuit breaker OPENED after %d consecutive failures — fast-failing Redis calls for %ds",
                    self._failure_count,
                    self.recovery_timeout,
                )
            self._state = self.OPEN

    def record_success(self) -> None:
        if not self.enabled:
            return
        if self._state == self.HALF_OPEN:
            verbose_logger.info("Redis circuit breaker CLOSED — Redis recovered")
        self._failure_count = 0
        self._state = self.CLOSED


_RedisCallResult = TypeVar("_RedisCallResult")


_swallowed_redis_failures: Final[ContextVar[int]] = ContextVar("litellm_swallowed_redis_failures", default=0)


def _opaque_kwarg_key(value: object) -> str:
    return f"{type(value).__name__}-{id(value)}"


@functools.lru_cache(maxsize=1)
def _redis_health_error_types() -> tuple[type, ...]:
    """Exception types that mean the Redis backend itself is unhealthy.

    Command and data errors say nothing about connectivity: an INCR against a non-numeric
    value or an undecodable cached entry is a request problem, and counting those would let
    a caller trip the shared breaker on demand, dropping rate limiting to per-process
    counters that spreading traffic across replicas can outrun.

    Imported lazily because this module is reachable from a base ``import litellm`` while
    redis is not a base dependency.
    """
    from redis.exceptions import BusyLoadingError, ClusterDownError
    from redis.exceptions import ConnectionError as RedisConnectionError
    from redis.exceptions import TimeoutError as RedisTimeoutError

    return (RedisConnectionError, RedisTimeoutError, BusyLoadingError, ClusterDownError, OSError, asyncio.TimeoutError)


def _is_redis_health_failure(exc: BaseException) -> bool:
    """True when ``exc`` indicates Redis is unreachable rather than the request being bad."""
    try:
        return isinstance(exc, _redis_health_error_types())
    except ImportError:
        return True


def _record_swallowed_redis_failure(breaker: RedisCircuitBreaker, exc: BaseException) -> None:
    """Record a Redis failure that the calling method is about to swallow.

    The marker is a ContextVar rather than a counter on the breaker because breakers are
    shared by every concurrent caller. A plain shared counter cannot tell "my call failed"
    from "some other in-flight call failed", so a success overlapping someone else's
    failure would be discarded and a Redis that is answering would still be evicted.
    asyncio gives each task its own copy of the context, so this is per-call.
    """
    if not _is_redis_health_failure(exc):
        return
    breaker.record_failure()
    _swallowed_redis_failures.set(_swallowed_redis_failures.get() + 1)


async def _run_under_circuit_breaker(
    breaker: RedisCircuitBreaker,
    name: str,
    call: Callable[[], Awaitable[_RedisCallResult]],
) -> _RedisCallResult:
    """Run one Redis coroutine under a circuit breaker.

    Shared by the method decorator and the Lua script executor so both feed the same
    health signal. Success is recorded only when nothing failed while ``call`` ran,
    because several Redis methods catch their own connection errors and return a default.
    """
    if breaker.is_open():
        raise Exception(f"Redis circuit breaker is open — skipping {name}")
    swallowed_before: Final = _swallowed_redis_failures.get()
    try:
        result: Final = await call()
    except Exception as e:
        if _is_redis_health_failure(e):
            breaker.record_failure()
        raise
    if _swallowed_redis_failures.get() == swallowed_before:
        breaker.record_success()
    return result


def _redis_circuit_breaker_guard(method):
    """
    Decorator for RedisCache async methods.
    Checks the circuit breaker before each call; records success/failure after.
    Does not apply to ping/disconnect/test_connection (health/teardown must always run).

    A returning method is not proof of a healthy Redis: several methods catch their own
    connection errors and return a default so callers degrade rather than fail. Counting
    those as successes reset the failure streak on every request, so the breaker could
    never open and Redis was never taken out of the pool. Success is therefore recorded
    only when no failure was registered while the method ran.
    """

    @functools.wraps(method)
    async def wrapper(self, *args, **kwargs):
        return await _run_under_circuit_breaker(
            self._circuit_breaker, method.__name__, lambda: method(self, *args, **kwargs)
        )

    return wrapper


class RedisCache(BaseCache):
    # if users don't provider one, use the default litellm cache

    def __init__(
        self,
        host=None,
        port=None,
        password=None,
        redis_flush_size: int | None = 100,
        namespace: str | None = None,
        startup_nodes: list | None = None,  # for redis-cluster
        socket_timeout: float | None = 5.0,  # default 5 second timeout
        **kwargs,
    ):
        from litellm._service_logger import ServiceLogging

        from .._redis import get_redis_client, get_redis_connection_pool

        redis_kwargs: Final = {}
        if host is not None:
            redis_kwargs["host"] = host
        if port is not None:
            redis_kwargs["port"] = port
        if password is not None:
            redis_kwargs["password"] = password
        if startup_nodes is not None:
            redis_kwargs["startup_nodes"] = startup_nodes
        if socket_timeout is not None:
            redis_kwargs["socket_timeout"] = socket_timeout

        ### HEALTH MONITORING OBJECT ###
        if kwargs.get("service_logger_obj", None) is not None and isinstance(
            kwargs["service_logger_obj"], ServiceLogging
        ):
            self.service_logger_obj = kwargs.pop("service_logger_obj")
        else:
            self.service_logger_obj = ServiceLogging()

        redis_kwargs.update(kwargs)
        self.redis_client = get_redis_client(**redis_kwargs)
        self.redis_async_client: async_redis_client | async_redis_cluster_client | None = None
        self.redis_kwargs = redis_kwargs
        self.async_redis_conn_pool = get_redis_connection_pool(**redis_kwargs)

        # redis namespaces
        self.namespace = namespace
        # for high traffic, we store the redis results in memory and then batch write to redis
        self.redis_batch_writing_buffer: list = []
        if redis_flush_size is None:
            self.redis_flush_size: int = 100
        else:
            self.redis_flush_size = redis_flush_size
        self.redis_version = "Unknown"
        try:
            if not coroutine_checker.is_async_callable(self.redis_client):
                self.redis_version = self.redis_client.info()["redis_version"]
        except Exception:
            pass

        self._circuit_breaker = RedisCircuitBreaker(
            failure_threshold=REDIS_CIRCUIT_BREAKER_FAILURE_THRESHOLD,
            recovery_timeout=REDIS_CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
            enabled=REDIS_CIRCUIT_BREAKER_ENABLED,
        )

        self._setup_health_pings()

        if litellm.default_redis_ttl is not None:
            super().__init__(default_ttl=int(litellm.default_redis_ttl))
        else:
            super().__init__()  # defaults to 60s

    def _setup_health_pings(self):
        """Setup async and sync health pings for Redis."""
        # ASYNC HEALTH PING
        try:
            _ = asyncio.get_running_loop().create_task(self.ping())
        except Exception as e:
            if "no running event loop" in str(e):
                verbose_logger.debug("Ignoring async redis ping. No running event loop.")
            else:
                verbose_logger.error(
                    "Error connecting to Async Redis client - %s",
                    e,
                    extra={"error": str(e)},
                )
                self._handle_async_ping_error(e)

        # SYNC HEALTH PING
        try:
            if hasattr(self.redis_client, "ping"):
                self.redis_client.ping()
        except Exception as e:
            verbose_logger.error("Error connecting to Sync Redis client", extra={"error": str(e)})
            self._handle_sync_ping_error(e)

    def _handle_async_ping_error(self, e: Exception):
        """Handle async ping error with service failure hook."""
        try:
            loop: Final = asyncio.get_running_loop()
            start_time: Final = time.time()
            end_time: Final = start_time
            loop.create_task(
                self.service_logger_obj.async_service_failure_hook(
                    service=ServiceTypes.REDIS,
                    duration=end_time - start_time,
                    error=e,
                    call_type="redis_async_ping",
                )
            )
        except Exception:
            pass

    def _handle_sync_ping_error(self, e: Exception):
        """Handle sync ping error with service failure hook."""
        try:
            loop: Final = asyncio.get_running_loop()
            start_time: Final = time.time()
            end_time: Final = start_time
            loop.create_task(
                self.service_logger_obj.async_service_failure_hook(
                    service=ServiceTypes.REDIS,
                    duration=end_time - start_time,
                    error=e,
                    call_type="redis_sync_ping",
                )
            )
        except Exception:
            pass

    def _get_async_client_cache_key(self) -> str:
        """
        Generate a cache key for the async Redis client based on connection parameters.
        This ensures different Redis configurations use different cached clients.
        """
        # Sort keys to ensure consistent hash regardless of parameter order
        sorted_kwargs: Final = sorted(self.redis_kwargs.items())
        kwargs_str: Final = json.dumps(sorted_kwargs, sort_keys=True, default=_opaque_kwarg_key)
        kwargs_hash: Final = hashlib.sha256(kwargs_str.encode()).hexdigest()[:16]
        return f"async-redis-client-{kwargs_hash}"

    def init_async_client(
        self,
    ) -> async_redis_client | async_redis_cluster_client:
        from litellm import in_memory_llm_clients_cache

        from .._redis import get_redis_async_client, get_redis_connection_pool

        cache_key: Final = self._get_async_client_cache_key()
        cached_client: Final = in_memory_llm_clients_cache.get_cache(key=cache_key)
        if cached_client is not None:
            redis_async_client = cast(async_redis_client | async_redis_cluster_client, cached_client)
        else:
            # Create new connection pool and client for current event loop
            self.async_redis_conn_pool = get_redis_connection_pool(**self.redis_kwargs)
            redis_async_client = get_redis_async_client(connection_pool=self.async_redis_conn_pool, **self.redis_kwargs)
            in_memory_llm_clients_cache.set_cache(key=cache_key, value=redis_async_client)

        self.redis_async_client = redis_async_client
        return redis_async_client

    def check_and_fix_namespace(self, key: str) -> str:
        """
        Make sure each key starts with the given namespace
        """
        if key is None:
            return key
        if self.namespace and not key.startswith(self.namespace + ":"):
            key = self.namespace + ":" + key

        return key

    def _parse_redis_major_version(self) -> int:
        """
        Parse Redis version to extract the major version number.

        Handles multiple version formats:
        - Strings: "7.0.0", "6", "7.0.0-rc1", " 7.0.0 "
        - Floats: 7.0 (e.g., from AWS ElastiCache Valkey)
        - Integers: 7
        - Malformed: "latest", "", "Unknown" (defaults to DEFAULT_REDIS_MAJOR_VERSION)

        Returns:
            int: The major version number (defaults to DEFAULT_REDIS_MAJOR_VERSION if unparseable)
        """
        if self.redis_version == "Unknown":
            return DEFAULT_REDIS_MAJOR_VERSION

        try:
            version_str: Final = str(self.redis_version).strip()
            # Handle cases where there's no dot (e.g., "7" or 7)
            if "." in version_str:
                major_version = int(version_str.split(".")[0])
            else:
                # Direct integer or single-digit string
                major_version = int(float(version_str))
            return major_version
        except (ValueError, AttributeError):
            # Fallback for unparseable versions (e.g., "v7.0.0", "latest")
            return DEFAULT_REDIS_MAJOR_VERSION

    def set_cache(self, key, value, **kwargs):
        ttl: Final = self.get_ttl(**kwargs)
        print_verbose(f"Set Redis Cache: key: {key}\nValue {value}\nttl={ttl}, redis_version={self.redis_version}")
        key = self.check_and_fix_namespace(key=key)
        try:
            start_time: Final = time.time()
            self.redis_client.set(name=key, value=str(value), ex=ttl)
            end_time: Final = time.time()
            _duration: Final = end_time - start_time
            self.service_logger_obj.service_success_hook(
                service=ServiceTypes.REDIS,
                duration=_duration,
                call_type=f"set_cache <- {_get_call_stack_info()}",
                start_time=start_time,
                end_time=end_time,
            )
        except Exception as e:
            # NON blocking - notify users Redis is throwing an exception
            print_verbose(f"litellm.caching.caching: set() - Got exception from REDIS : {e}")

    def increment_cache(self, key, value: int, ttl: float | None = None, **kwargs) -> int:
        _redis_client: Final = self.redis_client
        start_time = time.time()
        set_ttl: Final = self.get_ttl(ttl=ttl)
        key = self.check_and_fix_namespace(key=key)
        try:
            start_time = time.time()
            result: Final[int] = _redis_client.incr(name=key, amount=value)
            end_time = time.time()
            _duration = end_time - start_time
            self.service_logger_obj.service_success_hook(
                service=ServiceTypes.REDIS,
                duration=_duration,
                call_type=f"increment_cache <- {_get_call_stack_info()}",
                start_time=start_time,
                end_time=end_time,
            )

            if set_ttl is not None:
                # check if key already has ttl, if not -> set ttl
                start_time = time.time()
                current_ttl: Final = _redis_client.ttl(key)
                end_time = time.time()
                _duration = end_time - start_time
                self.service_logger_obj.service_success_hook(
                    service=ServiceTypes.REDIS,
                    duration=_duration,
                    call_type=f"increment_cache_ttl <- {_get_call_stack_info()}",
                    start_time=start_time,
                    end_time=end_time,
                )
                if current_ttl == -1:
                    # Key has no expiration
                    start_time = time.time()
                    _redis_client.expire(key, set_ttl)
                    end_time = time.time()
                    _duration = end_time - start_time
                    self.service_logger_obj.service_success_hook(
                        service=ServiceTypes.REDIS,
                        duration=_duration,
                        call_type=f"increment_cache_expire <- {_get_call_stack_info()}",
                        start_time=start_time,
                        end_time=end_time,
                    )
            return result
        except Exception as e:
            ## LOGGING ##
            end_time = time.time()
            _duration = end_time - start_time
            verbose_logger.error(
                "LiteLLM Redis Caching: increment_cache() - Got exception from REDIS %s, Writing value=%s",
                str(e),
                value,
            )
            raise e

    @_redis_circuit_breaker_guard
    async def async_scan_iter(self, pattern: str, count: int = 100) -> list:
        start_time: Final = time.time()
        try:
            keys: Final = []
            _redis_client: Final = self.init_async_client()
            if not hasattr(_redis_client, "scan_iter"):
                verbose_logger.debug(
                    "Redis client does not support scan_iter, potentially using Redis Cluster. Returning empty list."
                )
                return []

            pattern = self.check_and_fix_namespace(key=pattern)
            async for key in _redis_client.scan_iter(match=pattern + "*", count=count):
                keys.append(key)
                if len(keys) >= count:
                    break

            ## LOGGING ##
            end_time = time.time()
            _duration = end_time - start_time
            asyncio.create_task(
                self.service_logger_obj.async_service_success_hook(
                    service=ServiceTypes.REDIS,
                    duration=_duration,
                    call_type=f"async_scan_iter <- {_get_call_stack_info()}",
                    start_time=start_time,
                    end_time=end_time,
                )
            )  # DO NOT SLOW DOWN CALL B/C OF THIS
            return keys
        except Exception as e:
            # NON blocking - notify users Redis is throwing an exception
            ## LOGGING ##
            end_time = time.time()
            _duration = end_time - start_time
            asyncio.create_task(
                self.service_logger_obj.async_service_failure_hook(
                    service=ServiceTypes.REDIS,
                    duration=_duration,
                    error=e,
                    call_type=f"async_scan_iter <- {_get_call_stack_info()}",
                    start_time=start_time,
                    end_time=end_time,
                )
            )
            raise e

    def async_register_script(self, script: str) -> Callable[..., Awaitable[Any]]:
        """
        Register a Lua script with Redis asynchronously.
        Works with both standalone Redis and Redis Cluster.

        The returned callable namespaces every key it is invoked with, so Lua
        scripts hit the same prefixed keys as get/set/increment. Without this,
        scripts would operate on raw keys while the rest of the cache uses the
        namespace, leaving rate-limit and lock keys outside the configured prefix.

        Registration is deferred to call time and cached per running event loop
        (via in_memory_llm_clients_cache, which keys its entries on the loop). A
        registered script is bound to the connection of the loop it was created
        on; awaiting it from another loop raises "got Future attached to a
        different loop". Binding lazily on the calling loop gives the script the
        same per-loop scoping init_async_client already gives the clients, so a
        script registered once at startup is never reused across loops.

        Args:
            script (str): The Lua script to register

        Returns:
            A callable ``(keys, args, client=None)`` that runs the script
            against the calling loop's Redis client.
        """
        # Keyed by connection params and namespace as well as the script, so
        # two RedisCache instances pointing at different servers or using
        # different key prefixes never share an executor; in_memory_llm_clients_cache
        # then adds the running loop, completing the per-(client, namespace, loop)
        # scoping.
        script_cache_key: Final = (
            f"redis-registered-script-{self._get_async_client_cache_key()}-"
            f"{self.namespace}-{hashlib.sha256(script.encode()).hexdigest()[:16]}"
        )

        async def run_script(
            keys: Sequence[str],
            args: Sequence[str | bytes | int | float],
            client: object = None,
        ) -> object:
            async def execute() -> object:
                executor: Callable[..., Awaitable[Any]] | None = litellm.in_memory_llm_clients_cache.get_cache(
                    key=script_cache_key
                )
                if executor is None:
                    executor = self._register_script_for_current_loop(script)
                    litellm.in_memory_llm_clients_cache.set_cache(key=script_cache_key, value=executor)
                return await executor(keys=keys, args=args, client=client)

            return await _run_under_circuit_breaker(self._circuit_breaker, "run_script", execute)

        return run_script

    def _register_script_for_current_loop(self, script: str) -> Callable[..., Awaitable[Any]]:
        """
        Register the script against the current event loop's Redis client.

        Kept separate from async_register_script so each loop caches its own
        executor; see that method for why the binding must be per loop.
        """
        _redis_client: Final[Any] = self.init_async_client()
        if hasattr(_redis_client, "register_script"):
            registered_script: Final = _redis_client.register_script(script)

            async def standalone_executor(
                keys: Sequence[str],
                args: Sequence[str | bytes | int | float],
                client: object = None,
            ) -> object:
                namespaced_keys: Final = tuple(self.check_and_fix_namespace(key=key) for key in keys)
                return await registered_script(keys=namespaced_keys, args=args, client=client)

            return standalone_executor

        if hasattr(_redis_client, "script_load"):
            script_sha: Final = _redis_client.script_load(script)

            async def cluster_executor(
                keys: Sequence[str],
                args: Sequence[str | bytes | int | float],
                client: object = None,
            ) -> object:
                namespaced_keys: Final = tuple(self.check_and_fix_namespace(key=key) for key in keys)
                return await _redis_client.evalsha(script_sha, len(namespaced_keys), *namespaced_keys, *args)

            return cluster_executor

        raise ValueError("Redis client does not support Lua script registration")

    @_redis_circuit_breaker_guard
    async def async_set_cache(self, key, value, **kwargs):
        from redis.asyncio import Redis

        if key is None:
            verbose_logger.debug(
                "LiteLLM Redis Caching: async set() skipped — key is None, value=%r",
                value,
            )
            return None

        start_time: Final = time.time()
        try:
            _redis_client: Final[Redis] = self.init_async_client()
        except Exception as e:
            end_time = time.time()
            _duration = end_time - start_time
            asyncio.create_task(
                self.service_logger_obj.async_service_failure_hook(
                    service=ServiceTypes.REDIS,
                    duration=_duration,
                    error=e,
                    start_time=start_time,
                    end_time=end_time,
                    parent_otel_span=_get_parent_otel_span_from_kwargs(kwargs),
                    call_type=f"async_set_cache <- {_get_call_stack_info()}",
                )
            )
            verbose_logger.error(
                "LiteLLM Redis Caching: async set() - Got exception from REDIS %s, key=%r, value=%r",
                str(e),
                key,
                value,
            )
            raise e

        key = self.check_and_fix_namespace(key=key)
        ttl: Final = self.get_ttl(**kwargs)
        nx: Final = kwargs.get("nx", False)
        print_verbose(f"Set ASYNC Redis Cache: key: {key}\nValue {value}\nttl={ttl}")

        try:
            if not hasattr(_redis_client, "set"):
                raise Exception("Redis client cannot set cache. Attribute not found.")
            result: Final = await _redis_client.set(
                name=key,
                value=json.dumps(value),
                nx=nx,
                ex=ttl,
            )
            print_verbose(f"Successfully Set ASYNC Redis Cache: key: {key}\nValue {value}\nttl={ttl}")
            end_time = time.time()
            _duration = end_time - start_time
            asyncio.create_task(
                self.service_logger_obj.async_service_success_hook(
                    service=ServiceTypes.REDIS,
                    duration=_duration,
                    call_type=f"async_set_cache <- {_get_call_stack_info()}",
                    start_time=start_time,
                    end_time=end_time,
                    parent_otel_span=_get_parent_otel_span_from_kwargs(kwargs),
                    event_metadata={"key": key},
                )
            )
            return result
        except Exception as e:
            end_time = time.time()
            _duration = end_time - start_time
            asyncio.create_task(
                self.service_logger_obj.async_service_failure_hook(
                    service=ServiceTypes.REDIS,
                    duration=_duration,
                    error=e,
                    call_type=f"async_set_cache <- {_get_call_stack_info()}",
                    start_time=start_time,
                    end_time=end_time,
                    parent_otel_span=_get_parent_otel_span_from_kwargs(kwargs),
                    event_metadata={"key": key},
                )
            )
            verbose_logger.error(
                "LiteLLM Redis Caching: async set() - Got exception from REDIS %s, Writing value=%s",
                str(e),
                value,
            )
            _record_swallowed_redis_failure(self._circuit_breaker, e)

    async def _pipeline_helper(
        self,
        pipe: pipeline | cluster_pipeline,
        cache_list: Sequence[tuple[str, object]],
        ttl: float | None,
    ) -> list:
        """
        Helper function for executing a pipeline of set operations on Redis
        """
        ttl = self.get_ttl(ttl=ttl)
        # Iterate through each key-value pair in the cache_list and set them in the pipeline.
        for cache_key, cache_value in cache_list:
            cache_key = self.check_and_fix_namespace(key=cache_key)
            print_verbose(f"Set ASYNC Redis Cache PIPELINE: key: {cache_key}\nValue {cache_value}\nttl={ttl}")
            json_cache_value = json.dumps(cache_value)
            # Set the value with a TTL if it's provided.
            _td: timedelta | None = None
            if ttl is not None:
                _td = timedelta(seconds=ttl)
            pipe.set(
                name=cache_key,
                value=json_cache_value,
                ex=_td,
            )
        # Execute the pipeline and return the results.
        results: Final = await pipe.execute()
        return results

    @_redis_circuit_breaker_guard
    async def async_set_cache_pipeline(
        self, cache_list: Sequence[tuple[str, object]], ttl: float | None = None, **kwargs
    ):
        """
        Use Redis Pipelines for bulk write operations
        """
        # don't waste a network request if there's nothing to set
        if len(cache_list) == 0:
            return

        _redis_client: Final = self.init_async_client()
        start_time: Final = time.time()

        print_verbose(f"Set Async Redis Cache: key list: {cache_list}\nttl={ttl}, redis_version={self.redis_version}")
        cache_value: Final = None
        try:
            async with _redis_client.pipeline(transaction=False) as pipe:
                results: Final = await self._pipeline_helper(pipe, cache_list, ttl)

            print_verbose(f"pipeline results: {results}")
            # Optionally, you could process 'results' to make sure that all set operations were successful.
            ## LOGGING ##
            end_time = time.time()
            _duration = end_time - start_time
            asyncio.create_task(
                self.service_logger_obj.async_service_success_hook(
                    service=ServiceTypes.REDIS,
                    duration=_duration,
                    call_type=f"async_set_cache_pipeline <- {_get_call_stack_info()}",
                    start_time=start_time,
                    end_time=end_time,
                    parent_otel_span=_get_parent_otel_span_from_kwargs(kwargs),
                )
            )
            return
        except Exception as e:
            ## LOGGING ##
            end_time = time.time()
            _duration = end_time - start_time
            asyncio.create_task(
                self.service_logger_obj.async_service_failure_hook(
                    service=ServiceTypes.REDIS,
                    duration=_duration,
                    error=e,
                    call_type=f"async_set_cache_pipeline <- {_get_call_stack_info()}",
                    start_time=start_time,
                    end_time=end_time,
                    parent_otel_span=_get_parent_otel_span_from_kwargs(kwargs),
                )
            )

            verbose_logger.error(
                "LiteLLM Redis Caching: async set_cache_pipeline() - Got exception from REDIS %s, Writing value=%s",
                str(e),
                cache_value,
            )
            _record_swallowed_redis_failure(self._circuit_breaker, e)

    async def _set_cache_sadd_helper(
        self,
        redis_client: async_redis_client,
        key: str,
        value: list,
        ttl: float | None,
    ) -> None:
        """Helper function for async_set_cache_sadd. Separated for testing."""
        ttl = self.get_ttl(ttl=ttl)
        try:
            await redis_client.sadd(key, *value)
            if ttl is not None:
                _td: Final = timedelta(seconds=ttl)
                await redis_client.expire(key, _td)
        except Exception:
            raise

    @_redis_circuit_breaker_guard
    async def async_set_cache_sadd(self, key, value: list, ttl: float | None, **kwargs):
        from redis.asyncio import Redis

        start_time: Final = time.time()
        try:
            _redis_client: Final[Redis] = self.init_async_client()
        except Exception as e:
            end_time = time.time()
            _duration = end_time - start_time
            asyncio.create_task(
                self.service_logger_obj.async_service_failure_hook(
                    service=ServiceTypes.REDIS,
                    duration=_duration,
                    error=e,
                    start_time=start_time,
                    end_time=end_time,
                    parent_otel_span=_get_parent_otel_span_from_kwargs(kwargs),
                    call_type=f"async_set_cache_sadd <- {_get_call_stack_info()}",
                )
            )
            # NON blocking - notify users Redis is throwing an exception
            verbose_logger.error(
                "LiteLLM Redis Caching: async set() - Got exception from REDIS %s, Writing value=%s",
                str(e),
                value,
            )
            raise e

        key = self.check_and_fix_namespace(key=key)
        print_verbose(f"Set ASYNC Redis Cache: key: {key}\nValue {value}\nttl={ttl}")
        try:
            await self._set_cache_sadd_helper(redis_client=_redis_client, key=key, value=value, ttl=ttl)
            print_verbose(f"Successfully Set ASYNC Redis Cache SADD: key: {key}\nValue {value}\nttl={ttl}")
            end_time = time.time()
            _duration = end_time - start_time
            asyncio.create_task(
                self.service_logger_obj.async_service_success_hook(
                    service=ServiceTypes.REDIS,
                    duration=_duration,
                    call_type=f"async_set_cache_sadd <- {_get_call_stack_info()}",
                    start_time=start_time,
                    end_time=end_time,
                    parent_otel_span=_get_parent_otel_span_from_kwargs(kwargs),
                )
            )
        except Exception as e:
            end_time = time.time()
            _duration = end_time - start_time
            asyncio.create_task(
                self.service_logger_obj.async_service_failure_hook(
                    service=ServiceTypes.REDIS,
                    duration=_duration,
                    error=e,
                    call_type=f"async_set_cache_sadd <- {_get_call_stack_info()}",
                    start_time=start_time,
                    end_time=end_time,
                    parent_otel_span=_get_parent_otel_span_from_kwargs(kwargs),
                )
            )
            # NON blocking - notify users Redis is throwing an exception
            verbose_logger.error(
                "LiteLLM Redis Caching: async set_cache_sadd() - Got exception from REDIS %s, Writing value=%s",
                str(e),
                value,
            )
            _record_swallowed_redis_failure(self._circuit_breaker, e)

    @_redis_circuit_breaker_guard
    async def batch_cache_write(self, key, value, **kwargs):
        print_verbose(
            f"in batch cache writing for redis buffer size={len(self.redis_batch_writing_buffer)}",
        )
        key = self.check_and_fix_namespace(key=key)
        self.redis_batch_writing_buffer.append((key, value))
        if len(self.redis_batch_writing_buffer) >= self.redis_flush_size:
            await self.flush_cache_buffer()  # logging done in here

    @_redis_circuit_breaker_guard
    async def async_increment(
        self,
        key,
        value: float,
        ttl: int | None = None,
        parent_otel_span: Span | None = None,
        refresh_ttl: bool = False,
    ) -> float:
        from redis.asyncio import Redis

        _redis_client: Final[Redis] = self.init_async_client()
        start_time: Final = time.time()
        _used_ttl: Final = self.get_ttl(ttl=ttl)
        key = self.check_and_fix_namespace(key=key)
        try:
            result: Final = await _redis_client.incrbyfloat(name=key, amount=value)
            if _used_ttl is not None:
                if refresh_ttl:
                    await _redis_client.expire(key, _used_ttl)
                else:
                    current_ttl: Final = await _redis_client.ttl(key)
                    if current_ttl == -1:
                        await _redis_client.expire(key, _used_ttl)

            ## LOGGING ##
            end_time = time.time()
            _duration = end_time - start_time

            asyncio.create_task(
                self.service_logger_obj.async_service_success_hook(
                    service=ServiceTypes.REDIS,
                    duration=_duration,
                    call_type=f"async_increment <- {_get_call_stack_info()}",
                    start_time=start_time,
                    end_time=end_time,
                    parent_otel_span=parent_otel_span,
                )
            )
            return result
        except Exception as e:
            ## LOGGING ##
            end_time = time.time()
            _duration = end_time - start_time
            asyncio.create_task(
                self.service_logger_obj.async_service_failure_hook(
                    service=ServiceTypes.REDIS,
                    duration=_duration,
                    error=e,
                    call_type=f"async_increment <- {_get_call_stack_info()}",
                    start_time=start_time,
                    end_time=end_time,
                    parent_otel_span=parent_otel_span,
                )
            )
            verbose_logger.error(
                "LiteLLM Redis Caching: async async_increment() - Got exception from REDIS %s, Writing value=%s",
                str(e),
                value,
            )
            raise e

    @_redis_circuit_breaker_guard
    async def async_set_max(
        self,
        key: str,
        value: float,
        ttl: int | None = None,
    ) -> float | None:
        """Atomically set ``key`` to ``value`` only when ``value`` is greater
        than the stored value (or the key is unset), refreshing the TTL.

        Monotonic by construction: it never lowers the stored value, so a repair
        that writes an authoritative-but-slightly-stale total cannot clobber a
        concurrent increment that has already pushed the counter higher. The
        GET/compare/SET runs in a single Lua call, so it is also atomic across
        racing callers and pods. Returns the resulting value.
        """
        _redis_client: Final = self.init_async_client()
        _used_ttl: Final = self.get_ttl(ttl=ttl)
        key = self.check_and_fix_namespace(key=key)
        lua: Final = (
            "local cur = redis.call('GET', KEYS[1]) "
            "if cur == false or tonumber(cur) < tonumber(ARGV[1]) then "
            "redis.call('SET', KEYS[1], ARGV[1]) "
            "if tonumber(ARGV[2]) > 0 then redis.call('EXPIRE', KEYS[1], ARGV[2]) end "
            "return ARGV[1] end "
            "return cur"
        )
        result = cast(
            "str | bytes | int | float | None",
            await _redis_client.eval(lua, 1, key, str(value), str(int(_used_ttl or 0))),
        )
        if result is None:
            return None
        if isinstance(result, bytes):
            result = result.decode()
        return float(result)

    async def flush_cache_buffer(self):
        print_verbose(f"flushing to redis....reached size of buffer {len(self.redis_batch_writing_buffer)}")
        await self.async_set_cache_pipeline(self.redis_batch_writing_buffer)
        self.redis_batch_writing_buffer = []

    def _get_cache_logic(self, cached_response: Any):
        """
        Common 'get_cache_logic' across sync + async redis client implementations
        """
        if cached_response is None:
            return cached_response
        # cached_response is in `b{} convert it to ModelResponse
        cached_response = cached_response.decode("utf-8")  # Convert bytes to string
        try:
            cached_response = json.loads(cached_response)  # Convert string to dictionary
        except Exception:
            cached_response = ast.literal_eval(cached_response)
        return cached_response

    def get_cache(self, key, parent_otel_span: Span | None = None, **kwargs):
        try:
            key = self.check_and_fix_namespace(key=key)
            print_verbose(f"Get Redis Cache: key: {key}")
            start_time: Final = time.time()
            cached_response: Final = self.redis_client.get(key)
            end_time: Final = time.time()
            _duration: Final = end_time - start_time
            self.service_logger_obj.service_success_hook(
                service=ServiceTypes.REDIS,
                duration=_duration,
                call_type=f"get_cache <- {_get_call_stack_info()}",
                start_time=start_time,
                end_time=end_time,
                parent_otel_span=parent_otel_span,
            )
            print_verbose(f"Got Redis Cache: key: {key}, cached_response {cached_response}")
            return self._get_cache_logic(cached_response=cached_response)
        except Exception as e:
            # NON blocking - notify users Redis is throwing an exception
            verbose_logger.error("litellm.caching.caching: get() - Got exception from REDIS: ", e)

    def _run_redis_mget_operation(self, keys: list[str]) -> Sequence[bytes | str | None]:
        """
        Wrapper to call `mget` on the redis client

        We use a wrapper so RedisCluster can override this method
        """
        return self.redis_client.mget(keys=keys)

    async def _async_run_redis_mget_operation(self, keys: list[str]) -> Sequence[bytes | str | None]:
        """
        Wrapper to call `mget` on the redis client

        We use a wrapper so RedisCluster can override this method
        """
        async_redis_client: Final = self.init_async_client()
        return await async_redis_client.mget(keys=keys)

    def batch_get_cache(
        self,
        key_list: list[str] | list[str | None],
        parent_otel_span: Span | None = None,
    ) -> dict:
        """
        Use Redis for bulk read operations

        Args:
            key_list: List of keys to get from Redis
            parent_otel_span: Optional parent OpenTelemetry span

        Returns:
            dict: A dictionary mapping keys to their cached values
        """
        key_value_dict = {}
        _key_list: Final = [key for key in key_list if key is not None]

        try:
            _keys: Final = []
            for cache_key in _key_list:
                cache_key = self.check_and_fix_namespace(key=cache_key or "")
                _keys.append(cache_key)
            start_time: Final = time.time()
            results: Final = self._run_redis_mget_operation(keys=_keys)
            end_time: Final = time.time()
            _duration: Final = end_time - start_time
            self.service_logger_obj.service_success_hook(
                service=ServiceTypes.REDIS,
                duration=_duration,
                call_type=f"batch_get_cache <- {_get_call_stack_info()}",
                start_time=start_time,
                end_time=end_time,
                parent_otel_span=parent_otel_span,
            )

            # Associate the results back with their keys.
            # 'results' is a list of values corresponding to the order of keys in '_key_list'.
            key_value_dict = dict(zip(_key_list, results))

            decoded_results: Final = {}
            for k, v in key_value_dict.items():
                if isinstance(k, bytes):
                    k = k.decode("utf-8")
                v = self._get_cache_logic(v)
                decoded_results[k] = v

            return decoded_results
        except Exception as e:
            verbose_logger.error("Error occurred in batch get cache - %s", e)
            return key_value_dict

    @_redis_circuit_breaker_guard
    async def async_get_cache(self, key, parent_otel_span: Span | None = None, **kwargs):
        from redis.asyncio import Redis

        _redis_client: Final[Redis] = self.init_async_client()
        key = self.check_and_fix_namespace(key=key)
        start_time: Final = time.time()

        try:
            print_verbose(f"Get Async Redis Cache: key: {key}")
            cached_response: Final = await _redis_client.get(key)
            print_verbose(f"Got Async Redis Cache: key: {key}, cached_response {cached_response}")
            response: Final = self._get_cache_logic(cached_response=cached_response)

            end_time = time.time()
            _duration = end_time - start_time
            asyncio.create_task(
                self.service_logger_obj.async_service_success_hook(
                    service=ServiceTypes.REDIS,
                    duration=_duration,
                    call_type=f"async_get_cache <- {_get_call_stack_info()}",
                    start_time=start_time,
                    end_time=end_time,
                    parent_otel_span=parent_otel_span,
                    event_metadata={"key": key},
                )
            )
            return response
        except Exception as e:
            end_time = time.time()
            _duration = end_time - start_time
            asyncio.create_task(
                self.service_logger_obj.async_service_failure_hook(
                    service=ServiceTypes.REDIS,
                    duration=_duration,
                    error=e,
                    call_type=f"async_get_cache <- {_get_call_stack_info()}",
                    start_time=start_time,
                    end_time=end_time,
                    parent_otel_span=parent_otel_span,
                    event_metadata={"key": key},
                )
            )
            print_verbose(f"litellm.caching.caching: async get() - Got exception from REDIS: {e}")
            _record_swallowed_redis_failure(self._circuit_breaker, e)

    @_redis_circuit_breaker_guard
    async def async_batch_get_cache(
        self,
        key_list: list[str] | list[str | None],
        parent_otel_span: Span | None = None,
    ) -> dict:
        """
        Use Redis for bulk read operations

        Args:
            key_list: List of keys to get from Redis
            parent_otel_span: Optional parent OpenTelemetry span

        Returns:
            dict: A dictionary mapping keys to their cached values

        `.mget` does not support None keys. This will filter out None keys.
        """
        # typed as Any, redis python lib has incomplete type stubs for RedisCluster and does not include `mget`
        key_value_dict = {}
        start_time: Final = time.time()
        _key_list: Final = [key for key in key_list if key is not None]
        try:
            _keys: Final = []
            for cache_key in _key_list:
                cache_key = self.check_and_fix_namespace(key=cache_key)
                _keys.append(cache_key)
            results: Final = await self._async_run_redis_mget_operation(keys=_keys)
            ## LOGGING ##
            end_time = time.time()
            _duration = end_time - start_time
            asyncio.create_task(
                self.service_logger_obj.async_service_success_hook(
                    service=ServiceTypes.REDIS,
                    duration=_duration,
                    call_type=f"async_batch_get_cache <- {_get_call_stack_info()}",
                    start_time=start_time,
                    end_time=end_time,
                    parent_otel_span=parent_otel_span,
                )
            )

            # Associate the results back with their keys.
            # 'results' is a list of values corresponding to the order of keys in 'key_list'.
            key_value_dict = dict(zip(_key_list, results))

            decoded_results: Final = {}
            for k, v in key_value_dict.items():
                if isinstance(k, bytes):
                    k = k.decode("utf-8")
                v = self._get_cache_logic(v)
                decoded_results[k] = v

            return decoded_results
        except Exception as e:
            ## LOGGING ##
            end_time = time.time()
            _duration = end_time - start_time
            asyncio.create_task(
                self.service_logger_obj.async_service_failure_hook(
                    service=ServiceTypes.REDIS,
                    duration=_duration,
                    error=e,
                    call_type=f"async_batch_get_cache <- {_get_call_stack_info()}",
                    start_time=start_time,
                    end_time=end_time,
                    parent_otel_span=parent_otel_span,
                )
            )
            verbose_logger.error("Error occurred in async batch get cache - %s", e)
            _record_swallowed_redis_failure(self._circuit_breaker, e)
            return key_value_dict

    def sync_ping(self) -> bool:
        """
        Tests if the sync redis client is correctly setup.
        """
        print_verbose("Pinging Sync Redis Cache")
        start_time: Final = time.time()
        try:
            response: Final[bool] = self.redis_client.ping()
            print_verbose(f"Redis Cache PING: {response}")
            ## LOGGING ##
            end_time = time.time()
            _duration = end_time - start_time
            self.service_logger_obj.service_success_hook(
                service=ServiceTypes.REDIS,
                duration=_duration,
                call_type=f"sync_ping <- {_get_call_stack_info()}",
                start_time=start_time,
                end_time=end_time,
            )
            return response
        except Exception as e:
            # NON blocking - notify users Redis is throwing an exception
            ## LOGGING ##
            end_time = time.time()
            _duration = end_time - start_time
            self.service_logger_obj.service_failure_hook(
                service=ServiceTypes.REDIS,
                duration=_duration,
                error=e,
                call_type=f"sync_ping <- {_get_call_stack_info()}",
            )
            verbose_logger.error("LiteLLM Redis Cache PING: - Got exception from REDIS : %s", e)
            raise e

    async def ping(self) -> bool:
        # typed as Any, redis python lib has incomplete type stubs for RedisCluster and does not include `ping`
        _redis_client: Final[Any] = self.init_async_client()
        start_time: Final = time.time()
        print_verbose("Pinging Async Redis Cache")
        try:
            response: Final = await _redis_client.ping()
            ## LOGGING ##
            end_time = time.time()
            _duration = end_time - start_time
            asyncio.create_task(
                self.service_logger_obj.async_service_success_hook(
                    service=ServiceTypes.REDIS,
                    duration=_duration,
                    call_type=f"async_ping <- {_get_call_stack_info()}",
                )
            )
            return response
        except Exception as e:
            # NON blocking - notify users Redis is throwing an exception
            ## LOGGING ##
            end_time = time.time()
            _duration = end_time - start_time
            asyncio.create_task(
                self.service_logger_obj.async_service_failure_hook(
                    service=ServiceTypes.REDIS,
                    duration=_duration,
                    error=e,
                    call_type=f"async_ping <- {_get_call_stack_info()}",
                )
            )
            verbose_logger.error("LiteLLM Redis Cache PING: - Got exception from REDIS : %s", e)
            raise e

    @_redis_circuit_breaker_guard
    async def delete_cache_keys(self, keys):
        # typed as Any, redis python lib has incomplete type stubs for RedisCluster and does not include `delete`
        _redis_client: Final[Any] = self.init_async_client()
        keys = [self.check_and_fix_namespace(key=key) for key in keys]
        # keys is a list, unpack it so it gets passed as individual elements to delete
        await _redis_client.delete(*keys)

    def client_list(self) -> list:
        client_list: Final[list] = self.redis_client.client_list()
        return client_list

    def info(self):
        info: Final = self.redis_client.info()
        return info

    def flush_cache(self):
        self.redis_client.flushall()

    def flushall(self):
        self.redis_client.flushall()

    async def disconnect(self):
        await self.async_redis_conn_pool.disconnect(inuse_connections=True)
        try:
            self.redis_client.close()
        except Exception as e:
            verbose_logger.debug("Error closing sync Redis client: %s", e)

    async def test_connection(self) -> dict:
        """
        Test the Redis connection by creating a new client and pinging it.

        This creates a fresh connection without using cached clients or connection pools
        to ensure the credentials are actually valid.

        Returns:
            dict: {"status": "success" | "failed", "message": str, "error": Optional[str]}
        """
        try:
            from .._redis import get_redis_async_client

            # Create a fresh Redis client with current settings
            redis_client: Final = get_redis_async_client(**self.redis_kwargs)

            # Test the connection
            ping_result: Final = await redis_client.ping()

            # Close the connection
            await redis_client.aclose()

            if ping_result:
                return {
                    "status": "success",
                    "message": "Redis connection test successful",
                }
            else:
                return {"status": "failed", "message": "Redis ping returned False"}
        except Exception as e:
            verbose_logger.error("Redis connection test failed: %s", e)
            return {
                "status": "failed",
                "message": f"Redis connection failed: {e}",
                "error": str(e),
            }

    @_redis_circuit_breaker_guard
    async def async_delete_cache(self, key: str):
        # typed as Any, redis python lib has incomplete type stubs for RedisCluster and does not include `delete`
        _redis_client: Final[Any] = self.init_async_client()
        key = self.check_and_fix_namespace(key=key)
        # keys is str
        return await _redis_client.delete(key)

    def delete_cache(self, key):
        key = self.check_and_fix_namespace(key=key)
        self.redis_client.delete(key)

    async def _pipeline_increment_helper(
        self,
        pipe: pipeline,
        increment_list: list[RedisPipelineIncrementOperation],
    ) -> list[float] | None:
        """Helper function for pipeline increment operations"""
        # Iterate through each increment operation and add commands to pipeline
        for increment_op in increment_list:
            cache_key = self.check_and_fix_namespace(key=increment_op["key"])
            print_verbose(
                f"Increment ASYNC Redis Cache PIPELINE: key: {cache_key}\nValue {increment_op['increment_value']}\nttl={increment_op['ttl']}"
            )
            pipe.incrbyfloat(cache_key, increment_op["increment_value"])
            if increment_op["ttl"] is not None:
                _td = timedelta(seconds=increment_op["ttl"])
                pipe.expire(cache_key, _td)
        # Execute the pipeline and return results
        results: Final = await pipe.execute()
        # only return float values
        verbose_logger.debug("Increment ASYNC Redis Cache PIPELINE: results: %s", results)
        return [r for r in results if isinstance(r, float)]

    @_redis_circuit_breaker_guard
    async def async_increment_pipeline(
        self, increment_list: list[RedisPipelineIncrementOperation], **kwargs
    ) -> list[float] | None:
        """
        Use Redis Pipelines for bulk increment operations
        Args:
            increment_list: List of RedisPipelineIncrementOperation dicts containing:
                - key: str
                - increment_value: float
                - ttl_seconds: int
        """
        # don't waste a network request if there's nothing to increment
        if len(increment_list) == 0:
            return None

        from redis.asyncio import Redis

        _redis_client: Final[Redis] = self.init_async_client()
        start_time: Final = time.time()

        print_verbose(f"Increment Async Redis Cache Pipeline: increment list: {increment_list}")

        try:
            async with _redis_client.pipeline(transaction=False) as pipe:
                results: Final = await self._pipeline_increment_helper(pipe, increment_list)

            ## LOGGING ##
            end_time = time.time()
            _duration = end_time - start_time
            asyncio.create_task(
                self.service_logger_obj.async_service_success_hook(
                    service=ServiceTypes.REDIS,
                    duration=_duration,
                    call_type=f"async_increment_pipeline <- {_get_call_stack_info()}",
                    start_time=start_time,
                    end_time=end_time,
                    parent_otel_span=_get_parent_otel_span_from_kwargs(kwargs),
                )
            )
            return results
        except Exception as e:
            ## LOGGING ##
            end_time = time.time()
            _duration = end_time - start_time
            asyncio.create_task(
                self.service_logger_obj.async_service_failure_hook(
                    service=ServiceTypes.REDIS,
                    duration=_duration,
                    error=e,
                    call_type=f"async_increment_pipeline <- {_get_call_stack_info()}",
                    start_time=start_time,
                    end_time=end_time,
                    parent_otel_span=_get_parent_otel_span_from_kwargs(kwargs),
                )
            )
            verbose_logger.error(
                "LiteLLM Redis Caching: async increment_pipeline() - Got exception from REDIS %s",
                str(e),
            )
            raise e

    @_redis_circuit_breaker_guard
    async def async_get_ttl(self, key: str) -> int | None:
        """
        Get the remaining TTL of a key in Redis

        Args:
            key (str): The key to get TTL for

        Returns:
            Optional[int]: The remaining TTL in seconds, or None if key doesn't exist

        Redis ref: https://redis.io/docs/latest/commands/ttl/
        """
        try:
            # typed as Any, redis python lib has incomplete type stubs for RedisCluster and does not include `ttl`
            _redis_client: Final[Any] = self.init_async_client()
            key = self.check_and_fix_namespace(key=key)
            ttl: Final = await _redis_client.ttl(key)
            if ttl <= -1:  # -1 means the key does not exist, -2 key does not exist
                return None
            return ttl
        except Exception as e:
            verbose_logger.debug("Redis TTL Error: %s", e)
            _record_swallowed_redis_failure(self._circuit_breaker, e)
            return None

    @_redis_circuit_breaker_guard
    async def async_rpush(
        self,
        key: str,
        values: Sequence[str | bytes | int | float],
        parent_otel_span: Span | None = None,
        **kwargs,
    ) -> int:
        """
        Append one or multiple values to a list stored at key

        Args:
            key: The Redis key of the list
            values: One or more values to append to the list
            parent_otel_span: Optional parent OpenTelemetry span

        Returns:
            int: The length of the list after the push operation
        """
        _redis_client: Final[Any] = self.init_async_client()
        key = self.check_and_fix_namespace(key=key)
        start_time: Final = time.time()
        try:
            response: Final = await _redis_client.rpush(key, *values)
            ## LOGGING ##
            end_time = time.time()
            _duration = end_time - start_time
            asyncio.create_task(
                self.service_logger_obj.async_service_success_hook(
                    service=ServiceTypes.REDIS,
                    duration=_duration,
                    call_type=f"async_rpush <- {_get_call_stack_info()}",
                )
            )
            return response
        except Exception as e:
            # NON blocking - notify users Redis is throwing an exception
            ## LOGGING ##
            end_time = time.time()
            _duration = end_time - start_time
            asyncio.create_task(
                self.service_logger_obj.async_service_failure_hook(
                    service=ServiceTypes.REDIS,
                    duration=_duration,
                    error=e,
                    call_type=f"async_rpush <- {_get_call_stack_info()}",
                )
            )
            verbose_logger.error("LiteLLM Redis Cache RPUSH: - Got exception from REDIS : %s", e)
            raise e

    async def _pipeline_rpush_helper(
        self,
        pipe: pipeline,
        rpush_list: Sequence[RedisPipelineRpushOperation],
    ) -> list[int]:
        """Helper function for pipeline rpush operations"""
        for rpush_op in rpush_list:
            key = self.check_and_fix_namespace(key=rpush_op["key"])
            pipe.rpush(key, *rpush_op["values"])
        results: Final = await pipe.execute()
        # Preserve positional correspondence — raise on per-command errors
        for r in results:
            if isinstance(r, Exception):
                raise r
        return results

    @_redis_circuit_breaker_guard
    async def async_rpush_pipeline(
        self,
        rpush_list: Sequence[RedisPipelineRpushOperation],
    ) -> list[int]:
        """
        Use Redis Pipelines for bulk RPUSH operations

        Args:
            rpush_list: List of RedisPipelineRpushOperation dicts containing:
                - key: str
                - values: List[Any]

        Returns:
            List[int]: List lengths after each push
        """
        if len(rpush_list) == 0:
            return []

        _redis_client: Final[Any] = self.init_async_client()
        start_time: Final = time.time()

        try:
            async with _redis_client.pipeline(transaction=False) as pipe:
                results: Final = await self._pipeline_rpush_helper(pipe, rpush_list)

            ## LOGGING ##
            end_time = time.time()
            _duration = end_time - start_time
            asyncio.create_task(
                self.service_logger_obj.async_service_success_hook(
                    service=ServiceTypes.REDIS,
                    duration=_duration,
                    call_type=f"async_rpush_pipeline <- {_get_call_stack_info()}",
                )
            )
            return results
        except Exception as e:
            ## LOGGING ##
            end_time = time.time()
            _duration = end_time - start_time
            asyncio.create_task(
                self.service_logger_obj.async_service_failure_hook(
                    service=ServiceTypes.REDIS,
                    duration=_duration,
                    error=e,
                    call_type=f"async_rpush_pipeline <- {_get_call_stack_info()}",
                )
            )
            verbose_logger.error(
                "LiteLLM Redis Caching: async_rpush_pipeline() - Got exception from REDIS %s",
                str(e),
            )
            raise e

    async def handle_lpop_count_for_older_redis_versions(self, pipe: pipeline, key: str, count: int) -> list[bytes]:
        result: Final[list[bytes]] = []
        for _ in range(count):
            pipe.lpop(key)
            results = await pipe.execute()

            # Filter out None values and decode bytes
            for r in results:
                if r is not None:
                    result.append(r)

        return result

    @_redis_circuit_breaker_guard
    async def async_lpop(
        self,
        key: str,
        count: int | None = None,
        parent_otel_span: Span | None = None,
        **kwargs,
    ) -> Any | list[Any]:
        _redis_client: Final[Any] = self.init_async_client()
        key = self.check_and_fix_namespace(key=key)
        start_time: Final = time.time()
        print_verbose(f"LPOP from Redis list: key: {key}, count: {count}")
        try:
            major_version: Final = self._parse_redis_major_version()

            if count is not None and major_version < 7:
                # For Redis < 7.0, use pipeline to execute multiple LPOP commands
                async with _redis_client.pipeline(transaction=False) as pipe:
                    result = await self.handle_lpop_count_for_older_redis_versions(pipe, key, count)
            else:
                # For Redis >= 7.0 or when count is None, use native LPOP with count
                result = await _redis_client.lpop(key, count)

            ## LOGGING ##
            end_time = time.time()
            _duration = end_time - start_time
            asyncio.create_task(
                self.service_logger_obj.async_service_success_hook(
                    service=ServiceTypes.REDIS,
                    duration=_duration,
                    call_type=f"async_lpop <- {_get_call_stack_info()}",
                )
            )

            # Handle result parsing if needed
            if isinstance(result, bytes):
                try:
                    return result.decode("utf-8")
                except Exception:
                    return result
            elif isinstance(result, list) and all(isinstance(item, bytes) for item in result):
                try:
                    return [item.decode("utf-8") for item in result]
                except Exception:
                    return result
            return result
        except Exception as e:
            # NON blocking - notify users Redis is throwing an exception
            ## LOGGING ##
            end_time = time.time()
            _duration = end_time - start_time
            asyncio.create_task(
                self.service_logger_obj.async_service_failure_hook(
                    service=ServiceTypes.REDIS,
                    duration=_duration,
                    error=e,
                    call_type=f"async_lpop <- {_get_call_stack_info()}",
                )
            )
            verbose_logger.error("LiteLLM Redis Cache LPOP: - Got exception from REDIS : %s", e)
            raise e

    async def _pipeline_lpop_helper(
        self,
        pipe: pipeline,
        lpop_list: list[RedisPipelineLpopOperation],
    ) -> list[list[str] | None]:
        """Helper function for pipeline lpop operations.

        For Redis >= 7, queues one LPOP(key, count) per operation.
        For Redis < 7, queues `count` individual LPOP(key) commands per operation.
        """
        major_version: Final = self._parse_redis_major_version()

        if major_version >= 7:
            for lpop_op in lpop_list:
                key = self.check_and_fix_namespace(key=lpop_op["key"])
                pipe.lpop(key, lpop_op["count"])
            raw_results = await pipe.execute()
        else:
            # For Redis < 7, LPOP doesn't support count param.
            # Issue `count` individual LPOP commands per key, all in one pipeline.
            counts: Final[list[int]] = []
            for lpop_op in lpop_list:
                key = self.check_and_fix_namespace(key=lpop_op["key"])
                count = lpop_op["count"] or 1
                counts.append(count)
                for _ in range(count):
                    pipe.lpop(key)
            flat_results: Final = await pipe.execute()

            # Re-group the flat results back into per-key lists
            raw_results = []
            offset = 0
            for count in counts:
                key_results = [r for r in flat_results[offset : offset + count] if r is not None]
                raw_results.append(key_results if key_results else None)
                offset += count

        # Raise on per-command errors (matches _pipeline_rpush_helper behavior)
        for r in raw_results:
            if isinstance(r, Exception):
                raise r

        # Decode bytes -> str for each result set
        decoded_results: Final[list[list[str] | None]] = []
        for r in raw_results:
            if r is None:
                decoded_results.append(None)
            elif isinstance(r, list):
                try:
                    decoded_results.append(
                        [item.decode("utf-8") if isinstance(item, bytes) else item for item in r if item is not None]
                        or None
                    )
                except Exception:
                    decoded_results.append(r)
            else:
                decoded_results.append(None)
        return decoded_results

    @_redis_circuit_breaker_guard
    async def async_lpop_pipeline(
        self,
        lpop_list: list[RedisPipelineLpopOperation],
    ) -> list[list[str] | None]:
        """
        Use Redis Pipelines for bulk LPOP operations

        Args:
            lpop_list: List of RedisPipelineLpopOperation dicts containing:
                - key: str
                - count: Optional[int]

        Returns:
            List[Optional[List[str]]]: Decoded results per key, None if key was empty
        """
        if len(lpop_list) == 0:
            return []

        _redis_client: Final[Any] = self.init_async_client()
        start_time: Final = time.time()

        try:
            async with _redis_client.pipeline(transaction=False) as pipe:
                results: Final = await self._pipeline_lpop_helper(pipe, lpop_list)

            ## LOGGING ##
            end_time = time.time()
            _duration = end_time - start_time
            asyncio.create_task(
                self.service_logger_obj.async_service_success_hook(
                    service=ServiceTypes.REDIS,
                    duration=_duration,
                    call_type=f"async_lpop_pipeline <- {_get_call_stack_info()}",
                )
            )
            return results
        except Exception as e:
            ## LOGGING ##
            end_time = time.time()
            _duration = end_time - start_time
            asyncio.create_task(
                self.service_logger_obj.async_service_failure_hook(
                    service=ServiceTypes.REDIS,
                    duration=_duration,
                    error=e,
                    call_type=f"async_lpop_pipeline <- {_get_call_stack_info()}",
                )
            )
            verbose_logger.error(
                "LiteLLM Redis Caching: async_lpop_pipeline() - Got exception from REDIS %s",
                str(e),
            )
            raise e
