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
import inspect
import json
import time
from datetime import timedelta
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union, cast

import litellm
from litellm._logging import print_verbose, verbose_logger
from litellm.litellm_core_utils.core_helpers import _get_parent_otel_span_from_kwargs
from litellm.types.caching import RedisPipelineIncrementOperation
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
    Span = Union[_Span, Any]
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
        current_frame = inspect.currentframe()
        if current_frame is None:
            return "unknown"

        # Skip this function and the immediate caller (which sets call_type)
        f_back = current_frame.f_back
        if f_back is None:
            return "unknown"
        frame = f_back.f_back
        if frame is None:
            return "unknown"
        function_names = []

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


class RedisCache(BaseCache):
    # if users don't provider one, use the default litellm cache

    def __init__(
        self,
        host=None,
        port=None,
        password=None,
        redis_flush_size: Optional[int] = 100,
        namespace: Optional[str] = None,
        startup_nodes: Optional[List] = None,  # for redis-cluster
        socket_timeout: Optional[float] = 5.0,  # default 5 second timeout
        **kwargs,
    ):
        from litellm._service_logger import ServiceLogging

        from .._redis import get_redis_client, get_redis_connection_pool

        redis_kwargs = {}
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
        self.redis_async_client: Optional[
            Union[async_redis_client, async_redis_cluster_client]
        ] = None
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
            if not inspect.iscoroutinefunction(self.redis_client):
                self.redis_version = self.redis_client.info()["redis_version"]  # type: ignore
        except Exception:
            pass

        ### ASYNC HEALTH PING ###
        try:
            # asyncio.get_running_loop().create_task(self.ping())
            _ = asyncio.get_running_loop().create_task(self.ping())
        except Exception as e:
            if "no running event loop" in str(e):
                verbose_logger.debug(
                    "Ignoring async redis ping. No running event loop."
                )
            else:
                verbose_logger.error(
                    "Error connecting to Async Redis client - {}".format(str(e)),
                    extra={"error": str(e)},
                )

        ### SYNC HEALTH PING ###
        try:
            if hasattr(self.redis_client, "ping"):
                self.redis_client.ping()  # type: ignore
        except Exception as e:
            verbose_logger.error(
                "Error connecting to Sync Redis client", extra={"error": str(e)}
            )

        if litellm.default_redis_ttl is not None:
            super().__init__(default_ttl=int(litellm.default_redis_ttl))
        else:
            super().__init__()  # defaults to 60s

    def init_async_client(
        self,
    ) -> Union[async_redis_client, async_redis_cluster_client]:
        from litellm import in_memory_llm_clients_cache

        from .._redis import get_redis_async_client, get_redis_connection_pool

        cached_client = in_memory_llm_clients_cache.get_cache(key="async-redis-client")
        if cached_client is not None:
            redis_async_client = cast(
                Union[async_redis_client, async_redis_cluster_client], cached_client
            )
        else:
            # Create new connection pool and client for current event loop
            self.async_redis_conn_pool = get_redis_connection_pool(**self.redis_kwargs)
            redis_async_client = get_redis_async_client(
                connection_pool=self.async_redis_conn_pool, **self.redis_kwargs
            )
            in_memory_llm_clients_cache.set_cache(
                key="async-redis-client", value=self.redis_async_client
            )

        self.redis_async_client = redis_async_client  # type: ignore
        return redis_async_client

    def check_and_fix_namespace(self, key: str) -> str:
        """
        Make sure each key starts with the given namespace
        """
        if self.namespace is not None and not key.startswith(self.namespace):
            key = self.namespace + ":" + key

        return key

    def set_cache(self, key, value, **kwargs):
        ttl = self.get_ttl(**kwargs)
        print_verbose(
            f"Set Redis Cache: key: {key}\nValue {value}\nttl={ttl}, redis_version={self.redis_version}"
        )
        key = self.check_and_fix_namespace(key=key)
        try:
            start_time = time.time()
            self.redis_client.set(name=key, value=str(value), ex=ttl)
            end_time = time.time()
            _duration = end_time - start_time
            self.service_logger_obj.service_success_hook(
                service=ServiceTypes.REDIS,
                duration=_duration,
                call_type=f"set_cache <- {_get_call_stack_info()}",
                start_time=start_time,
                end_time=end_time,
            )
        except Exception as e:
            # NON blocking - notify users Redis is throwing an exception
            print_verbose(
                f"litellm.caching.caching: set() - Got exception from REDIS : {str(e)}"
            )

    def increment_cache(
        self, key, value: int, ttl: Optional[float] = None, **kwargs
    ) -> int:
        _redis_client = self.redis_client
        start_time = time.time()
        set_ttl = self.get_ttl(ttl=ttl)
        try:
            start_time = time.time()
            result: int = _redis_client.incr(name=key, amount=value)  # type: ignore
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
                current_ttl = _redis_client.ttl(key)
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
                    _redis_client.expire(key, set_ttl)  # type: ignore
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

    async def async_scan_iter(self, pattern: str, count: int = 100) -> list:
        start_time = time.time()
        try:
            keys = []
            _redis_client = self.init_async_client()
            if not hasattr(_redis_client, "scan_iter"):
                verbose_logger.debug(
                    "Redis client does not support scan_iter, potentially using Redis Cluster. Returning empty list."
                )
                return []

            async for key in _redis_client.scan_iter(match=pattern + "*", count=count):  # type: ignore
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

    def async_register_script(self, script: str) -> Any:
        """
        Register a Lua script with Redis asynchronously.
        Works with both standalone Redis and Redis Cluster.

        Args:
            script (str): The Lua script to register

        Returns:
            Any: A script object that can be called with keys and args
        """
        try:
            _redis_client = self.init_async_client()
            # For standalone Redis
            if hasattr(_redis_client, "register_script"):
                return _redis_client.register_script(script)  # type: ignore
            # For Redis Cluster
            elif hasattr(_redis_client, "script_load"):
                # Load the script and get its SHA
                script_sha = _redis_client.script_load(script)  # type: ignore

                # Return a callable that uses evalsha
                async def script_callable(keys: List[str], args: List[Any]) -> Any:
                    return _redis_client.evalsha(script_sha, len(keys), *keys, *args)  # type: ignore

                return script_callable
        except Exception as e:
            verbose_logger.error(f"Error registering Redis script: {str(e)}")
            raise e

    async def async_set_cache(self, key, value, **kwargs):
        from redis.asyncio import Redis

        start_time = time.time()
        try:
            _redis_client: Redis = self.init_async_client()  # type: ignore
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
                "LiteLLM Redis Caching: async set() - Got exception from REDIS %s, Writing value=%s",
                str(e),
                value,
            )
            raise e

        key = self.check_and_fix_namespace(key=key)
        ttl = self.get_ttl(**kwargs)
        nx = kwargs.get("nx", False)
        print_verbose(f"Set ASYNC Redis Cache: key: {key}\nValue {value}\nttl={ttl}")

        try:
            if not hasattr(_redis_client, "set"):
                raise Exception("Redis client cannot set cache. Attribute not found.")
            result = await _redis_client.set(
                name=key,
                value=json.dumps(value),
                nx=nx,
                ex=ttl,
            )
            print_verbose(
                f"Successfully Set ASYNC Redis Cache: key: {key}\nValue {value}\nttl={ttl}"
            )
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

    async def _pipeline_helper(
        self,
        pipe: Union[pipeline, cluster_pipeline],
        cache_list: List[Tuple[Any, Any]],
        ttl: Optional[float],
    ) -> List:
        """
        Helper function for executing a pipeline of set operations on Redis
        """
        ttl = self.get_ttl(ttl=ttl)
        # Iterate through each key-value pair in the cache_list and set them in the pipeline.
        for cache_key, cache_value in cache_list:
            cache_key = self.check_and_fix_namespace(key=cache_key)
            print_verbose(
                f"Set ASYNC Redis Cache PIPELINE: key: {cache_key}\nValue {cache_value}\nttl={ttl}"
            )
            json_cache_value = json.dumps(cache_value)
            # Set the value with a TTL if it's provided.
            _td: Optional[timedelta] = None
            if ttl is not None:
                _td = timedelta(seconds=ttl)
            pipe.set(  # type: ignore
                name=cache_key,
                value=json_cache_value,
                ex=_td,
            )
        # Execute the pipeline and return the results.
        results = await pipe.execute()
        return results

    async def async_set_cache_pipeline(
        self, cache_list: List[Tuple[Any, Any]], ttl: Optional[float] = None, **kwargs
    ):
        """
        Use Redis Pipelines for bulk write operations
        """
        # don't waste a network request if there's nothing to set
        if len(cache_list) == 0:
            return

        _redis_client = self.init_async_client()
        start_time = time.time()

        print_verbose(
            f"Set Async Redis Cache: key list: {cache_list}\nttl={ttl}, redis_version={self.redis_version}"
        )
        cache_value: Any = None
        try:
            async with _redis_client.pipeline(transaction=False) as pipe:
                results = await self._pipeline_helper(pipe, cache_list, ttl)

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
            return None
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

    async def _set_cache_sadd_helper(
        self,
        redis_client: async_redis_client,
        key: str,
        value: List,
        ttl: Optional[float],
    ) -> None:
        """Helper function for async_set_cache_sadd. Separated for testing."""
        ttl = self.get_ttl(ttl=ttl)
        try:
            await redis_client.sadd(key, *value)  # type: ignore
            if ttl is not None:
                _td = timedelta(seconds=ttl)
                await redis_client.expire(key, _td)
        except Exception:
            raise

    async def async_set_cache_sadd(
        self, key, value: List, ttl: Optional[float], **kwargs
    ):
        from redis.asyncio import Redis

        start_time = time.time()
        try:
            _redis_client: Redis = self.init_async_client()  # type: ignore
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
            await self._set_cache_sadd_helper(
                redis_client=_redis_client, key=key, value=value, ttl=ttl
            )
            print_verbose(
                f"Successfully Set ASYNC Redis Cache SADD: key: {key}\nValue {value}\nttl={ttl}"
            )
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

    async def batch_cache_write(self, key, value, **kwargs):
        print_verbose(
            f"in batch cache writing for redis buffer size={len(self.redis_batch_writing_buffer)}",
        )
        key = self.check_and_fix_namespace(key=key)
        self.redis_batch_writing_buffer.append((key, value))
        if len(self.redis_batch_writing_buffer) >= self.redis_flush_size:
            await self.flush_cache_buffer()  # logging done in here

    async def async_increment(
        self,
        key,
        value: float,
        ttl: Optional[int] = None,
        parent_otel_span: Optional[Span] = None,
    ) -> float:
        from redis.asyncio import Redis

        _redis_client: Redis = self.init_async_client()  # type: ignore
        start_time = time.time()
        _used_ttl = self.get_ttl(ttl=ttl)
        key = self.check_and_fix_namespace(key=key)
        try:
            result = await _redis_client.incrbyfloat(name=key, amount=value)
            if _used_ttl is not None:
                # check if key already has ttl, if not -> set ttl
                current_ttl = await _redis_client.ttl(key)
                if current_ttl == -1:
                    # Key has no expiration
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

    async def flush_cache_buffer(self):
        print_verbose(
            f"flushing to redis....reached size of buffer {len(self.redis_batch_writing_buffer)}"
        )
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
            cached_response = json.loads(
                cached_response
            )  # Convert string to dictionary
        except Exception:
            cached_response = ast.literal_eval(cached_response)
        return cached_response

    def get_cache(self, key, parent_otel_span: Optional[Span] = None, **kwargs):
        try:
            key = self.check_and_fix_namespace(key=key)
            print_verbose(f"Get Redis Cache: key: {key}")
            start_time = time.time()
            cached_response = self.redis_client.get(key)
            end_time = time.time()
            _duration = end_time - start_time
            self.service_logger_obj.service_success_hook(
                service=ServiceTypes.REDIS,
                duration=_duration,
                call_type=f"get_cache <- {_get_call_stack_info()}",
                start_time=start_time,
                end_time=end_time,
                parent_otel_span=parent_otel_span,
            )
            print_verbose(
                f"Got Redis Cache: key: {key}, cached_response {cached_response}"
            )
            return self._get_cache_logic(cached_response=cached_response)
        except Exception as e:
            # NON blocking - notify users Redis is throwing an exception
            verbose_logger.error(
                "litellm.caching.caching: get() - Got exception from REDIS: ", e
            )

    def _run_redis_mget_operation(self, keys: List[str]) -> List[Any]:
        """
        Wrapper to call `mget` on the redis client

        We use a wrapper so RedisCluster can override this method
        """
        return self.redis_client.mget(keys=keys)  # type: ignore

    async def _async_run_redis_mget_operation(self, keys: List[str]) -> List[Any]:
        """
        Wrapper to call `mget` on the redis client

        We use a wrapper so RedisCluster can override this method
        """
        async_redis_client = self.init_async_client()
        return await async_redis_client.mget(keys=keys)  # type: ignore

    def batch_get_cache(
        self,
        key_list: Union[List[str], List[Optional[str]]],
        parent_otel_span: Optional[Span] = None,
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
        _key_list = [key for key in key_list if key is not None]

        try:
            _keys = []
            for cache_key in _key_list:
                cache_key = self.check_and_fix_namespace(key=cache_key or "")
                _keys.append(cache_key)
            start_time = time.time()
            results: List = self._run_redis_mget_operation(keys=_keys)
            end_time = time.time()
            _duration = end_time - start_time
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

            decoded_results = {}
            for k, v in key_value_dict.items():
                if isinstance(k, bytes):
                    k = k.decode("utf-8")
                v = self._get_cache_logic(v)
                decoded_results[k] = v

            return decoded_results
        except Exception as e:
            verbose_logger.error(f"Error occurred in batch get cache - {str(e)}")
            return key_value_dict

    async def async_get_cache(
        self, key, parent_otel_span: Optional[Span] = None, **kwargs
    ):
        from redis.asyncio import Redis

        _redis_client: Redis = self.init_async_client()  # type: ignore
        key = self.check_and_fix_namespace(key=key)
        start_time = time.time()

        try:
            print_verbose(f"Get Async Redis Cache: key: {key}")
            cached_response = await _redis_client.get(key)
            print_verbose(
                f"Got Async Redis Cache: key: {key}, cached_response {cached_response}"
            )
            response = self._get_cache_logic(cached_response=cached_response)

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
            print_verbose(
                f"litellm.caching.caching: async get() - Got exception from REDIS: {str(e)}"
            )

    async def async_batch_get_cache(
        self,
        key_list: Union[List[str], List[Optional[str]]],
        parent_otel_span: Optional[Span] = None,
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
        start_time = time.time()
        _key_list = [key for key in key_list if key is not None]
        try:
            _keys = []
            for cache_key in _key_list:
                cache_key = self.check_and_fix_namespace(key=cache_key)
                _keys.append(cache_key)
            results = await self._async_run_redis_mget_operation(keys=_keys)
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

            decoded_results = {}
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
            verbose_logger.error(f"Error occurred in async batch get cache - {str(e)}")
            return key_value_dict

    def sync_ping(self) -> bool:
        """
        Tests if the sync redis client is correctly setup.
        """
        print_verbose("Pinging Sync Redis Cache")
        start_time = time.time()
        try:
            response: bool = self.redis_client.ping()  # type: ignore
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
            verbose_logger.error(
                f"LiteLLM Redis Cache PING: - Got exception from REDIS : {str(e)}"
            )
            raise e

    async def ping(self) -> bool:
        # typed as Any, redis python lib has incomplete type stubs for RedisCluster and does not include `ping`
        _redis_client: Any = self.init_async_client()
        start_time = time.time()
        print_verbose("Pinging Async Redis Cache")
        try:
            response = await _redis_client.ping()
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
            verbose_logger.error(
                f"LiteLLM Redis Cache PING: - Got exception from REDIS : {str(e)}"
            )
            raise e

    async def delete_cache_keys(self, keys):
        # typed as Any, redis python lib has incomplete type stubs for RedisCluster and does not include `delete`
        _redis_client: Any = self.init_async_client()
        # keys is a list, unpack it so it gets passed as individual elements to delete
        await _redis_client.delete(*keys)

    def client_list(self) -> List:
        client_list: List = self.redis_client.client_list()  # type: ignore
        return client_list

    def info(self):
        info = self.redis_client.info()
        return info

    def flush_cache(self):
        self.redis_client.flushall()

    def flushall(self):
        self.redis_client.flushall()

    async def disconnect(self):
        await self.async_redis_conn_pool.disconnect(inuse_connections=True)

    async def async_delete_cache(self, key: str):
        # typed as Any, redis python lib has incomplete type stubs for RedisCluster and does not include `delete`
        _redis_client: Any = self.init_async_client()
        # keys is str
        return await _redis_client.delete(key)

    def delete_cache(self, key):
        self.redis_client.delete(key)

    async def _pipeline_increment_helper(
        self,
        pipe: pipeline,
        increment_list: List[RedisPipelineIncrementOperation],
    ) -> Optional[List[float]]:
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
        results = await pipe.execute()
        # only return float values
        verbose_logger.debug(
            f"Increment ASYNC Redis Cache PIPELINE: results: {results}"
        )
        return [r for r in results if isinstance(r, float)]

    async def async_increment_pipeline(
        self, increment_list: List[RedisPipelineIncrementOperation], **kwargs
    ) -> Optional[List[float]]:
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

        _redis_client: Redis = self.init_async_client()  # type: ignore
        start_time = time.time()

        print_verbose(
            f"Increment Async Redis Cache Pipeline: increment list: {increment_list}"
        )

        try:
            async with _redis_client.pipeline(transaction=False) as pipe:
                results = await self._pipeline_increment_helper(pipe, increment_list)

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

    async def async_get_ttl(self, key: str) -> Optional[int]:
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
            _redis_client: Any = self.init_async_client()
            ttl = await _redis_client.ttl(key)
            if ttl <= -1:  # -1 means the key does not exist, -2 key does not exist
                return None
            return ttl
        except Exception as e:
            verbose_logger.debug(f"Redis TTL Error: {e}")
            return None

    async def async_rpush(
        self,
        key: str,
        values: List[Any],
        parent_otel_span: Optional[Span] = None,
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
        _redis_client: Any = self.init_async_client()
        start_time = time.time()
        try:
            response = await _redis_client.rpush(key, *values)
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
            verbose_logger.error(
                f"LiteLLM Redis Cache RPUSH: - Got exception from REDIS : {str(e)}"
            )
            raise e

    async def handle_lpop_count_for_older_redis_versions(
        self, pipe: pipeline, key: str, count: int
    ) -> List[bytes]:
        result: List[bytes] = []
        for _ in range(count):
            pipe.lpop(key)
            results = await pipe.execute()

            # Filter out None values and decode bytes
            for r in results:
                if r is not None:
                    result.append(r)

        return result

    async def async_lpop(
        self,
        key: str,
        count: Optional[int] = None,
        parent_otel_span: Optional[Span] = None,
        **kwargs,
    ) -> Union[Any, List[Any]]:
        _redis_client: Any = self.init_async_client()
        start_time = time.time()
        print_verbose(f"LPOP from Redis list: key: {key}, count: {count}")
        try:
            major_version: int = 7
            # Check Redis version and use appropriate method
            if self.redis_version != "Unknown":
                # Parse version string like "6.0.0" to get major version
                major_version = int(self.redis_version.split(".")[0])

            if count is not None and major_version < 7:
                # For Redis < 7.0, use pipeline to execute multiple LPOP commands
                async with _redis_client.pipeline(transaction=False) as pipe:
                    result = await self.handle_lpop_count_for_older_redis_versions(
                        pipe, key, count
                    )
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
            elif isinstance(result, list) and all(
                isinstance(item, bytes) for item in result
            ):
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
            verbose_logger.error(
                f"LiteLLM Redis Cache LPOP: - Got exception from REDIS : {str(e)}"
            )
            raise e
