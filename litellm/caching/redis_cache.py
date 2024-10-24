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
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import litellm
from litellm._logging import print_verbose, verbose_logger
from litellm.litellm_core_utils.core_helpers import _get_parent_otel_span_from_kwargs
from litellm.types.services import ServiceLoggerPayload, ServiceTypes
from litellm.types.utils import all_litellm_params

from .base_cache import BaseCache

if TYPE_CHECKING:
    from redis.asyncio import Redis
    from redis.asyncio.client import Pipeline

    pipeline = Pipeline
    async_redis_client = Redis
else:
    pipeline = Any
    async_redis_client = Any


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
        **kwargs,
    ):
        import redis

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
        ### HEALTH MONITORING OBJECT ###
        if kwargs.get("service_logger_obj", None) is not None and isinstance(
            kwargs["service_logger_obj"], ServiceLogging
        ):
            self.service_logger_obj = kwargs.pop("service_logger_obj")
        else:
            self.service_logger_obj = ServiceLogging()

        redis_kwargs.update(kwargs)
        self.redis_client = get_redis_client(**redis_kwargs)
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

    def init_async_client(self):
        from .._redis import get_redis_async_client

        return get_redis_async_client(
            connection_pool=self.async_redis_conn_pool, **self.redis_kwargs
        )

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
            self.redis_client.set(name=key, value=str(value), ex=ttl)
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
            result: int = _redis_client.incr(name=key, amount=value)  # type: ignore

            if set_ttl is not None:
                # check if key already has ttl, if not -> set ttl
                current_ttl = _redis_client.ttl(key)
                if current_ttl == -1:
                    # Key has no expiration
                    _redis_client.expire(key, set_ttl)  # type: ignore
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
        from redis.asyncio import Redis

        start_time = time.time()
        try:
            keys = []
            _redis_client: Redis = self.init_async_client()  # type: ignore

            async with _redis_client as redis_client:
                async for key in redis_client.scan_iter(
                    match=pattern + "*", count=count
                ):
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
                        call_type="async_scan_iter",
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
                    call_type="async_scan_iter",
                    start_time=start_time,
                    end_time=end_time,
                )
            )
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
                    call_type="async_set_cache",
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
        async with _redis_client as redis_client:
            ttl = self.get_ttl(**kwargs)
            print_verbose(
                f"Set ASYNC Redis Cache: key: {key}\nValue {value}\nttl={ttl}"
            )
            try:
                if not hasattr(redis_client, "set"):
                    raise Exception(
                        "Redis client cannot set cache. Attribute not found."
                    )
                await redis_client.set(name=key, value=json.dumps(value), ex=ttl)
                print_verbose(
                    f"Successfully Set ASYNC Redis Cache: key: {key}\nValue {value}\nttl={ttl}"
                )
                end_time = time.time()
                _duration = end_time - start_time
                asyncio.create_task(
                    self.service_logger_obj.async_service_success_hook(
                        service=ServiceTypes.REDIS,
                        duration=_duration,
                        call_type="async_set_cache",
                        start_time=start_time,
                        end_time=end_time,
                        parent_otel_span=_get_parent_otel_span_from_kwargs(kwargs),
                        event_metadata={"key": key},
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
                        call_type="async_set_cache",
                        start_time=start_time,
                        end_time=end_time,
                        parent_otel_span=_get_parent_otel_span_from_kwargs(kwargs),
                        event_metadata={"key": key},
                    )
                )
                # NON blocking - notify users Redis is throwing an exception
                verbose_logger.error(
                    "LiteLLM Redis Caching: async set() - Got exception from REDIS %s, Writing value=%s",
                    str(e),
                    value,
                )

    async def _pipeline_helper(
        self, pipe: pipeline, cache_list: List[Tuple[Any, Any]], ttl: Optional[float]
    ) -> List:
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
            pipe.set(cache_key, json_cache_value, ex=_td)
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
        from redis.asyncio import Redis

        _redis_client: Redis = self.init_async_client()  # type: ignore
        start_time = time.time()

        print_verbose(
            f"Set Async Redis Cache: key list: {cache_list}\nttl={ttl}, redis_version={self.redis_version}"
        )
        cache_value: Any = None
        try:
            async with _redis_client as redis_client:
                async with redis_client.pipeline(transaction=True) as pipe:
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
                    call_type="async_set_cache_pipeline",
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
                    call_type="async_set_cache_pipeline",
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
                    call_type="async_set_cache_sadd",
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
        async with _redis_client as redis_client:
            print_verbose(
                f"Set ASYNC Redis Cache: key: {key}\nValue {value}\nttl={ttl}"
            )
            try:
                await self._set_cache_sadd_helper(
                    redis_client=redis_client, key=key, value=value, ttl=ttl
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
                        call_type="async_set_cache_sadd",
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
                        call_type="async_set_cache_sadd",
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
        self, key, value: float, ttl: Optional[int] = None, **kwargs
    ) -> float:
        from redis.asyncio import Redis

        _redis_client: Redis = self.init_async_client()  # type: ignore
        start_time = time.time()
        _used_ttl = self.get_ttl(ttl=ttl)
        try:
            async with _redis_client as redis_client:
                result = await redis_client.incrbyfloat(name=key, amount=value)

                if _used_ttl is not None:
                    # check if key already has ttl, if not -> set ttl
                    current_ttl = await redis_client.ttl(key)
                    if current_ttl == -1:
                        # Key has no expiration
                        await redis_client.expire(key, _used_ttl)

                ## LOGGING ##
                end_time = time.time()
                _duration = end_time - start_time
                asyncio.create_task(
                    self.service_logger_obj.async_service_success_hook(
                        service=ServiceTypes.REDIS,
                        duration=_duration,
                        call_type="async_increment",
                        start_time=start_time,
                        end_time=end_time,
                        parent_otel_span=_get_parent_otel_span_from_kwargs(kwargs),
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
                    call_type="async_increment",
                    start_time=start_time,
                    end_time=end_time,
                    parent_otel_span=_get_parent_otel_span_from_kwargs(kwargs),
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

    def get_cache(self, key, **kwargs):
        try:
            key = self.check_and_fix_namespace(key=key)
            print_verbose(f"Get Redis Cache: key: {key}")
            cached_response = self.redis_client.get(key)
            print_verbose(
                f"Got Redis Cache: key: {key}, cached_response {cached_response}"
            )
            return self._get_cache_logic(cached_response=cached_response)
        except Exception as e:
            # NON blocking - notify users Redis is throwing an exception
            verbose_logger.error(
                "litellm.caching.caching: get() - Got exception from REDIS: ", e
            )

    def batch_get_cache(self, key_list) -> dict:
        """
        Use Redis for bulk read operations
        """
        key_value_dict = {}
        try:
            _keys = []
            for cache_key in key_list:
                cache_key = self.check_and_fix_namespace(key=cache_key)
                _keys.append(cache_key)
            results: List = self.redis_client.mget(keys=_keys)  # type: ignore

            # Associate the results back with their keys.
            # 'results' is a list of values corresponding to the order of keys in 'key_list'.
            key_value_dict = dict(zip(key_list, results))

            decoded_results = {
                k.decode("utf-8"): self._get_cache_logic(v)
                for k, v in key_value_dict.items()
            }

            return decoded_results
        except Exception as e:
            print_verbose(f"Error occurred in pipeline read - {str(e)}")
            return key_value_dict

    async def async_get_cache(self, key, **kwargs):
        from redis.asyncio import Redis

        _redis_client: Redis = self.init_async_client()  # type: ignore
        key = self.check_and_fix_namespace(key=key)
        start_time = time.time()
        async with _redis_client as redis_client:
            try:
                print_verbose(f"Get Async Redis Cache: key: {key}")
                cached_response = await redis_client.get(key)
                print_verbose(
                    f"Got Async Redis Cache: key: {key}, cached_response {cached_response}"
                )
                response = self._get_cache_logic(cached_response=cached_response)
                ## LOGGING ##
                end_time = time.time()
                _duration = end_time - start_time
                asyncio.create_task(
                    self.service_logger_obj.async_service_success_hook(
                        service=ServiceTypes.REDIS,
                        duration=_duration,
                        call_type="async_get_cache",
                        start_time=start_time,
                        end_time=end_time,
                        parent_otel_span=_get_parent_otel_span_from_kwargs(kwargs),
                        event_metadata={"key": key},
                    )
                )
                return response
            except Exception as e:
                ## LOGGING ##
                end_time = time.time()
                _duration = end_time - start_time
                asyncio.create_task(
                    self.service_logger_obj.async_service_failure_hook(
                        service=ServiceTypes.REDIS,
                        duration=_duration,
                        error=e,
                        call_type="async_get_cache",
                        start_time=start_time,
                        end_time=end_time,
                        parent_otel_span=_get_parent_otel_span_from_kwargs(kwargs),
                        event_metadata={"key": key},
                    )
                )
                # NON blocking - notify users Redis is throwing an exception
                print_verbose(
                    f"litellm.caching.caching: async get() - Got exception from REDIS: {str(e)}"
                )

    async def async_batch_get_cache(self, key_list) -> dict:
        """
        Use Redis for bulk read operations
        """
        _redis_client = await self.init_async_client()
        key_value_dict = {}
        start_time = time.time()
        try:
            async with _redis_client as redis_client:
                _keys = []
                for cache_key in key_list:
                    cache_key = self.check_and_fix_namespace(key=cache_key)
                    _keys.append(cache_key)
                results = await redis_client.mget(keys=_keys)

            ## LOGGING ##
            end_time = time.time()
            _duration = end_time - start_time
            asyncio.create_task(
                self.service_logger_obj.async_service_success_hook(
                    service=ServiceTypes.REDIS,
                    duration=_duration,
                    call_type="async_batch_get_cache",
                    start_time=start_time,
                    end_time=end_time,
                )
            )

            # Associate the results back with their keys.
            # 'results' is a list of values corresponding to the order of keys in 'key_list'.
            key_value_dict = dict(zip(key_list, results))

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
                    call_type="async_batch_get_cache",
                    start_time=start_time,
                    end_time=end_time,
                )
            )
            print_verbose(f"Error occurred in pipeline read - {str(e)}")
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
                call_type="sync_ping",
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
                call_type="sync_ping",
            )
            verbose_logger.error(
                f"LiteLLM Redis Cache PING: - Got exception from REDIS : {str(e)}"
            )
            raise e

    async def ping(self) -> bool:
        _redis_client = self.init_async_client()
        start_time = time.time()
        async with _redis_client as redis_client:
            print_verbose("Pinging Async Redis Cache")
            try:
                response = await redis_client.ping()
                ## LOGGING ##
                end_time = time.time()
                _duration = end_time - start_time
                asyncio.create_task(
                    self.service_logger_obj.async_service_success_hook(
                        service=ServiceTypes.REDIS,
                        duration=_duration,
                        call_type="async_ping",
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
                        call_type="async_ping",
                    )
                )
                verbose_logger.error(
                    f"LiteLLM Redis Cache PING: - Got exception from REDIS : {str(e)}"
                )
                raise e

    async def delete_cache_keys(self, keys):
        _redis_client = self.init_async_client()
        # keys is a list, unpack it so it gets passed as individual elements to delete
        async with _redis_client as redis_client:
            await redis_client.delete(*keys)

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
        _redis_client = self.init_async_client()
        # keys is str
        async with _redis_client as redis_client:
            await redis_client.delete(key)

    def delete_cache(self, key):
        self.redis_client.delete(key)
