# +-----------------------------------------------+
# |                                               |
# |           Give Feedback / Get Help            |
# | https://github.com/BerriAI/litellm/issues/new |
# |                                               |
# +-----------------------------------------------+
#
#  Thank you users! We ❤️ you! - Krrish & Ishaan

import ast
import asyncio
import hashlib
import io
import json
import logging
import time
import traceback
from datetime import timedelta
from enum import Enum
from typing import Any, List, Literal, Optional, Union

from openai._models import BaseModel as OpenAIObject

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.core_helpers import _get_parent_otel_span_from_kwargs
from litellm.types.services import ServiceLoggerPayload, ServiceTypes
from litellm.types.utils import all_litellm_params


def print_verbose(print_statement):
    try:
        verbose_logger.debug(print_statement)
        if litellm.set_verbose:
            print(print_statement)  # noqa
    except Exception:
        pass


class CacheMode(str, Enum):
    default_on = "default_on"
    default_off = "default_off"


class BaseCache:
    def set_cache(self, key, value, **kwargs):
        raise NotImplementedError

    async def async_set_cache(self, key, value, **kwargs):
        raise NotImplementedError

    def get_cache(self, key, **kwargs):
        raise NotImplementedError

    async def async_get_cache(self, key, **kwargs):
        raise NotImplementedError

    async def batch_cache_write(self, result, *args, **kwargs):
        raise NotImplementedError

    async def disconnect(self):
        raise NotImplementedError


class InMemoryCache(BaseCache):
    def __init__(
        self,
        max_size_in_memory: Optional[int] = 200,
        default_ttl: Optional[
            int
        ] = 600,  # default ttl is 10 minutes. At maximum litellm rate limiting logic requires objects to be in memory for 1 minute
    ):
        """
        max_size_in_memory [int]: Maximum number of items in cache. done to prevent memory leaks. Use 200 items as a default
        """
        self.max_size_in_memory = (
            max_size_in_memory or 200
        )  # set an upper bound of 200 items in-memory
        self.default_ttl = default_ttl or 600

        # in-memory cache
        self.cache_dict: dict = {}
        self.ttl_dict: dict = {}

    def evict_cache(self):
        """
        Eviction policy:
        - check if any items in ttl_dict are expired -> remove them from ttl_dict and cache_dict


        This guarantees the following:
        - 1. When item ttl not set: At minimumm each item will remain in memory for 5 minutes
        - 2. When ttl is set: the item will remain in memory for at least that amount of time
        - 3. the size of in-memory cache is bounded

        """
        for key in list(self.ttl_dict.keys()):
            if time.time() > self.ttl_dict[key]:
                self.cache_dict.pop(key, None)
                self.ttl_dict.pop(key, None)

                # de-reference the removed item
                # https://www.geeksforgeeks.org/diagnosing-and-fixing-memory-leaks-in-python/
                # One of the most common causes of memory leaks in Python is the retention of objects that are no longer being used.
                # This can occur when an object is referenced by another object, but the reference is never removed.

    def set_cache(self, key, value, **kwargs):
        if len(self.cache_dict) >= self.max_size_in_memory:
            # only evict when cache is full
            self.evict_cache()

        self.cache_dict[key] = value
        if "ttl" in kwargs:
            self.ttl_dict[key] = time.time() + kwargs["ttl"]
        else:
            self.ttl_dict[key] = time.time() + self.default_ttl

    async def async_set_cache(self, key, value, **kwargs):
        self.set_cache(key=key, value=value, **kwargs)

    async def async_set_cache_pipeline(self, cache_list, ttl=None, **kwargs):
        for cache_key, cache_value in cache_list:
            if ttl is not None:
                self.set_cache(key=cache_key, value=cache_value, ttl=ttl)
            else:
                self.set_cache(key=cache_key, value=cache_value)

    async def async_set_cache_sadd(self, key, value: List, ttl: Optional[float]):
        """
        Add value to set
        """
        # get the value
        init_value = self.get_cache(key=key) or set()
        for val in value:
            init_value.add(val)
        self.set_cache(key, init_value, ttl=ttl)
        return value

    def get_cache(self, key, **kwargs):
        if key in self.cache_dict:
            if key in self.ttl_dict:
                if time.time() > self.ttl_dict[key]:
                    self.cache_dict.pop(key, None)
                    return None
            original_cached_response = self.cache_dict[key]
            try:
                cached_response = json.loads(original_cached_response)
            except Exception:
                cached_response = original_cached_response
            return cached_response
        return None

    def batch_get_cache(self, keys: list, **kwargs):
        return_val = []
        for k in keys:
            val = self.get_cache(key=k, **kwargs)
            return_val.append(val)
        return return_val

    def increment_cache(self, key, value: int, **kwargs) -> int:
        # get the value
        init_value = self.get_cache(key=key) or 0
        value = init_value + value
        self.set_cache(key, value, **kwargs)
        return value

    async def async_get_cache(self, key, **kwargs):
        return self.get_cache(key=key, **kwargs)

    async def async_batch_get_cache(self, keys: list, **kwargs):
        return_val = []
        for k in keys:
            val = self.get_cache(key=k, **kwargs)
            return_val.append(val)
        return return_val

    async def async_increment(self, key, value: float, **kwargs) -> float:
        # get the value
        init_value = await self.async_get_cache(key=key) or 0
        value = init_value + value
        await self.async_set_cache(key, value, **kwargs)

        return value

    def flush_cache(self):
        self.cache_dict.clear()
        self.ttl_dict.clear()

    async def disconnect(self):
        pass

    def delete_cache(self, key):
        self.cache_dict.pop(key, None)
        self.ttl_dict.pop(key, None)


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

        from ._redis import get_redis_client, get_redis_connection_pool

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
            self.redis_version = self.redis_client.info()["redis_version"]
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
            self.redis_client.ping()
        except Exception as e:
            verbose_logger.error(
                "Error connecting to Sync Redis client", extra={"error": str(e)}
            )

    def init_async_client(self):
        from ._redis import get_redis_async_client

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
        ttl = kwargs.get("ttl", None)
        print_verbose(
            f"Set Redis Cache: key: {key}\nValue {value}\nttl={ttl}, redis_version={self.redis_version}"
        )
        key = self.check_and_fix_namespace(key=key)
        try:
            self.redis_client.set(name=key, value=str(value), ex=ttl)
        except Exception as e:
            # NON blocking - notify users Redis is throwing an exception
            print_verbose(
                f"LiteLLM Caching: set() - Got exception from REDIS : {str(e)}"
            )

    def increment_cache(
        self, key, value: int, ttl: Optional[float] = None, **kwargs
    ) -> int:
        _redis_client = self.redis_client
        start_time = time.time()
        try:
            result = _redis_client.incr(name=key, amount=value)

            if ttl is not None:
                # check if key already has ttl, if not -> set ttl
                current_ttl = _redis_client.ttl(key)
                if current_ttl == -1:
                    # Key has no expiration
                    _redis_client.expire(key, ttl)  # type: ignore
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
            ttl = kwargs.get("ttl", None)
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

    async def async_set_cache_pipeline(self, cache_list, ttl=None, **kwargs):
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
                    # Iterate through each key-value pair in the cache_list and set them in the pipeline.
                    for cache_key, cache_value in cache_list:
                        cache_key = self.check_and_fix_namespace(key=cache_key)
                        print_verbose(
                            f"Set ASYNC Redis Cache PIPELINE: key: {cache_key}\nValue {cache_value}\nttl={ttl}"
                        )
                        json_cache_value = json.dumps(cache_value)
                        # Set the value with a TTL if it's provided.

                        if ttl is not None:
                            pipe.setex(cache_key, ttl, json_cache_value)
                        else:
                            pipe.set(cache_key, json_cache_value)
                    # Execute the pipeline and return the results.
                    results = await pipe.execute()

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
                await redis_client.sadd(key, *value)
                if ttl is not None:
                    _td = timedelta(seconds=ttl)
                    await redis_client.expire(key, _td)
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
        try:
            async with _redis_client as redis_client:
                result = await redis_client.incrbyfloat(name=key, amount=value)

                if ttl is not None:
                    # check if key already has ttl, if not -> set ttl
                    current_ttl = await redis_client.ttl(key)
                    if current_ttl == -1:
                        # Key has no expiration
                        await redis_client.expire(key, ttl)

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
                "LiteLLM Caching: get() - Got exception from REDIS: ", e
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
            results = self.redis_client.mget(keys=_keys)

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
                    f"LiteLLM Caching: async get() - Got exception from REDIS: {str(e)}"
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
            response = self.redis_client.ping()
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

    def client_list(self):
        client_list = self.redis_client.client_list()
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


class RedisSemanticCache(BaseCache):
    def __init__(
        self,
        host=None,
        port=None,
        password=None,
        redis_url=None,
        similarity_threshold=None,
        use_async=False,
        embedding_model="text-embedding-ada-002",
        **kwargs,
    ):
        from redisvl.index import SearchIndex
        from redisvl.query import VectorQuery

        print_verbose(
            "redis semantic-cache initializing INDEX - litellm_semantic_cache_index"
        )
        if similarity_threshold is None:
            raise Exception("similarity_threshold must be provided, passed None")
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model
        schema = {
            "index": {
                "name": "litellm_semantic_cache_index",
                "prefix": "litellm",
                "storage_type": "hash",
            },
            "fields": {
                "text": [{"name": "response"}],
                "vector": [
                    {
                        "name": "litellm_embedding",
                        "dims": 1536,
                        "distance_metric": "cosine",
                        "algorithm": "flat",
                        "datatype": "float32",
                    }
                ],
            },
        }
        if redis_url is None:
            # if no url passed, check if host, port and password are passed, if not raise an Exception
            if host is None or port is None or password is None:
                # try checking env for host, port and password
                import os

                host = os.getenv("REDIS_HOST")
                port = os.getenv("REDIS_PORT")
                password = os.getenv("REDIS_PASSWORD")
                if host is None or port is None or password is None:
                    raise Exception("Redis host, port, and password must be provided")

            redis_url = "redis://:" + password + "@" + host + ":" + port
        print_verbose(f"redis semantic-cache redis_url: {redis_url}")
        if use_async is False:
            self.index = SearchIndex.from_dict(schema)
            self.index.connect(redis_url=redis_url)
            try:
                self.index.create(overwrite=False)  # don't overwrite existing index
            except Exception as e:
                print_verbose(f"Got exception creating semantic cache index: {str(e)}")
        elif use_async is True:
            schema["index"]["name"] = "litellm_semantic_cache_index_async"
            self.index = SearchIndex.from_dict(schema)
            self.index.connect(redis_url=redis_url, use_async=True)

    #
    def _get_cache_logic(self, cached_response: Any):
        """
        Common 'get_cache_logic' across sync + async redis client implementations
        """
        if cached_response is None:
            return cached_response

        # check if cached_response is bytes
        if isinstance(cached_response, bytes):
            cached_response = cached_response.decode("utf-8")

        try:
            cached_response = json.loads(
                cached_response
            )  # Convert string to dictionary
        except Exception:
            cached_response = ast.literal_eval(cached_response)
        return cached_response

    def set_cache(self, key, value, **kwargs):
        import numpy as np

        print_verbose(f"redis semantic-cache set_cache, kwargs: {kwargs}")

        # get the prompt
        messages = kwargs["messages"]
        prompt = "".join(message["content"] for message in messages)

        # create an embedding for prompt
        embedding_response = litellm.embedding(
            model=self.embedding_model,
            input=prompt,
            cache={"no-store": True, "no-cache": True},
        )

        # get the embedding
        embedding = embedding_response["data"][0]["embedding"]

        # make the embedding a numpy array, convert to bytes
        embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
        value = str(value)
        assert isinstance(value, str)

        new_data = [
            {"response": value, "prompt": prompt, "litellm_embedding": embedding_bytes}
        ]

        # Add more data
        self.index.load(new_data)

        return

    def get_cache(self, key, **kwargs):
        print_verbose(f"sync redis semantic-cache get_cache, kwargs: {kwargs}")
        import numpy as np
        from redisvl.query import VectorQuery

        # query
        # get the messages
        messages = kwargs["messages"]
        prompt = "".join(message["content"] for message in messages)

        # convert to embedding
        embedding_response = litellm.embedding(
            model=self.embedding_model,
            input=prompt,
            cache={"no-store": True, "no-cache": True},
        )

        # get the embedding
        embedding = embedding_response["data"][0]["embedding"]

        query = VectorQuery(
            vector=embedding,
            vector_field_name="litellm_embedding",
            return_fields=["response", "prompt", "vector_distance"],
            num_results=1,
        )

        results = self.index.query(query)
        if results is None:
            return None
        if isinstance(results, list):
            if len(results) == 0:
                return None

        vector_distance = results[0]["vector_distance"]
        vector_distance = float(vector_distance)
        similarity = 1 - vector_distance
        cached_prompt = results[0]["prompt"]

        # check similarity, if more than self.similarity_threshold, return results
        print_verbose(
            f"semantic cache: similarity threshold: {self.similarity_threshold}, similarity: {similarity}, prompt: {prompt}, closest_cached_prompt: {cached_prompt}"
        )
        if similarity > self.similarity_threshold:
            # cache hit !
            cached_value = results[0]["response"]
            print_verbose(
                f"got a cache hit, similarity: {similarity}, Current prompt: {prompt}, cached_prompt: {cached_prompt}"
            )
            return self._get_cache_logic(cached_response=cached_value)
        else:
            # cache miss !
            return None

        pass

    async def async_set_cache(self, key, value, **kwargs):
        import numpy as np

        from litellm.proxy.proxy_server import llm_model_list, llm_router

        try:
            await self.index.acreate(overwrite=False)  # don't overwrite existing index
        except Exception as e:
            print_verbose(f"Got exception creating semantic cache index: {str(e)}")
        print_verbose(f"async redis semantic-cache set_cache, kwargs: {kwargs}")

        # get the prompt
        messages = kwargs["messages"]
        prompt = "".join(message["content"] for message in messages)
        # create an embedding for prompt
        router_model_names = (
            [m["model_name"] for m in llm_model_list]
            if llm_model_list is not None
            else []
        )
        if llm_router is not None and self.embedding_model in router_model_names:
            user_api_key = kwargs.get("metadata", {}).get("user_api_key", "")
            embedding_response = await llm_router.aembedding(
                model=self.embedding_model,
                input=prompt,
                cache={"no-store": True, "no-cache": True},
                metadata={
                    "user_api_key": user_api_key,
                    "semantic-cache-embedding": True,
                    "trace_id": kwargs.get("metadata", {}).get("trace_id", None),
                },
            )
        else:
            # convert to embedding
            embedding_response = await litellm.aembedding(
                model=self.embedding_model,
                input=prompt,
                cache={"no-store": True, "no-cache": True},
            )

        # get the embedding
        embedding = embedding_response["data"][0]["embedding"]

        # make the embedding a numpy array, convert to bytes
        embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
        value = str(value)
        assert isinstance(value, str)

        new_data = [
            {"response": value, "prompt": prompt, "litellm_embedding": embedding_bytes}
        ]

        # Add more data
        await self.index.aload(new_data)
        return

    async def async_get_cache(self, key, **kwargs):
        print_verbose(f"async redis semantic-cache get_cache, kwargs: {kwargs}")
        import numpy as np
        from redisvl.query import VectorQuery

        from litellm.proxy.proxy_server import llm_model_list, llm_router

        # query
        # get the messages
        messages = kwargs["messages"]
        prompt = "".join(message["content"] for message in messages)

        router_model_names = (
            [m["model_name"] for m in llm_model_list]
            if llm_model_list is not None
            else []
        )
        if llm_router is not None and self.embedding_model in router_model_names:
            user_api_key = kwargs.get("metadata", {}).get("user_api_key", "")
            embedding_response = await llm_router.aembedding(
                model=self.embedding_model,
                input=prompt,
                cache={"no-store": True, "no-cache": True},
                metadata={
                    "user_api_key": user_api_key,
                    "semantic-cache-embedding": True,
                    "trace_id": kwargs.get("metadata", {}).get("trace_id", None),
                },
            )
        else:
            # convert to embedding
            embedding_response = await litellm.aembedding(
                model=self.embedding_model,
                input=prompt,
                cache={"no-store": True, "no-cache": True},
            )

        # get the embedding
        embedding = embedding_response["data"][0]["embedding"]

        query = VectorQuery(
            vector=embedding,
            vector_field_name="litellm_embedding",
            return_fields=["response", "prompt", "vector_distance"],
        )
        results = await self.index.aquery(query)
        if results is None:
            kwargs.setdefault("metadata", {})["semantic-similarity"] = 0.0
            return None
        if isinstance(results, list):
            if len(results) == 0:
                kwargs.setdefault("metadata", {})["semantic-similarity"] = 0.0
                return None

        vector_distance = results[0]["vector_distance"]
        vector_distance = float(vector_distance)
        similarity = 1 - vector_distance
        cached_prompt = results[0]["prompt"]

        # check similarity, if more than self.similarity_threshold, return results
        print_verbose(
            f"semantic cache: similarity threshold: {self.similarity_threshold}, similarity: {similarity}, prompt: {prompt}, closest_cached_prompt: {cached_prompt}"
        )

        # update kwargs["metadata"] with similarity, don't rewrite the original metadata
        kwargs.setdefault("metadata", {})["semantic-similarity"] = similarity

        if similarity > self.similarity_threshold:
            # cache hit !
            cached_value = results[0]["response"]
            print_verbose(
                f"got a cache hit, similarity: {similarity}, Current prompt: {prompt}, cached_prompt: {cached_prompt}"
            )
            return self._get_cache_logic(cached_response=cached_value)
        else:
            # cache miss !
            return None
        pass

    async def _index_info(self):
        return await self.index.ainfo()


class QdrantSemanticCache(BaseCache):
    def __init__(
        self,
        qdrant_api_base=None,
        qdrant_api_key=None,
        collection_name=None,
        similarity_threshold=None,
        quantization_config=None,
        embedding_model="text-embedding-ada-002",
        host_type=None,
    ):
        import os

        from litellm.llms.custom_httpx.http_handler import (
            _get_httpx_client,
            get_async_httpx_client,
            httpxSpecialProvider,
        )
        from litellm.secret_managers.main import get_secret_str

        if collection_name is None:
            raise Exception("collection_name must be provided, passed None")

        self.collection_name = collection_name
        print_verbose(
            f"qdrant semantic-cache initializing COLLECTION - {self.collection_name}"
        )

        if similarity_threshold is None:
            raise Exception("similarity_threshold must be provided, passed None")
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model
        headers = {}

        # check if defined as os.environ/ variable
        if qdrant_api_base:
            if isinstance(qdrant_api_base, str) and qdrant_api_base.startswith(
                "os.environ/"
            ):
                qdrant_api_base = get_secret_str(qdrant_api_base)
        if qdrant_api_key:
            if isinstance(qdrant_api_key, str) and qdrant_api_key.startswith(
                "os.environ/"
            ):
                qdrant_api_key = get_secret_str(qdrant_api_key)

        qdrant_api_base = (
            qdrant_api_base or os.getenv("QDRANT_URL") or os.getenv("QDRANT_API_BASE")
        )
        qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        headers = {"Content-Type": "application/json"}
        if qdrant_api_key:
            headers["api-key"] = qdrant_api_key

        if qdrant_api_base is None:
            raise ValueError("Qdrant url must be provided")

        self.qdrant_api_base = qdrant_api_base
        self.qdrant_api_key = qdrant_api_key
        print_verbose(f"qdrant semantic-cache qdrant_api_base: {self.qdrant_api_base}")

        self.headers = headers

        self.sync_client = _get_httpx_client()
        self.async_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.Caching
        )

        if quantization_config is None:
            print_verbose(
                "Quantization config is not provided. Default binary quantization will be used."
            )
        collection_exists = self.sync_client.get(
            url=f"{self.qdrant_api_base}/collections/{self.collection_name}/exists",
            headers=self.headers,
        )
        if collection_exists.status_code != 200:
            raise ValueError(
                f"Error from qdrant checking if /collections exist {collection_exists.text}"
            )

        if collection_exists.json()["result"]["exists"]:
            collection_details = self.sync_client.get(
                url=f"{self.qdrant_api_base}/collections/{self.collection_name}",
                headers=self.headers,
            )
            self.collection_info = collection_details.json()
            print_verbose(
                f"Collection already exists.\nCollection details:{self.collection_info}"
            )
        else:
            if quantization_config is None or quantization_config == "binary":
                quantization_params = {
                    "binary": {
                        "always_ram": False,
                    }
                }
            elif quantization_config == "scalar":
                quantization_params = {
                    "scalar": {"type": "int8", "quantile": 0.99, "always_ram": False}
                }
            elif quantization_config == "product":
                quantization_params = {
                    "product": {"compression": "x16", "always_ram": False}
                }
            else:
                raise Exception(
                    "Quantization config must be one of 'scalar', 'binary' or 'product'"
                )

            new_collection_status = self.sync_client.put(
                url=f"{self.qdrant_api_base}/collections/{self.collection_name}",
                json={
                    "vectors": {"size": 1536, "distance": "Cosine"},
                    "quantization_config": quantization_params,
                },
                headers=self.headers,
            )
            if new_collection_status.json()["result"]:
                collection_details = self.sync_client.get(
                    url=f"{self.qdrant_api_base}/collections/{self.collection_name}",
                    headers=self.headers,
                )
                self.collection_info = collection_details.json()
                print_verbose(
                    f"New collection created.\nCollection details:{self.collection_info}"
                )
            else:
                raise Exception("Error while creating new collection")

    def _get_cache_logic(self, cached_response: Any):
        if cached_response is None:
            return cached_response
        try:
            cached_response = json.loads(
                cached_response
            )  # Convert string to dictionary
        except Exception:
            cached_response = ast.literal_eval(cached_response)
        return cached_response

    def set_cache(self, key, value, **kwargs):
        print_verbose(f"qdrant semantic-cache set_cache, kwargs: {kwargs}")
        import uuid

        # get the prompt
        messages = kwargs["messages"]
        prompt = ""
        for message in messages:
            prompt += message["content"]

        # create an embedding for prompt
        embedding_response = litellm.embedding(
            model=self.embedding_model,
            input=prompt,
            cache={"no-store": True, "no-cache": True},
        )

        # get the embedding
        embedding = embedding_response["data"][0]["embedding"]

        value = str(value)
        assert isinstance(value, str)

        data = {
            "points": [
                {
                    "id": str(uuid.uuid4()),
                    "vector": embedding,
                    "payload": {
                        "text": prompt,
                        "response": value,
                    },
                },
            ]
        }
        self.sync_client.put(
            url=f"{self.qdrant_api_base}/collections/{self.collection_name}/points",
            headers=self.headers,
            json=data,
        )
        return

    def get_cache(self, key, **kwargs):
        print_verbose(f"sync qdrant semantic-cache get_cache, kwargs: {kwargs}")

        # get the messages
        messages = kwargs["messages"]
        prompt = ""
        for message in messages:
            prompt += message["content"]

        # convert to embedding
        embedding_response = litellm.embedding(
            model=self.embedding_model,
            input=prompt,
            cache={"no-store": True, "no-cache": True},
        )

        # get the embedding
        embedding = embedding_response["data"][0]["embedding"]

        data = {
            "vector": embedding,
            "params": {
                "quantization": {
                    "ignore": False,
                    "rescore": True,
                    "oversampling": 3.0,
                }
            },
            "limit": 1,
            "with_payload": True,
        }

        search_response = self.sync_client.post(
            url=f"{self.qdrant_api_base}/collections/{self.collection_name}/points/search",
            headers=self.headers,
            json=data,
        )
        results = search_response.json()["result"]

        if results is None:
            return None
        if isinstance(results, list):
            if len(results) == 0:
                return None

        similarity = results[0]["score"]
        cached_prompt = results[0]["payload"]["text"]

        # check similarity, if more than self.similarity_threshold, return results
        print_verbose(
            f"semantic cache: similarity threshold: {self.similarity_threshold}, similarity: {similarity}, prompt: {prompt}, closest_cached_prompt: {cached_prompt}"
        )
        if similarity >= self.similarity_threshold:
            # cache hit !
            cached_value = results[0]["payload"]["response"]
            print_verbose(
                f"got a cache hit, similarity: {similarity}, Current prompt: {prompt}, cached_prompt: {cached_prompt}"
            )
            return self._get_cache_logic(cached_response=cached_value)
        else:
            # cache miss !
            return None
        pass

    async def async_set_cache(self, key, value, **kwargs):
        import uuid

        from litellm.proxy.proxy_server import llm_model_list, llm_router

        print_verbose(f"async qdrant semantic-cache set_cache, kwargs: {kwargs}")

        # get the prompt
        messages = kwargs["messages"]
        prompt = ""
        for message in messages:
            prompt += message["content"]
        # create an embedding for prompt
        router_model_names = (
            [m["model_name"] for m in llm_model_list]
            if llm_model_list is not None
            else []
        )
        if llm_router is not None and self.embedding_model in router_model_names:
            user_api_key = kwargs.get("metadata", {}).get("user_api_key", "")
            embedding_response = await llm_router.aembedding(
                model=self.embedding_model,
                input=prompt,
                cache={"no-store": True, "no-cache": True},
                metadata={
                    "user_api_key": user_api_key,
                    "semantic-cache-embedding": True,
                    "trace_id": kwargs.get("metadata", {}).get("trace_id", None),
                },
            )
        else:
            # convert to embedding
            embedding_response = await litellm.aembedding(
                model=self.embedding_model,
                input=prompt,
                cache={"no-store": True, "no-cache": True},
            )

        # get the embedding
        embedding = embedding_response["data"][0]["embedding"]

        value = str(value)
        assert isinstance(value, str)

        data = {
            "points": [
                {
                    "id": str(uuid.uuid4()),
                    "vector": embedding,
                    "payload": {
                        "text": prompt,
                        "response": value,
                    },
                },
            ]
        }

        await self.async_client.put(
            url=f"{self.qdrant_api_base}/collections/{self.collection_name}/points",
            headers=self.headers,
            json=data,
        )
        return

    async def async_get_cache(self, key, **kwargs):
        print_verbose(f"async qdrant semantic-cache get_cache, kwargs: {kwargs}")
        from litellm.proxy.proxy_server import llm_model_list, llm_router

        # get the messages
        messages = kwargs["messages"]
        prompt = ""
        for message in messages:
            prompt += message["content"]

        router_model_names = (
            [m["model_name"] for m in llm_model_list]
            if llm_model_list is not None
            else []
        )
        if llm_router is not None and self.embedding_model in router_model_names:
            user_api_key = kwargs.get("metadata", {}).get("user_api_key", "")
            embedding_response = await llm_router.aembedding(
                model=self.embedding_model,
                input=prompt,
                cache={"no-store": True, "no-cache": True},
                metadata={
                    "user_api_key": user_api_key,
                    "semantic-cache-embedding": True,
                    "trace_id": kwargs.get("metadata", {}).get("trace_id", None),
                },
            )
        else:
            # convert to embedding
            embedding_response = await litellm.aembedding(
                model=self.embedding_model,
                input=prompt,
                cache={"no-store": True, "no-cache": True},
            )

        # get the embedding
        embedding = embedding_response["data"][0]["embedding"]

        data = {
            "vector": embedding,
            "params": {
                "quantization": {
                    "ignore": False,
                    "rescore": True,
                    "oversampling": 3.0,
                }
            },
            "limit": 1,
            "with_payload": True,
        }

        search_response = await self.async_client.post(
            url=f"{self.qdrant_api_base}/collections/{self.collection_name}/points/search",
            headers=self.headers,
            json=data,
        )

        results = search_response.json()["result"]

        if results is None:
            kwargs.setdefault("metadata", {})["semantic-similarity"] = 0.0
            return None
        if isinstance(results, list):
            if len(results) == 0:
                kwargs.setdefault("metadata", {})["semantic-similarity"] = 0.0
                return None

        similarity = results[0]["score"]
        cached_prompt = results[0]["payload"]["text"]

        # check similarity, if more than self.similarity_threshold, return results
        print_verbose(
            f"semantic cache: similarity threshold: {self.similarity_threshold}, similarity: {similarity}, prompt: {prompt}, closest_cached_prompt: {cached_prompt}"
        )

        # update kwargs["metadata"] with similarity, don't rewrite the original metadata
        kwargs.setdefault("metadata", {})["semantic-similarity"] = similarity

        if similarity >= self.similarity_threshold:
            # cache hit !
            cached_value = results[0]["payload"]["response"]
            print_verbose(
                f"got a cache hit, similarity: {similarity}, Current prompt: {prompt}, cached_prompt: {cached_prompt}"
            )
            return self._get_cache_logic(cached_response=cached_value)
        else:
            # cache miss !
            return None
        pass

    async def _collection_info(self):
        return self.collection_info


class S3Cache(BaseCache):
    def __init__(
        self,
        s3_bucket_name,
        s3_region_name=None,
        s3_api_version=None,
        s3_use_ssl: Optional[bool] = True,
        s3_verify=None,
        s3_endpoint_url=None,
        s3_aws_access_key_id=None,
        s3_aws_secret_access_key=None,
        s3_aws_session_token=None,
        s3_config=None,
        s3_path=None,
        **kwargs,
    ):
        import boto3

        self.bucket_name = s3_bucket_name
        self.key_prefix = s3_path.rstrip("/") + "/" if s3_path else ""
        # Create an S3 client with custom endpoint URL

        self.s3_client = boto3.client(
            "s3",
            region_name=s3_region_name,
            endpoint_url=s3_endpoint_url,
            api_version=s3_api_version,
            use_ssl=s3_use_ssl,
            verify=s3_verify,
            aws_access_key_id=s3_aws_access_key_id,
            aws_secret_access_key=s3_aws_secret_access_key,
            aws_session_token=s3_aws_session_token,
            config=s3_config,
            **kwargs,
        )

    def set_cache(self, key, value, **kwargs):
        try:
            print_verbose(f"LiteLLM SET Cache - S3. Key={key}. Value={value}")
            ttl = kwargs.get("ttl", None)
            # Convert value to JSON before storing in S3
            serialized_value = json.dumps(value)
            key = self.key_prefix + key

            if ttl is not None:
                cache_control = f"immutable, max-age={ttl}, s-maxage={ttl}"
                import datetime

                # Calculate expiration time
                expiration_time = datetime.datetime.now() + ttl

                # Upload the data to S3 with the calculated expiration time
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=key,
                    Body=serialized_value,
                    Expires=expiration_time,
                    CacheControl=cache_control,
                    ContentType="application/json",
                    ContentLanguage="en",
                    ContentDisposition=f'inline; filename="{key}.json"',
                )
            else:
                cache_control = "immutable, max-age=31536000, s-maxage=31536000"
                # Upload the data to S3 without specifying Expires
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=key,
                    Body=serialized_value,
                    CacheControl=cache_control,
                    ContentType="application/json",
                    ContentLanguage="en",
                    ContentDisposition=f'inline; filename="{key}.json"',
                )
        except Exception as e:
            # NON blocking - notify users S3 is throwing an exception
            print_verbose(f"S3 Caching: set_cache() - Got exception from S3: {e}")

    async def async_set_cache(self, key, value, **kwargs):
        self.set_cache(key=key, value=value, **kwargs)

    def get_cache(self, key, **kwargs):
        import boto3
        import botocore

        try:
            key = self.key_prefix + key

            print_verbose(f"Get S3 Cache: key: {key}")
            # Download the data from S3
            cached_response = self.s3_client.get_object(
                Bucket=self.bucket_name, Key=key
            )

            if cached_response is not None:
                # cached_response is in `b{} convert it to ModelResponse
                cached_response = (
                    cached_response["Body"].read().decode("utf-8")
                )  # Convert bytes to string
                try:
                    cached_response = json.loads(
                        cached_response
                    )  # Convert string to dictionary
                except Exception:
                    cached_response = ast.literal_eval(cached_response)
            if type(cached_response) is not dict:
                cached_response = dict(cached_response)
            verbose_logger.debug(
                f"Got S3 Cache: key: {key}, cached_response {cached_response}. Type Response {type(cached_response)}"
            )

            return cached_response
        except botocore.exceptions.ClientError as e:  # type: ignore
            if e.response["Error"]["Code"] == "NoSuchKey":
                verbose_logger.debug(
                    f"S3 Cache: The specified key '{key}' does not exist in the S3 bucket."
                )
                return None

        except Exception as e:
            # NON blocking - notify users S3 is throwing an exception
            verbose_logger.error(
                f"S3 Caching: get_cache() - Got exception from S3: {e}"
            )

    async def async_get_cache(self, key, **kwargs):
        return self.get_cache(key=key, **kwargs)

    def flush_cache(self):
        pass

    async def disconnect(self):
        pass


class DualCache(BaseCache):
    """
    This updates both Redis and an in-memory cache simultaneously.
    When data is updated or inserted, it is written to both the in-memory cache + Redis.
    This ensures that even if Redis hasn't been updated yet, the in-memory cache reflects the most recent data.
    """

    def __init__(
        self,
        in_memory_cache: Optional[InMemoryCache] = None,
        redis_cache: Optional[RedisCache] = None,
        default_in_memory_ttl: Optional[float] = None,
        default_redis_ttl: Optional[float] = None,
        always_read_redis: Optional[bool] = True,
    ) -> None:
        super().__init__()
        # If in_memory_cache is not provided, use the default InMemoryCache
        self.in_memory_cache = in_memory_cache or InMemoryCache()
        # If redis_cache is not provided, use the default RedisCache
        self.redis_cache = redis_cache

        self.default_in_memory_ttl = (
            default_in_memory_ttl or litellm.default_in_memory_ttl
        )
        self.default_redis_ttl = default_redis_ttl or litellm.default_redis_ttl
        self.always_read_redis = always_read_redis

    def update_cache_ttl(
        self, default_in_memory_ttl: Optional[float], default_redis_ttl: Optional[float]
    ):
        if default_in_memory_ttl is not None:
            self.default_in_memory_ttl = default_in_memory_ttl

        if default_redis_ttl is not None:
            self.default_redis_ttl = default_redis_ttl

    def set_cache(self, key, value, local_only: bool = False, **kwargs):
        # Update both Redis and in-memory cache
        try:
            if self.in_memory_cache is not None:
                if "ttl" not in kwargs and self.default_in_memory_ttl is not None:
                    kwargs["ttl"] = self.default_in_memory_ttl

                self.in_memory_cache.set_cache(key, value, **kwargs)

            if self.redis_cache is not None and local_only is False:
                self.redis_cache.set_cache(key, value, **kwargs)
        except Exception as e:
            print_verbose(e)

    def increment_cache(
        self, key, value: int, local_only: bool = False, **kwargs
    ) -> int:
        """
        Key - the key in cache

        Value - int - the value you want to increment by

        Returns - int - the incremented value
        """
        try:
            result: int = value
            if self.in_memory_cache is not None:
                result = self.in_memory_cache.increment_cache(key, value, **kwargs)

            if self.redis_cache is not None and local_only is False:
                result = self.redis_cache.increment_cache(key, value, **kwargs)

            return result
        except Exception as e:
            verbose_logger.error(f"LiteLLM Cache: Excepton async add_cache: {str(e)}")
            raise e

    def get_cache(self, key, local_only: bool = False, **kwargs):
        # Try to fetch from in-memory cache first
        try:
            result = None
            if self.in_memory_cache is not None:
                in_memory_result = self.in_memory_cache.get_cache(key, **kwargs)

                if in_memory_result is not None:
                    result = in_memory_result

            if (
                (self.always_read_redis is True)
                and self.redis_cache is not None
                and local_only is False
            ):
                # If not found in in-memory cache or always_read_redis is True, try fetching from Redis
                redis_result = self.redis_cache.get_cache(key, **kwargs)

                if redis_result is not None:
                    # Update in-memory cache with the value from Redis
                    self.in_memory_cache.set_cache(key, redis_result, **kwargs)

                result = redis_result

            print_verbose(f"get cache: cache result: {result}")
            return result
        except Exception:
            verbose_logger.error(traceback.format_exc())

    def batch_get_cache(self, keys: list, local_only: bool = False, **kwargs):
        try:
            result = [None for _ in range(len(keys))]
            if self.in_memory_cache is not None:
                in_memory_result = self.in_memory_cache.batch_get_cache(keys, **kwargs)

                if in_memory_result is not None:
                    result = in_memory_result

            if None in result and self.redis_cache is not None and local_only is False:
                """
                - for the none values in the result
                - check the redis cache
                """
                sublist_keys = [
                    key for key, value in zip(keys, result) if value is None
                ]
                # If not found in in-memory cache, try fetching from Redis
                redis_result = self.redis_cache.batch_get_cache(sublist_keys, **kwargs)
                if redis_result is not None:
                    # Update in-memory cache with the value from Redis
                    for key in redis_result:
                        self.in_memory_cache.set_cache(key, redis_result[key], **kwargs)

                for key, value in redis_result.items():
                    result[keys.index(key)] = value

            print_verbose(f"async batch get cache: cache result: {result}")
            return result
        except Exception:
            verbose_logger.error(traceback.format_exc())

    async def async_get_cache(self, key, local_only: bool = False, **kwargs):
        # Try to fetch from in-memory cache first
        try:
            print_verbose(
                f"async get cache: cache key: {key}; local_only: {local_only}"
            )
            result = None
            if self.in_memory_cache is not None:
                in_memory_result = await self.in_memory_cache.async_get_cache(
                    key, **kwargs
                )

                print_verbose(f"in_memory_result: {in_memory_result}")
                if in_memory_result is not None:
                    result = in_memory_result

            if result is None and self.redis_cache is not None and local_only is False:
                # If not found in in-memory cache, try fetching from Redis
                redis_result = await self.redis_cache.async_get_cache(key, **kwargs)

                if redis_result is not None:
                    # Update in-memory cache with the value from Redis
                    await self.in_memory_cache.async_set_cache(
                        key, redis_result, **kwargs
                    )

                result = redis_result

            print_verbose(f"get cache: cache result: {result}")
            return result
        except Exception:
            verbose_logger.error(traceback.format_exc())

    async def async_batch_get_cache(
        self, keys: list, local_only: bool = False, **kwargs
    ):
        try:
            result = [None for _ in range(len(keys))]
            if self.in_memory_cache is not None:
                in_memory_result = await self.in_memory_cache.async_batch_get_cache(
                    keys, **kwargs
                )

                if in_memory_result is not None:
                    result = in_memory_result
            if None in result and self.redis_cache is not None and local_only is False:
                """
                - for the none values in the result
                - check the redis cache
                """
                sublist_keys = [
                    key for key, value in zip(keys, result) if value is None
                ]
                # If not found in in-memory cache, try fetching from Redis
                redis_result = await self.redis_cache.async_batch_get_cache(
                    sublist_keys, **kwargs
                )

                if redis_result is not None:
                    # Update in-memory cache with the value from Redis
                    for key, value in redis_result.items():
                        if value is not None:
                            await self.in_memory_cache.async_set_cache(
                                key, redis_result[key], **kwargs
                            )
                for key, value in redis_result.items():
                    index = keys.index(key)
                    result[index] = value

            return result
        except Exception:
            verbose_logger.error(traceback.format_exc())

    async def async_set_cache(self, key, value, local_only: bool = False, **kwargs):
        print_verbose(
            f"async set cache: cache key: {key}; local_only: {local_only}; value: {value}"
        )
        try:
            if self.in_memory_cache is not None:
                await self.in_memory_cache.async_set_cache(key, value, **kwargs)

            if self.redis_cache is not None and local_only is False:
                await self.redis_cache.async_set_cache(key, value, **kwargs)
        except Exception as e:
            verbose_logger.exception(
                f"LiteLLM Cache: Excepton async add_cache: {str(e)}"
            )

    async def async_batch_set_cache(
        self, cache_list: list, local_only: bool = False, **kwargs
    ):
        """
        Batch write values to the cache
        """
        print_verbose(
            f"async batch set cache: cache keys: {cache_list}; local_only: {local_only}"
        )
        try:
            if self.in_memory_cache is not None:
                await self.in_memory_cache.async_set_cache_pipeline(
                    cache_list=cache_list, **kwargs
                )

            if self.redis_cache is not None and local_only is False:
                await self.redis_cache.async_set_cache_pipeline(
                    cache_list=cache_list, ttl=kwargs.pop("ttl", None), **kwargs
                )
        except Exception as e:
            verbose_logger.exception(
                f"LiteLLM Cache: Excepton async add_cache: {str(e)}"
            )

    async def async_increment_cache(
        self, key, value: float, local_only: bool = False, **kwargs
    ) -> float:
        """
        Key - the key in cache

        Value - float - the value you want to increment by

        Returns - float - the incremented value
        """
        try:
            result: float = value
            if self.in_memory_cache is not None:
                result = await self.in_memory_cache.async_increment(
                    key, value, **kwargs
                )

            if self.redis_cache is not None and local_only is False:
                result = await self.redis_cache.async_increment(key, value, **kwargs)

            return result
        except Exception as e:
            raise e  # don't log if exception is raised

    async def async_set_cache_sadd(
        self, key, value: List, local_only: bool = False, **kwargs
    ) -> None:
        """
        Add value to a set

        Key - the key in cache

        Value - str - the value you want to add to the set

        Returns - None
        """
        try:
            if self.in_memory_cache is not None:
                _ = await self.in_memory_cache.async_set_cache_sadd(
                    key, value, ttl=kwargs.get("ttl", None)
                )

            if self.redis_cache is not None and local_only is False:
                _ = await self.redis_cache.async_set_cache_sadd(
                    key, value, ttl=kwargs.get("ttl", None) ** kwargs
                )

            return None
        except Exception as e:
            raise e  # don't log, if exception is raised

    def flush_cache(self):
        if self.in_memory_cache is not None:
            self.in_memory_cache.flush_cache()
        if self.redis_cache is not None:
            self.redis_cache.flush_cache()

    def delete_cache(self, key):
        """
        Delete a key from the cache
        """
        if self.in_memory_cache is not None:
            self.in_memory_cache.delete_cache(key)
        if self.redis_cache is not None:
            self.redis_cache.delete_cache(key)

    async def async_delete_cache(self, key: str):
        """
        Delete a key from the cache
        """
        if self.in_memory_cache is not None:
            self.in_memory_cache.delete_cache(key)
        if self.redis_cache is not None:
            await self.redis_cache.async_delete_cache(key)


#### LiteLLM.Completion / Embedding Cache ####
class Cache:
    def __init__(
        self,
        type: Optional[
            Literal["local", "redis", "redis-semantic", "s3", "disk", "qdrant-semantic"]
        ] = "local",
        mode: Optional[
            CacheMode
        ] = CacheMode.default_on,  # when default_on cache is always on, when default_off cache is opt in
        host: Optional[str] = None,
        port: Optional[str] = None,
        password: Optional[str] = None,
        namespace: Optional[str] = None,
        ttl: Optional[float] = None,
        default_in_memory_ttl: Optional[float] = None,
        default_in_redis_ttl: Optional[float] = None,
        similarity_threshold: Optional[float] = None,
        supported_call_types: Optional[
            List[
                Literal[
                    "completion",
                    "acompletion",
                    "embedding",
                    "aembedding",
                    "atranscription",
                    "transcription",
                    "atext_completion",
                    "text_completion",
                ]
            ]
        ] = [
            "completion",
            "acompletion",
            "embedding",
            "aembedding",
            "atranscription",
            "transcription",
            "atext_completion",
            "text_completion",
        ],
        # s3 Bucket, boto3 configuration
        s3_bucket_name: Optional[str] = None,
        s3_region_name: Optional[str] = None,
        s3_api_version: Optional[str] = None,
        s3_use_ssl: Optional[bool] = True,
        s3_verify: Optional[Union[bool, str]] = None,
        s3_endpoint_url: Optional[str] = None,
        s3_aws_access_key_id: Optional[str] = None,
        s3_aws_secret_access_key: Optional[str] = None,
        s3_aws_session_token: Optional[str] = None,
        s3_config: Optional[Any] = None,
        s3_path: Optional[str] = None,
        redis_semantic_cache_use_async=False,
        redis_semantic_cache_embedding_model="text-embedding-ada-002",
        redis_flush_size: Optional[int] = None,
        redis_startup_nodes: Optional[List] = None,
        disk_cache_dir=None,
        qdrant_api_base: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        qdrant_collection_name: Optional[str] = None,
        qdrant_quantization_config: Optional[str] = None,
        qdrant_semantic_cache_embedding_model="text-embedding-ada-002",
        **kwargs,
    ):
        """
        Initializes the cache based on the given type.

        Args:
            type (str, optional): The type of cache to initialize. Can be "local", "redis", "redis-semantic", "qdrant-semantic", "s3" or "disk". Defaults to "local".
            host (str, optional): The host address for the Redis cache. Required if type is "redis".
            port (int, optional): The port number for the Redis cache. Required if type is "redis".
            password (str, optional): The password for the Redis cache. Required if type is "redis".
            qdrant_api_base (str, optional): The url for your qdrant cluster. Required if type is "qdrant-semantic".
            qdrant_api_key (str, optional): The api_key for the local or cloud qdrant cluster.
            qdrant_collection_name (str, optional): The name for your qdrant collection. Required if type is "qdrant-semantic".
            similarity_threshold (float, optional): The similarity threshold for semantic-caching, Required if type is "redis-semantic" or "qdrant-semantic".

            supported_call_types (list, optional): List of call types to cache for. Defaults to cache == on for all call types.
            **kwargs: Additional keyword arguments for redis.Redis() cache

        Raises:
            ValueError: If an invalid cache type is provided.

        Returns:
            None. Cache is set as a litellm param
        """
        if type == "redis":
            self.cache: BaseCache = RedisCache(
                host,
                port,
                password,
                redis_flush_size,
                startup_nodes=redis_startup_nodes,
                **kwargs,
            )
        elif type == "redis-semantic":
            self.cache = RedisSemanticCache(
                host,
                port,
                password,
                similarity_threshold=similarity_threshold,
                use_async=redis_semantic_cache_use_async,
                embedding_model=redis_semantic_cache_embedding_model,
                **kwargs,
            )
        elif type == "qdrant-semantic":
            self.cache = QdrantSemanticCache(
                qdrant_api_base=qdrant_api_base,
                qdrant_api_key=qdrant_api_key,
                collection_name=qdrant_collection_name,
                similarity_threshold=similarity_threshold,
                quantization_config=qdrant_quantization_config,
                embedding_model=qdrant_semantic_cache_embedding_model,
            )
        elif type == "local":
            self.cache = InMemoryCache()
        elif type == "s3":
            self.cache = S3Cache(
                s3_bucket_name=s3_bucket_name,
                s3_region_name=s3_region_name,
                s3_api_version=s3_api_version,
                s3_use_ssl=s3_use_ssl,
                s3_verify=s3_verify,
                s3_endpoint_url=s3_endpoint_url,
                s3_aws_access_key_id=s3_aws_access_key_id,
                s3_aws_secret_access_key=s3_aws_secret_access_key,
                s3_aws_session_token=s3_aws_session_token,
                s3_config=s3_config,
                s3_path=s3_path,
                **kwargs,
            )
        elif type == "disk":
            self.cache = DiskCache(disk_cache_dir=disk_cache_dir)
        if "cache" not in litellm.input_callback:
            litellm.input_callback.append("cache")
        if "cache" not in litellm.success_callback:
            litellm.success_callback.append("cache")
        if "cache" not in litellm._async_success_callback:
            litellm._async_success_callback.append("cache")
        self.supported_call_types = supported_call_types  # default to ["completion", "acompletion", "embedding", "aembedding"]
        self.type = type
        self.namespace = namespace
        self.redis_flush_size = redis_flush_size
        self.ttl = ttl
        self.mode: CacheMode = mode or CacheMode.default_on

        if self.type == "local" and default_in_memory_ttl is not None:
            self.ttl = default_in_memory_ttl

        if (
            self.type == "redis" or self.type == "redis-semantic"
        ) and default_in_redis_ttl is not None:
            self.ttl = default_in_redis_ttl

        if self.namespace is not None and isinstance(self.cache, RedisCache):
            self.cache.namespace = self.namespace

    def get_cache_key(self, *args, **kwargs) -> str:
        """
        Get the cache key for the given arguments.

        Args:
            *args: args to litellm.completion() or embedding()
            **kwargs: kwargs to litellm.completion() or embedding()

        Returns:
            str: The cache key generated from the arguments, or None if no cache key could be generated.
        """
        cache_key = ""
        print_verbose(f"\nGetting Cache key. Kwargs: {kwargs}")

        # for streaming, we use preset_cache_key. It's created in wrapper(), we do this because optional params like max_tokens, get transformed for bedrock -> max_new_tokens
        if kwargs.get("litellm_params", {}).get("preset_cache_key", None) is not None:
            _preset_cache_key = kwargs.get("litellm_params", {}).get(
                "preset_cache_key", None
            )
            print_verbose(f"\nReturning preset cache key: {_preset_cache_key}")
            return _preset_cache_key

        # sort kwargs by keys, since model: [gpt-4, temperature: 0.2, max_tokens: 200] == [temperature: 0.2, max_tokens: 200, model: gpt-4]
        completion_kwargs = [
            "model",
            "messages",
            "prompt",
            "temperature",
            "top_p",
            "n",
            "stop",
            "max_tokens",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "user",
            "response_format",
            "seed",
            "tools",
            "tool_choice",
            "stream",
        ]
        embedding_only_kwargs = [
            "input",
            "encoding_format",
        ]  # embedding kwargs = model, input, user, encoding_format. Model, user are checked in completion_kwargs
        transcription_only_kwargs = [
            "file",
            "language",
        ]
        # combined_kwargs - NEEDS to be ordered across get_cache_key(). Do not use a set()
        combined_kwargs = (
            completion_kwargs + embedding_only_kwargs + transcription_only_kwargs
        )
        litellm_param_kwargs = all_litellm_params
        for param in kwargs:
            if param in combined_kwargs:
                # check if param == model and model_group is passed in, then override model with model_group
                if param == "model":
                    model_group = None
                    caching_group = None
                    metadata = kwargs.get("metadata", None)
                    litellm_params = kwargs.get("litellm_params", {})
                    if metadata is not None:
                        model_group = metadata.get("model_group")
                        model_group = metadata.get("model_group", None)
                        caching_groups = metadata.get("caching_groups", None)
                        if caching_groups:
                            for group in caching_groups:
                                if model_group in group:
                                    caching_group = group
                                    break
                    if litellm_params is not None:
                        metadata = litellm_params.get("metadata", None)
                        if metadata is not None:
                            model_group = metadata.get("model_group", None)
                            caching_groups = metadata.get("caching_groups", None)
                            if caching_groups:
                                for group in caching_groups:
                                    if model_group in group:
                                        caching_group = group
                                        break
                    param_value = (
                        caching_group or model_group or kwargs[param]
                    )  # use caching_group, if set then model_group if it exists, else use kwargs["model"]
                elif param == "file":
                    file = kwargs.get("file")
                    metadata = kwargs.get("metadata", {})
                    litellm_params = kwargs.get("litellm_params", {})

                    # get checksum of file content
                    param_value = (
                        metadata.get("file_checksum")
                        or getattr(file, "name", None)
                        or metadata.get("file_name")
                        or litellm_params.get("file_name")
                    )
                else:
                    if kwargs[param] is None:
                        continue  # ignore None params
                    param_value = kwargs[param]
                cache_key += f"{str(param)}: {str(param_value)}"
            elif (
                param not in litellm_param_kwargs
            ):  # check if user passed in optional param - e.g. top_k
                if (
                    litellm.enable_caching_on_provider_specific_optional_params is True
                ):  # feature flagged for now
                    if kwargs[param] is None:
                        continue  # ignore None params
                    param_value = kwargs[param]
                    cache_key += f"{str(param)}: {str(param_value)}"

        print_verbose(f"\nCreated cache key: {cache_key}")
        # Use hashlib to create a sha256 hash of the cache key
        hash_object = hashlib.sha256(cache_key.encode())
        # Hexadecimal representation of the hash
        hash_hex = hash_object.hexdigest()
        print_verbose(f"Hashed cache key (SHA-256): {hash_hex}")
        if kwargs.get("metadata", {}).get("redis_namespace", None) is not None:
            _namespace = kwargs.get("metadata", {}).get("redis_namespace", None)
            hash_hex = f"{_namespace}:{hash_hex}"
            print_verbose(f"Hashed Key with Namespace: {hash_hex}")
        elif self.namespace is not None:
            hash_hex = f"{self.namespace}:{hash_hex}"
            print_verbose(f"Hashed Key with Namespace: {hash_hex}")
        return hash_hex

    def generate_streaming_content(self, content):
        chunk_size = 5  # Adjust the chunk size as needed
        for i in range(0, len(content), chunk_size):
            yield {
                "choices": [
                    {
                        "delta": {
                            "role": "assistant",
                            "content": content[i : i + chunk_size],
                        }
                    }
                ]
            }
            time.sleep(0.02)

    def _get_cache_logic(
        self,
        cached_result: Optional[Any],
        max_age: Optional[float],
    ):
        """
        Common get cache logic across sync + async implementations
        """
        # Check if a timestamp was stored with the cached response
        if (
            cached_result is not None
            and isinstance(cached_result, dict)
            and "timestamp" in cached_result
        ):
            timestamp = cached_result["timestamp"]
            current_time = time.time()

            # Calculate age of the cached response
            response_age = current_time - timestamp

            # Check if the cached response is older than the max-age
            if max_age is not None and response_age > max_age:
                return None  # Cached response is too old

            # If the response is fresh, or there's no max-age requirement, return the cached response
            # cached_response is in `b{} convert it to ModelResponse
            cached_response = cached_result.get("response")
            try:
                if isinstance(cached_response, dict):
                    pass
                else:
                    cached_response = json.loads(
                        cached_response  # type: ignore
                    )  # Convert string to dictionary
            except Exception:
                cached_response = ast.literal_eval(cached_response)  # type: ignore
            return cached_response
        return cached_result

    def get_cache(self, *args, **kwargs):
        """
        Retrieves the cached result for the given arguments.

        Args:
            *args: args to litellm.completion() or embedding()
            **kwargs: kwargs to litellm.completion() or embedding()

        Returns:
            The cached result if it exists, otherwise None.
        """
        try:  # never block execution
            if self.should_use_cache(*args, **kwargs) is not True:
                return
            messages = kwargs.get("messages", [])
            if "cache_key" in kwargs:
                cache_key = kwargs["cache_key"]
            else:
                cache_key = self.get_cache_key(*args, **kwargs)
            if cache_key is not None:
                cache_control_args = kwargs.get("cache", {})
                max_age = cache_control_args.get(
                    "s-max-age", cache_control_args.get("s-maxage", float("inf"))
                )
                cached_result = self.cache.get_cache(cache_key, messages=messages)
                return self._get_cache_logic(
                    cached_result=cached_result, max_age=max_age
                )
        except Exception:
            print_verbose(f"An exception occurred: {traceback.format_exc()}")
            return None

    async def async_get_cache(self, *args, **kwargs):
        """
        Async get cache implementation.

        Used for embedding calls in async wrapper
        """
        try:  # never block execution
            if self.should_use_cache(*args, **kwargs) is not True:
                return

            kwargs.get("messages", [])
            if "cache_key" in kwargs:
                cache_key = kwargs["cache_key"]
            else:
                cache_key = self.get_cache_key(*args, **kwargs)
            if cache_key is not None:
                cache_control_args = kwargs.get("cache", {})
                max_age = cache_control_args.get(
                    "s-max-age", cache_control_args.get("s-maxage", float("inf"))
                )
                cached_result = await self.cache.async_get_cache(
                    cache_key, *args, **kwargs
                )
                return self._get_cache_logic(
                    cached_result=cached_result, max_age=max_age
                )
        except Exception:
            print_verbose(f"An exception occurred: {traceback.format_exc()}")
            return None

    def _add_cache_logic(self, result, *args, **kwargs):
        """
        Common implementation across sync + async add_cache functions
        """
        try:
            if "cache_key" in kwargs:
                cache_key = kwargs["cache_key"]
            else:
                cache_key = self.get_cache_key(*args, **kwargs)
            if cache_key is not None:
                if isinstance(result, OpenAIObject):
                    result = result.model_dump_json()

                ## DEFAULT TTL ##
                if self.ttl is not None:
                    kwargs["ttl"] = self.ttl
                ## Get Cache-Controls ##
                _cache_kwargs = kwargs.get("cache", None)
                if isinstance(_cache_kwargs, dict):
                    for k, v in _cache_kwargs.items():
                        if k == "ttl":
                            kwargs["ttl"] = v

                cached_data = {"timestamp": time.time(), "response": result}
                return cache_key, cached_data, kwargs
            else:
                raise Exception("cache key is None")
        except Exception as e:
            raise e

    def add_cache(self, result, *args, **kwargs):
        """
        Adds a result to the cache.

        Args:
            *args: args to litellm.completion() or embedding()
            **kwargs: kwargs to litellm.completion() or embedding()

        Returns:
            None
        """
        try:
            if self.should_use_cache(*args, **kwargs) is not True:
                return
            cache_key, cached_data, kwargs = self._add_cache_logic(
                result=result, *args, **kwargs
            )
            self.cache.set_cache(cache_key, cached_data, **kwargs)
        except Exception as e:
            verbose_logger.exception(f"LiteLLM Cache: Excepton add_cache: {str(e)}")

    async def async_add_cache(self, result, *args, **kwargs):
        """
        Async implementation of add_cache
        """
        try:
            if self.should_use_cache(*args, **kwargs) is not True:
                return
            if self.type == "redis" and self.redis_flush_size is not None:
                # high traffic - fill in results in memory and then flush
                await self.batch_cache_write(result, *args, **kwargs)
            else:
                cache_key, cached_data, kwargs = self._add_cache_logic(
                    result=result, *args, **kwargs
                )
                await self.cache.async_set_cache(cache_key, cached_data, **kwargs)
        except Exception as e:
            verbose_logger.exception(f"LiteLLM Cache: Excepton add_cache: {str(e)}")

    async def async_add_cache_pipeline(self, result, *args, **kwargs):
        """
        Async implementation of add_cache for Embedding calls

        Does a bulk write, to prevent using too many clients
        """
        try:
            if self.should_use_cache(*args, **kwargs) is not True:
                return
            cache_list = []
            for idx, i in enumerate(kwargs["input"]):
                preset_cache_key = self.get_cache_key(*args, **{**kwargs, "input": i})
                kwargs["cache_key"] = preset_cache_key
                embedding_response = result.data[idx]
                cache_key, cached_data, kwargs = self._add_cache_logic(
                    result=embedding_response,
                    *args,
                    **kwargs,
                )
                cache_list.append((cache_key, cached_data))
            async_set_cache_pipeline = getattr(
                self.cache, "async_set_cache_pipeline", None
            )
            if async_set_cache_pipeline:
                await async_set_cache_pipeline(cache_list=cache_list)
            else:
                tasks = []
                for val in cache_list:
                    tasks.append(self.cache.async_set_cache(val[0], val[1], **kwargs))
                await asyncio.gather(*tasks)
        except Exception as e:
            verbose_logger.exception(f"LiteLLM Cache: Excepton add_cache: {str(e)}")

    def should_use_cache(self, *args, **kwargs):
        """
        Returns true if we should use the cache for LLM API calls

        If cache is default_on then this is True
        If cache is default_off then this is only true when user has opted in to use cache
        """
        if self.mode == CacheMode.default_on:
            return True

        # when mode == default_off -> Cache is opt in only
        _cache = kwargs.get("cache", None)
        verbose_logger.debug("should_use_cache: kwargs: %s; _cache: %s", kwargs, _cache)
        if _cache and isinstance(_cache, dict):
            if _cache.get("use-cache", False) is True:
                return True
        return False

    async def batch_cache_write(self, result, *args, **kwargs):
        cache_key, cached_data, kwargs = self._add_cache_logic(
            result=result, *args, **kwargs
        )
        await self.cache.batch_cache_write(cache_key, cached_data, **kwargs)

    async def ping(self):
        cache_ping = getattr(self.cache, "ping")
        if cache_ping:
            return await cache_ping()
        return None

    async def delete_cache_keys(self, keys):
        cache_delete_cache_keys = getattr(self.cache, "delete_cache_keys")
        if cache_delete_cache_keys:
            return await cache_delete_cache_keys(keys)
        return None

    async def disconnect(self):
        if hasattr(self.cache, "disconnect"):
            await self.cache.disconnect()


class DiskCache(BaseCache):
    def __init__(self, disk_cache_dir: Optional[str] = None):
        import diskcache as dc

        # if users don't provider one, use the default litellm cache
        if disk_cache_dir is None:
            self.disk_cache = dc.Cache(".litellm_cache")
        else:
            self.disk_cache = dc.Cache(disk_cache_dir)

    def set_cache(self, key, value, **kwargs):
        print_verbose("DiskCache: set_cache")
        if "ttl" in kwargs:
            self.disk_cache.set(key, value, expire=kwargs["ttl"])
        else:
            self.disk_cache.set(key, value)

    async def async_set_cache(self, key, value, **kwargs):
        self.set_cache(key=key, value=value, **kwargs)

    async def async_set_cache_pipeline(self, cache_list, ttl=None):
        for cache_key, cache_value in cache_list:
            if ttl is not None:
                self.set_cache(key=cache_key, value=cache_value, ttl=ttl)
            else:
                self.set_cache(key=cache_key, value=cache_value)

    def get_cache(self, key, **kwargs):
        original_cached_response = self.disk_cache.get(key)
        if original_cached_response:
            try:
                cached_response = json.loads(original_cached_response)  # type: ignore
            except Exception:
                cached_response = original_cached_response
            return cached_response
        return None

    def batch_get_cache(self, keys: list, **kwargs):
        return_val = []
        for k in keys:
            val = self.get_cache(key=k, **kwargs)
            return_val.append(val)
        return return_val

    def increment_cache(self, key, value: int, **kwargs) -> int:
        # get the value
        init_value = self.get_cache(key=key) or 0
        value = init_value + value  # type: ignore
        self.set_cache(key, value, **kwargs)
        return value

    async def async_get_cache(self, key, **kwargs):
        return self.get_cache(key=key, **kwargs)

    async def async_batch_get_cache(self, keys: list, **kwargs):
        return_val = []
        for k in keys:
            val = self.get_cache(key=k, **kwargs)
            return_val.append(val)
        return return_val

    async def async_increment(self, key, value: int, **kwargs) -> int:
        # get the value
        init_value = await self.async_get_cache(key=key) or 0
        value = init_value + value  # type: ignore
        await self.async_set_cache(key, value, **kwargs)
        return value

    def flush_cache(self):
        self.disk_cache.clear()

    async def disconnect(self):
        pass

    def delete_cache(self, key):
        self.disk_cache.pop(key)


def enable_cache(
    type: Optional[Literal["local", "redis", "s3", "disk"]] = "local",
    host: Optional[str] = None,
    port: Optional[str] = None,
    password: Optional[str] = None,
    supported_call_types: Optional[
        List[
            Literal[
                "completion",
                "acompletion",
                "embedding",
                "aembedding",
                "atranscription",
                "transcription",
                "atext_completion",
                "text_completion",
            ]
        ]
    ] = [
        "completion",
        "acompletion",
        "embedding",
        "aembedding",
        "atranscription",
        "transcription",
        "atext_completion",
        "text_completion",
    ],
    **kwargs,
):
    """
    Enable cache with the specified configuration.

    Args:
        type (Optional[Literal["local", "redis", "s3", "disk"]]): The type of cache to enable. Defaults to "local".
        host (Optional[str]): The host address of the cache server. Defaults to None.
        port (Optional[str]): The port number of the cache server. Defaults to None.
        password (Optional[str]): The password for the cache server. Defaults to None.
        supported_call_types (Optional[List[Literal["completion", "acompletion", "embedding", "aembedding"]]]):
            The supported call types for the cache. Defaults to ["completion", "acompletion", "embedding", "aembedding"].
        **kwargs: Additional keyword arguments.

    Returns:
        None

    Raises:
        None
    """
    print_verbose("LiteLLM: Enabling Cache")
    if "cache" not in litellm.input_callback:
        litellm.input_callback.append("cache")
    if "cache" not in litellm.success_callback:
        litellm.success_callback.append("cache")
    if "cache" not in litellm._async_success_callback:
        litellm._async_success_callback.append("cache")

    if litellm.cache is None:
        litellm.cache = Cache(
            type=type,
            host=host,
            port=port,
            password=password,
            supported_call_types=supported_call_types,
            **kwargs,
        )
    print_verbose(f"LiteLLM: Cache enabled, litellm.cache={litellm.cache}")
    print_verbose(f"LiteLLM Cache: {vars(litellm.cache)}")


def update_cache(
    type: Optional[Literal["local", "redis", "s3", "disk"]] = "local",
    host: Optional[str] = None,
    port: Optional[str] = None,
    password: Optional[str] = None,
    supported_call_types: Optional[
        List[
            Literal[
                "completion",
                "acompletion",
                "embedding",
                "aembedding",
                "atranscription",
                "transcription",
                "atext_completion",
                "text_completion",
            ]
        ]
    ] = [
        "completion",
        "acompletion",
        "embedding",
        "aembedding",
        "atranscription",
        "transcription",
        "atext_completion",
        "text_completion",
    ],
    **kwargs,
):
    """
    Update the cache for LiteLLM.

    Args:
        type (Optional[Literal["local", "redis", "s3", "disk"]]): The type of cache. Defaults to "local".
        host (Optional[str]): The host of the cache. Defaults to None.
        port (Optional[str]): The port of the cache. Defaults to None.
        password (Optional[str]): The password for the cache. Defaults to None.
        supported_call_types (Optional[List[Literal["completion", "acompletion", "embedding", "aembedding"]]]):
            The supported call types for the cache. Defaults to ["completion", "acompletion", "embedding", "aembedding"].
        **kwargs: Additional keyword arguments for the cache.

    Returns:
        None

    """
    print_verbose("LiteLLM: Updating Cache")
    litellm.cache = Cache(
        type=type,
        host=host,
        port=port,
        password=password,
        supported_call_types=supported_call_types,
        **kwargs,
    )
    print_verbose(f"LiteLLM: Cache Updated, litellm.cache={litellm.cache}")
    print_verbose(f"LiteLLM Cache: {vars(litellm.cache)}")


def disable_cache():
    """
    Disable the cache used by LiteLLM.

    This function disables the cache used by the LiteLLM module. It removes the cache-related callbacks from the input_callback, success_callback, and _async_success_callback lists. It also sets the litellm.cache attribute to None.

    Parameters:
    None

    Returns:
    None
    """
    from contextlib import suppress

    print_verbose("LiteLLM: Disabling Cache")
    with suppress(ValueError):
        litellm.input_callback.remove("cache")
        litellm.success_callback.remove("cache")
        litellm._async_success_callback.remove("cache")

    litellm.cache = None
    print_verbose(f"LiteLLM: Cache disabled, litellm.cache={litellm.cache}")
