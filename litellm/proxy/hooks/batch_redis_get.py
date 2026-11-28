# What this does?
## Gets a key's redis cache, and store it in memory for 1 minute.
## This reduces the number of REDIS GET requests made during high-traffic by the proxy.
### [BETA] this is in Beta. And might change.

import traceback
from typing import Final, Literal

from fastapi import HTTPException

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache, InMemoryCache, RedisCache, response_cache_phase
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.semantic_cache import semantic_cache_policy


class _PROXY_BatchRedisRequests(CustomLogger):
    # Class variables or attributes
    in_memory_cache: InMemoryCache | None = None

    def __init__(self):
        if litellm.cache is not None:
            litellm.cache.async_get_cache = (
                self.async_get_cache
            )  # map the litellm 'get_cache' function to our custom function

    def print_verbose(self, print_statement: object, debug_level: Literal["INFO", "DEBUG"] = "DEBUG") -> None:
        if debug_level == "DEBUG" or debug_level == "INFO":
            verbose_proxy_logger.debug(print_statement)
        if litellm.set_verbose is True:
            print(print_statement)  # noqa: T201

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
    ) -> None:
        try:
            """
            Get the user key

            Check if a key starting with `litellm:<api_key>:<call_type:` exists in-memory

            If no, then get relevant cache from redis
            """
            api_key: Final = user_api_key_dict.api_key

            cache_key_name: Final = f"litellm:{api_key}:{call_type}"
            self.in_memory_cache = cache.in_memory_cache

            key_value_dict = {}
            in_memory_cache_exists = False
            for key in cache.in_memory_cache.cache_dict:
                if isinstance(key, str) and key.startswith(cache_key_name):
                    in_memory_cache_exists = True

            response_cache: Final = litellm.cache
            request_fields: Final = as_str_mapping(data) or {}
            if not in_memory_cache_exists and response_cache is not None:
                """
                - Check if `litellm.Cache` is redis
                - Get the relevant values
                """
                allow_read: Final = semantic_cache_policy(request_fields) is None or response_cache.should_use_cache(
                    **request_fields
                )
                if allow_read and isinstance(response_cache.cache, RedisCache):
                    self.print_verbose(f"cache_key_name: {cache_key_name}")
                    with response_cache_phase("get"):
                        keys = await response_cache.cache.async_scan_iter(pattern=cache_key_name, count=100)
                        self.print_verbose(f"redis keys: {keys}")
                        if len(keys) > 0:
                            key_value_dict = await response_cache.cache.async_batch_get_cache(key_list=keys)

            ## Add to cache
            if len(key_value_dict.items()) > 0:
                await cache.in_memory_cache.async_set_cache_pipeline(cache_list=list(key_value_dict.items()), ttl=60)
            ## Set cache namespace if it's a miss
            data["metadata"]["redis_namespace"] = cache_key_name
        except HTTPException as e:
            raise e
        except Exception as e:
            verbose_proxy_logger.error(
                "litellm.proxy.hooks.batch_redis_get.py::async_pre_call_hook(): Exception occured - %s", e
            )
            verbose_proxy_logger.debug(traceback.format_exc())

    async def async_get_cache(self, *args: object, **kwargs: object) -> object | None:
        """
        - Check if the cache key is in-memory

        - Else:
            - add missing cache key from REDIS
            - update in-memory cache
            - return redis cache request
        """
        try:  # never block execution
            response_cache: Final = litellm.cache
            if response_cache is None:
                return None
            if semantic_cache_policy(kwargs) is not None and not response_cache.should_use_cache(**kwargs):
                return None
            cache_key: Final = kwargs["cache_key"] if "cache_key" in kwargs else response_cache.get_cache_key(**kwargs)
            worker_cache: Final = self.in_memory_cache
            if not isinstance(cache_key, str) or worker_cache is None:
                return None
            max_age: Final = response_cache.response_cache_max_age(kwargs)
            local_entry: Final = worker_cache.get_cache(cache_key, *args, **kwargs)
            local_response: Final = response_cache.get_cache_logic(local_entry, max_age)
            if local_response is not None:
                return local_response
            with response_cache_phase("get"):
                redis_entry: Final = await response_cache.cache.async_get_cache(cache_key, *args, **kwargs)
            redis_response: Final = response_cache.get_cache_logic(redis_entry, max_age)
            if redis_response is not None:
                await worker_cache.async_set_cache(
                    cache_key, redis_entry, ttl=60, replace_ttl=True, preserve_newer_response=True
                )
        except Exception:
            return None
        else:
            return redis_response


PROXY_BatchRedisRequests: Final = _PROXY_BatchRedisRequests
