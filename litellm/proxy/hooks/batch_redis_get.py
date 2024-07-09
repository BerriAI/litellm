# What this does?
## Gets a key's redis cache, and store it in memory for 1 minute.
## This reduces the number of REDIS GET requests made during high-traffic by the proxy.
### [BETA] this is in Beta. And might change.

from typing import Optional, Literal
import litellm
from litellm.caching import DualCache, RedisCache, InMemoryCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.integrations.custom_logger import CustomLogger
from litellm._logging import verbose_proxy_logger
from fastapi import HTTPException
import json, traceback


class _PROXY_BatchRedisRequests(CustomLogger):
    # Class variables or attributes
    in_memory_cache: Optional[InMemoryCache] = None

    def __init__(self):
        litellm.cache.async_get_cache = (
            self.async_get_cache
        )  # map the litellm 'get_cache' function to our custom function

    def print_verbose(
        self, print_statement, debug_level: Literal["INFO", "DEBUG"] = "DEBUG"
    ):
        if debug_level == "DEBUG":
            verbose_proxy_logger.debug(print_statement)
        elif debug_level == "INFO":
            verbose_proxy_logger.debug(print_statement)
        if litellm.set_verbose is True:
            print(print_statement)  # noqa

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
    ):
        try:
            """
            Get the user key

            Check if a key starting with `litellm:<api_key>:<call_type:` exists in-memory

            If no, then get relevant cache from redis
            """
            api_key = user_api_key_dict.api_key

            cache_key_name = f"litellm:{api_key}:{call_type}"
            self.in_memory_cache = cache.in_memory_cache

            key_value_dict = {}
            in_memory_cache_exists = False
            for key in cache.in_memory_cache.cache_dict.keys():
                if isinstance(key, str) and key.startswith(cache_key_name):
                    in_memory_cache_exists = True

            if in_memory_cache_exists == False and litellm.cache is not None:
                """
                - Check if `litellm.Cache` is redis
                - Get the relevant values
                """
                if litellm.cache.type is not None and isinstance(
                    litellm.cache.cache, RedisCache
                ):
                    # Initialize an empty list to store the keys
                    keys = []
                    self.print_verbose(f"cache_key_name: {cache_key_name}")
                    # Use the SCAN iterator to fetch keys matching the pattern
                    keys = await litellm.cache.cache.async_scan_iter(
                        pattern=cache_key_name, count=100
                    )
                    # If you need the truly "last" based on time or another criteria,
                    # ensure your key naming or storage strategy allows this determination
                    # Here you would sort or filter the keys as needed based on your strategy
                    self.print_verbose(f"redis keys: {keys}")
                    if len(keys) > 0:
                        key_value_dict = (
                            await litellm.cache.cache.async_batch_get_cache(
                                key_list=keys
                            )
                        )

            ## Add to cache
            if len(key_value_dict.items()) > 0:
                await cache.in_memory_cache.async_set_cache_pipeline(
                    cache_list=list(key_value_dict.items()), ttl=60
                )
            ## Set cache namespace if it's a miss
            data["metadata"]["redis_namespace"] = cache_key_name
        except HTTPException as e:
            raise e
        except Exception as e:
            verbose_proxy_logger.error(
                "litellm.proxy.hooks.batch_redis_get.py::async_pre_call_hook(): Exception occured - {}".format(
                    str(e)
                )
            )
            verbose_proxy_logger.debug(traceback.format_exc())

    async def async_get_cache(self, *args, **kwargs):
        """
        - Check if the cache key is in-memory

        - Else return None
        """
        try:  # never block execution
            if "cache_key" in kwargs:
                cache_key = kwargs["cache_key"]
            else:
                cache_key = litellm.cache.get_cache_key(
                    *args, **kwargs
                )  # returns "<cache_key_name>:<hash>" - we pass redis_namespace in async_pre_call_hook. Done to avoid rewriting the async_set_cache logic
            if cache_key is not None and self.in_memory_cache is not None:
                cache_control_args = kwargs.get("cache", {})
                max_age = cache_control_args.get(
                    "s-max-age", cache_control_args.get("s-maxage", float("inf"))
                )
                cached_result = self.in_memory_cache.get_cache(
                    cache_key, *args, **kwargs
                )
                return litellm.cache._get_cache_logic(
                    cached_result=cached_result, max_age=max_age
                )
        except Exception as e:
            return None
