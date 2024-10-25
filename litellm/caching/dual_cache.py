"""
Dual Cache implementation - Class to update both Redis and an in-memory cache simultaneously.

Has 4 primary methods:
    - set_cache
    - get_cache
    - async_set_cache
    - async_get_cache
"""

import traceback
from typing import List, Optional

import litellm
from litellm._logging import print_verbose, verbose_logger

from .base_cache import BaseCache
from .in_memory_cache import InMemoryCache
from .redis_cache import RedisCache


class DualCache(BaseCache):
    """
    DualCache is a cache implementation that updates both Redis and an in-memory cache simultaneously.
    When data is updated or inserted, it is written to both the in-memory cache + Redis.
    This ensures that even if Redis hasn't been updated yet, the in-memory cache reflects the most recent data.
    """

    def __init__(
        self,
        in_memory_cache: Optional[InMemoryCache] = None,
        redis_cache: Optional[RedisCache] = None,
        default_in_memory_ttl: Optional[float] = None,
        default_redis_ttl: Optional[float] = None,
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

            if result is None and self.redis_cache is not None and local_only is False:
                # If not found in in-memory cache, try fetching from Redis
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
