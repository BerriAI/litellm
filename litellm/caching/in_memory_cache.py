"""
In-Memory Cache implementation

Has 4 methods:
    - set_cache
    - get_cache
    - async_set_cache
    - async_get_cache
"""

import json
import time
from typing import List, Optional

from .base_cache import BaseCache


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
