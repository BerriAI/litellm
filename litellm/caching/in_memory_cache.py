"""
In-Memory Cache implementation

Has 4 methods:
    - set_cache
    - get_cache
    - async_set_cache
    - async_get_cache
"""

import json
import sys
import time
import heapq
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from litellm.types.caching import RedisPipelineIncrementOperation

from pydantic import BaseModel

from litellm.constants import MAX_SIZE_PER_ITEM_IN_MEMORY_CACHE_IN_KB

from .base_cache import BaseCache


class InMemoryCache(BaseCache):
    def __init__(
        self,
        max_size_in_memory: Optional[int] = 200,
        default_ttl: Optional[
            int
        ] = 600,  # default ttl is 10 minutes. At maximum litellm rate limiting logic requires objects to be in memory for 1 minute
        max_size_per_item: Optional[int] = 1024,  # 1MB = 1024KB
    ):
        """
        max_size_in_memory [int]: Maximum number of items in cache. done to prevent memory leaks. Use 200 items as a default
        """
        self.max_size_in_memory = (
            max_size_in_memory if max_size_in_memory is not None else 200
        )  # set an upper bound of 200 items in-memory
        self.default_ttl = default_ttl or 600
        self.max_size_per_item = (
            max_size_per_item or MAX_SIZE_PER_ITEM_IN_MEMORY_CACHE_IN_KB
        )  # 1MB = 1024KB

        # in-memory cache
        self.cache_dict: dict = {}
        self.ttl_dict: dict = {}
        self.expiration_heap: list[tuple[float, str]] = []

    def check_value_size(self, value: Any):
        """
        Check if value size exceeds max_size_per_item (1MB)
        Returns True if value size is acceptable, False otherwise
        """
        try:
            # Fast path for common primitive types that are typically small
            if (
                isinstance(value, (bool, int, float, str))
                and len(str(value))
                < self.max_size_per_item * MAX_SIZE_PER_ITEM_IN_MEMORY_CACHE_IN_KB
            ):  # Conservative estimate
                return True

            # Direct size check for bytes objects
            if isinstance(value, bytes):
                return sys.getsizeof(value) / 1024 <= self.max_size_per_item

            # Handle special types without full conversion when possible
            if hasattr(value, "__sizeof__"):  # Use __sizeof__ if available
                size = value.__sizeof__() / 1024
                return size <= self.max_size_per_item

            # Fallback for complex types
            if isinstance(value, BaseModel) and hasattr(
                value, "model_dump"
            ):  # Pydantic v2
                value = value.model_dump()
            elif hasattr(value, "isoformat"):  # datetime objects
                return True  # datetime strings are always small

            # Only convert to JSON if absolutely necessary
            if not isinstance(value, (str, bytes)):
                value = json.dumps(value, default=str)

            return sys.getsizeof(value) / 1024 <= self.max_size_per_item

        except Exception:
            return False

    def _is_key_expired(self, key: str) -> bool:
        """
        Check if a specific key is expired
        """
        return key in self.ttl_dict and time.time() > self.ttl_dict[key]

    def _remove_key(self, key: str) -> None:
        """
        Remove a key from both cache_dict and ttl_dict
        """
        self.cache_dict.pop(key, None)
        self.ttl_dict.pop(key, None)

    def evict_cache(self):
        """
        Eviction policy:
        1. First, remove expired items from ttl_dict and cache_dict
        2. If cache is still at or above max_size_in_memory, evict items with earliest expiration times


        This guarantees the following:
        - 1. When item ttl not set: At minimum each item will remain in memory for the default ttl
        - 2. When ttl is set: the item will remain in memory for at least that amount of time, unless cache size requires eviction
        - 3. the size of in-memory cache is bounded

        """
        current_time = time.time()

        # Step 1: Remove expired or outdated items
        while self.expiration_heap:
            expiration_time, key = self.expiration_heap[0]

            # Case 1: Heap entry is outdated
            if expiration_time != self.ttl_dict.get(key):
                heapq.heappop(self.expiration_heap)
            # Case 2: Entry is valid but expired
            elif expiration_time <= current_time:
                heapq.heappop(self.expiration_heap)
                self._remove_key(key)
            else:
                # Case 3: Entry is valid and not expired
                break

        # Step 2: Evict if cache is still full
        while len(self.cache_dict) >= self.max_size_in_memory:
            expiration_time, key = heapq.heappop(self.expiration_heap)
            # Skip if key was removed or updated
            if self.ttl_dict.get(key) == expiration_time:
                self._remove_key(key)

        # de-reference the removed item
        # https://www.geeksforgeeks.org/diagnosing-and-fixing-memory-leaks-in-python/
        # One of the most common causes of memory leaks in Python is the retention of objects that are no longer being used.
        # This can occur when an object is referenced by another object, but the reference is never removed.

    def allow_ttl_override(self, key: str) -> bool:
        """
        Check if ttl is set for a key
        """
        ttl_time = self.ttl_dict.get(key)
        if ttl_time is None:  # if ttl is not set, allow override
            return True
        elif float(ttl_time) < time.time():  # if ttl is expired, allow override
            return True
        else:
            return False

    def set_cache(self, key, value, **kwargs):
        # Handle the edge case where max_size_in_memory is 0
        if self.max_size_in_memory == 0:
            return  # Don't cache anything if max size is 0

        if len(self.cache_dict) >= self.max_size_in_memory:
            # only evict when cache is full
            self.evict_cache()
        if not self.check_value_size(value):
            return

        self.cache_dict[key] = value
        if self.allow_ttl_override(key):  # if ttl is not set, set it to default ttl
            if "ttl" in kwargs and kwargs["ttl"] is not None:
                self.ttl_dict[key] = time.time() + float(kwargs["ttl"])
                heapq.heappush(self.expiration_heap, (self.ttl_dict[key], key))
            else:
                self.ttl_dict[key] = time.time() + self.default_ttl
                heapq.heappush(self.expiration_heap, (self.ttl_dict[key], key))

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

    def evict_element_if_expired(self, key: str) -> bool:
        """
        Returns True if the element is expired and removed from the cache

        Returns False if the element is not expired
        """
        if self._is_key_expired(key):
            self._remove_key(key)
            return True
        return False

    def get_cache(self, key, **kwargs):
        if key in self.cache_dict:
            if self.evict_element_if_expired(key):
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

    async def async_increment_pipeline(
        self, increment_list: List["RedisPipelineIncrementOperation"], **kwargs
    ) -> Optional[List[float]]:
        results = []
        for increment in increment_list:
            result = await self.async_increment(
                increment["key"], increment["increment_value"], **kwargs
            )
            results.append(result)
        return results

    def flush_cache(self):
        self.cache_dict.clear()
        self.ttl_dict.clear()
        self.expiration_heap.clear()

    async def disconnect(self):
        pass

    def delete_cache(self, key):
        self._remove_key(key)

    async def async_get_ttl(self, key: str) -> Optional[int]:
        """
        Get the remaining TTL of a key in in-memory cache
        """
        return self.ttl_dict.get(key, None)

    async def async_get_oldest_n_keys(self, n: int) -> List[str]:
        """
        Get the oldest n keys in the cache
        """
        # sorted ttl dict by ttl
        sorted_ttl_dict = sorted(self.ttl_dict.items(), key=lambda x: x[1])
        return [key for key, _ in sorted_ttl_dict[:n]]
