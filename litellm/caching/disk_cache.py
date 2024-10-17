import json
from typing import Optional

from litellm._logging import print_verbose

from .base_cache import BaseCache


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
