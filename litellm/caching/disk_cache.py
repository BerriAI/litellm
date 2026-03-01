import json
from typing import TYPE_CHECKING, Any, Optional, Union

from .base_cache import BaseCache

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    Span = Union[_Span, Any]
else:
    Span = Any


class DiskCache(BaseCache):
    def __init__(self, disk_cache_dir: Optional[str] = None):
        try:
            import diskcache as dc
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Please install litellm with `litellm[caching]` to use disk caching."
            ) from e

        # if users don't provider one, use the default litellm cache
        if disk_cache_dir is None:
            self.disk_cache = dc.Cache(".litellm_cache")
        else:
            self.disk_cache = dc.Cache(disk_cache_dir)

    def set_cache(self, key, value, **kwargs):
        if "ttl" in kwargs:
            self.disk_cache.set(key, value, expire=kwargs["ttl"])
        else:
            self.disk_cache.set(key, value)

    async def async_set_cache(self, key, value, **kwargs):
        self.set_cache(key=key, value=value, **kwargs)

    async def async_set_cache_pipeline(self, cache_list, **kwargs):
        for cache_key, cache_value in cache_list:
            if "ttl" in kwargs:
                self.set_cache(key=cache_key, value=cache_value, ttl=kwargs["ttl"])
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
        with self.disk_cache.transact():
            init_value = self.disk_cache.get(key, default=0)

            if isinstance(init_value, (str, bytes, bytearray)):
                try:
                    parsed_value = json.loads(init_value)  # type: ignore[arg-type]
                except Exception:
                    parsed_value = init_value
            else:
                parsed_value = init_value

            if parsed_value is None:
                parsed_value = 0

            new_value = parsed_value + value  # type: ignore[operator]
            self.set_cache(key, new_value, **kwargs)
            return new_value

    async def async_get_cache(self, key, **kwargs):
        return self.get_cache(key=key, **kwargs)

    async def async_batch_get_cache(self, keys: list, **kwargs):
        return_val = []
        for k in keys:
            val = self.get_cache(key=k, **kwargs)
            return_val.append(val)
        return return_val

    async def async_increment(self, key, value: int, **kwargs) -> int:
        return self.increment_cache(key=key, value=value, **kwargs)

    def flush_cache(self):
        self.disk_cache.clear()

    async def disconnect(self):
        pass

    def delete_cache(self, key):
        self.disk_cache.pop(key)
