"""
Base Cache implementation. All cache implementations should inherit from this class.

Has 4 methods:
    - set_cache
    - get_cache
    - async_set_cache
    - async_get_cache
"""

from typing import Optional


class BaseCache:
    def __init__(self, default_ttl: int = 60):
        self.default_ttl = default_ttl

    def get_ttl(self, **kwargs) -> Optional[int]:
        if kwargs.get("ttl") is not None:
            return kwargs.get("ttl")
        return self.default_ttl

    def set_cache(self, key, value, **kwargs):
        raise NotImplementedError

    async def async_set_cache(self, key, value, **kwargs):
        raise NotImplementedError

    def get_cache(self, key, **kwargs):
        raise NotImplementedError

    async def async_get_cache(self, key, **kwargs):
        raise NotImplementedError

    async def batch_cache_write(self, key, value, **kwargs):
        raise NotImplementedError

    async def disconnect(self):
        raise NotImplementedError
