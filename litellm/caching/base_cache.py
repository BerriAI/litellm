"""
Base Cache implementation. All cache implementations should inherit from this class.

Has 4 methods:
    - set_cache
    - get_cache
    - async_set_cache
    - async_get_cache
"""


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
