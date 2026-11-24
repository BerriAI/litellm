"""
Base Cache implementation. All cache implementations should inherit from this class.

Has 4 methods:
    - set_cache
    - get_cache
    - async_set_cache
    - async_get_cache
"""

from abc import ABC, abstractmethod
from typing import Final


class BaseCache(ABC):
    def __init__(self, default_ttl: int = 60):
        self.default_ttl = default_ttl

    def get_ttl(self, **kwargs) -> int | None:
        kwargs_ttl: Final[int | None] = kwargs.get("ttl")
        if kwargs_ttl is not None:
            try:
                return int(kwargs_ttl)
            except ValueError:
                return self.default_ttl
        return self.default_ttl

    def set_cache(self, key, value: object, **kwargs) -> None:
        raise NotImplementedError

    async def async_set_cache(self, key, value: object, **kwargs) -> object:
        raise NotImplementedError

    @abstractmethod
    async def async_set_cache_pipeline(self, cache_list: list[tuple[str, object]], **kwargs: object) -> None:
        pass

    def get_cache(self, key, **kwargs) -> object:
        raise NotImplementedError

    async def async_get_cache(self, key, **kwargs) -> object:
        raise NotImplementedError

    async def batch_cache_write(self, key, value: object, **kwargs: object) -> None:
        raise NotImplementedError

    async def disconnect(self) -> None:
        raise NotImplementedError

    async def test_connection(self) -> dict:
        """
        Test the cache connection.

        Returns:
            dict: {"status": "success" | "failed", "message": str, "error": Optional[str]}
        """
        raise NotImplementedError
