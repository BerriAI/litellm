"""
Base Cache implementation. All cache implementations should inherit from this class.

Has 4 methods:
    - set_cache
    - get_cache
    - async_set_cache
    - async_get_cache
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

from litellm.types.caching import DynamicCacheControl

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    Span = _Span
else:
    Span = Any


class BaseCache(ABC):
    def __init__(
        self,
        default_ttl: Optional[int] = 60,
        max_allowed_ttl: Optional[int] = None,
    ):
        """
        Initialize the BaseCache

        Args:
            default_ttl: The default TTL for the cache
            max_allowed_ttl: The maximum allowed dynamic TTL for the cache
        """
        self.default_ttl = default_ttl
        self.max_allowed_ttl = max_allowed_ttl

    def get_ttl(self, **kwargs) -> Optional[int]:
        """
        Get the TTL to use for storing a LLM response in the cache
        """
        return self._get_dynamic_ttl(kwargs=kwargs) or self.default_ttl

    def set_cache(self, key, value, **kwargs):
        raise NotImplementedError

    async def async_set_cache(self, key, value, **kwargs):
        raise NotImplementedError

    @abstractmethod
    async def async_set_cache_pipeline(self, cache_list, **kwargs):
        pass

    def get_cache(self, key, **kwargs):
        raise NotImplementedError

    async def async_get_cache(self, key, **kwargs):
        raise NotImplementedError

    async def batch_cache_write(self, key, value, **kwargs):
        raise NotImplementedError

    async def disconnect(self):
        raise NotImplementedError

    def _get_dynamic_ttl(
        self,
        kwargs: dict,
    ) -> Optional[int]:
        """
        Returns the dynamic TTL to use for storing a LLM response in the cache

        - Checks if `cache` is set in kwargs. Uses cache: {ttl: <ttl>} to set the TTL
        - Checks if `ttl` is set in kwargs. Uses ttl: <ttl> to set the TTL
            - If ttl passed dynamically, it will be set to the minimum of the max allowed ttl and the ttl passed
        - Returns None if neither are set

        Allows a admin to set the maxmium allowed TTL for dynamic ttl requests
        """
        kwargs_ttl: Optional[int] = kwargs.get("ttl")
        dynamic_cache_control: Optional[DynamicCacheControl] = kwargs.get("cache")

        if dynamic_cache_control and dynamic_cache_control.get("ttl") is not None:
            kwargs_ttl = dynamic_cache_control.get("ttl")

        if kwargs_ttl is None:
            return None

        if self.max_allowed_ttl is None:
            return kwargs_ttl

        return min(kwargs_ttl, self.max_allowed_ttl)
