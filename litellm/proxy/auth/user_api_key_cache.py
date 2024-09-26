"""
class UserAPIKeyCache used in user_api_key_auth.py

Used to store the virtual key, team, user, end user in cache 

manages reading / writing to in memory and redis cache
"""

from typing import TYPE_CHECKING, Any, Union

from litellm.caching import DualCache

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    Span = _Span
else:
    Span = Any


class UserAPIKeyCache:
    def __init__(self, dual_cache: DualCache):
        self.dual_cache: DualCache = dual_cache

    async def async_get_cache(
        self,
        key,
        litellm_parent_otel_span: Union[Span, None],
        local_only: bool = False,
        **kwargs,
    ) -> Any:
        return await self.dual_cache.async_get_cache(
            key=key,
            local_only=local_only,
            litellm_parent_otel_span=litellm_parent_otel_span,
            **kwargs,
        )

    async def async_set_cache(
        self,
        key,
        value,
        litellm_parent_otel_span: Union[Span, None],
        local_only: bool = False,
        **kwargs,
    ) -> None:
        return await self.dual_cache.async_set_cache(
            key=key,
            value=value,
            local_only=local_only,
            litellm_parent_otel_span=litellm_parent_otel_span,
            **kwargs,
        )

    async def async_increment_cache(
        self,
        key,
        value: float,
        litellm_parent_otel_span: Union[Span, None],
        local_only: bool = False,
        **kwargs,
    ):
        return await self.dual_cache.async_increment_cache(
            key=key,
            value=value,
            local_only=local_only,
            litellm_parent_otel_span=litellm_parent_otel_span,
            **kwargs,
        )

    def set_cache(
        self,
        key,
        value,
        local_only: bool = False,
        **kwargs,
    ) -> None:
        return self.dual_cache.set_cache(
            key=key,
            value=value,
            local_only=local_only,
            **kwargs,
        )

    def get_cache(
        self,
        key,
        local_only: bool = False,
        **kwargs,
    ) -> Any:
        return self.dual_cache.get_cache(
            key=key,
            local_only=local_only,
            **kwargs,
        )
