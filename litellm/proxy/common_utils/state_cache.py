from __future__ import annotations

from typing import TYPE_CHECKING

from litellm.caching.dual_cache import DualCache

if TYPE_CHECKING:
    from collections.abc import Sequence

    from opentelemetry.trace import Span


class StateCache(DualCache):
    """
    A proxy cache for state other than auth objects: hook state, MCP state, JWKS and OIDC documents, UI settings and
    SSO login state. It is ``DualCache`` with its reads and writes typed, which ``DualCache`` leaves untyped.
    """

    def get_cache(
        self,
        key: str,
        parent_otel_span: Span | None = None,
        local_only: bool = False,
        **kwargs: object,  # kwargs-ok: DualCache's signature, which callers pass ttl through
    ) -> object:
        return super().get_cache(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # DualCache leaves its reads untyped
            key, parent_otel_span=parent_otel_span, local_only=local_only, **kwargs
        )

    async def async_get_cache(
        self,
        key: str,
        parent_otel_span: Span | None = None,
        local_only: bool = False,
        **kwargs: object,  # kwargs-ok: DualCache's signature, which callers pass ttl through
    ) -> object:
        return await super().async_get_cache(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # DualCache leaves its reads untyped
            key, parent_otel_span=parent_otel_span, local_only=local_only, **kwargs
        )

    def set_cache(
        self,
        key: str,
        value: object,
        local_only: bool = False,
        **kwargs: object,  # kwargs-ok: DualCache's signature, which callers pass ttl through
    ) -> None:
        super().set_cache(  # pyright: ignore[reportUnknownMemberType]  # DualCache leaves its writes untyped
            key, value, local_only=local_only, **kwargs
        )

    async def async_set_cache(
        self,
        key: str,
        value: object,
        local_only: bool = False,
        **kwargs: object,  # kwargs-ok: DualCache's signature, which callers pass ttl through
    ) -> None:
        await super().async_set_cache(  # pyright: ignore[reportUnknownMemberType]  # DualCache leaves its writes untyped
            key, value, local_only=local_only, **kwargs
        )

    async def async_set_cache_pipeline(
        self,
        cache_list: Sequence[tuple[str, object]],
        local_only: bool = False,
        **kwargs: object,  # kwargs-ok: DualCache's signature, which callers pass ttl through
    ) -> None:
        await super().async_set_cache_pipeline(  # pyright: ignore[reportUnknownMemberType]  # DualCache leaves its writes untyped
            cache_list, local_only=local_only, **kwargs
        )
