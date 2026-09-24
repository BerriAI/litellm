from __future__ import annotations

import math
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Protocol, TypeVar, cast

from typing_extensions import ReadOnly, Required, TypedDict, assert_never

from litellm.rust_bridge.bindings import NativeBinding, native_exception_types
from litellm.rust_bridge.catalog import CacheContext, CacheFacadeContext, Rules, decision
from litellm.rust_bridge.configuration import Decision


class CacheBackendOwner(Protocol):
    @property
    def type(self) -> object: ...


class CacheFacade(CacheBackendOwner, Protocol):
    @property
    def ttl(self) -> float | None: ...

    @property
    def semantic_cache_scope(self) -> str: ...

    def get_cache_key(self, **kwargs: object) -> str: ...  # kwargs-ok: mirrors the legacy cache facade contract


class NativeCacheKey(TypedDict):
    preset: ReadOnly[str]


class NativeCacheRequest(TypedDict, total=False):
    key: Required[ReadOnly[NativeCacheKey]]
    ttl_seconds: ReadOnly[float | None]
    max_age_seconds: ReadOnly[float | None]
    messages: ReadOnly[object | None]
    input: ReadOnly[object | None]
    metadata: ReadOnly[object | None]
    litellm_metadata: ReadOnly[object | None]
    litellm_params: ReadOnly[object | None]
    scope: ReadOnly[str]


class NativeResponseCacheRuntime(Protocol):
    @property
    def kind(self) -> str: ...

    def lookup(self, request: NativeCacheRequest) -> object: ...
    def lookup_semantic(self, request: NativeCacheRequest) -> tuple[object, float | None]: ...
    def store(self, request: NativeCacheRequest, response: object) -> None: ...
    def lookup_batch(self, requests: Sequence[NativeCacheRequest]) -> object: ...
    def async_lookup(self, request: NativeCacheRequest) -> Awaitable[object]: ...
    def async_lookup_semantic(self, request: NativeCacheRequest) -> Awaitable[tuple[object, float | None]]: ...
    def async_store(self, request: NativeCacheRequest, response: object) -> Awaitable[None]: ...
    def async_lookup_batch(self, requests: Sequence[NativeCacheRequest]) -> Awaitable[object]: ...
    def async_store_batch(
        self,
        requests: Sequence[NativeCacheRequest],
        responses: Sequence[object],
    ) -> Awaitable[object]: ...
    def async_flush(self) -> Awaitable[None]: ...
    def ping(self) -> Awaitable[object]: ...


class NativeResponseCacheRuntimeFactory(Protocol):
    @staticmethod
    def from_cache(cache: CacheBackendOwner) -> NativeResponseCacheRuntime: ...


def _runtime_factory(value: object) -> NativeResponseCacheRuntimeFactory | None:
    return cast(NativeResponseCacheRuntimeFactory, value) if callable(getattr(value, "from_cache", None)) else None


_RUNTIME: Final = NativeBinding("_ResponseCacheRuntime", validate=_runtime_factory)

FacadeT = TypeVar("FacadeT")


def _facade_class(value: object) -> type | None:
    return value if isinstance(value, type) else None


_FACADE: Final = NativeBinding("Cache", validate=_facade_class)


def select_cache_facade(
    python_facade: type[FacadeT],
    rules: Rules | None = None,
    native_facade: NativeBinding[type] = _FACADE,
) -> type[FacadeT]:
    """The class `litellm.caching.caching.Cache` names: the Python facade, or the native one the catalog selects."""
    selected: Final = decision(CacheFacadeContext(), rules)
    match selected:
        case Decision.PYTHON:
            return python_facade
        case Decision.RUST_WITH_FALLBACK | Decision.RUST_REQUIRED:
            native: Final = native_facade.load()
            if native is not None:
                return cast(type[FacadeT], native)  # cast-ok: the native facade implements the Python facade's API
            if selected is Decision.RUST_REQUIRED:
                raise RuntimeError("Rust cache facade is unavailable")
            return python_facade
        case _:
            assert_never(selected)


@dataclass(frozen=True, slots=True)
class ResponseCacheRuntime:
    native: NativeResponseCacheRuntime

    @property
    def kind(self) -> str:
        return self.native.kind

    def request(self, cache: CacheFacade, kwargs: Mapping[str, object]) -> NativeCacheRequest | None:
        key_value: Final = kwargs.get("cache_key")
        key: Final = key_value if isinstance(key_value, str) else cache.get_cache_key(**dict(kwargs))
        if not key:
            return None
        control_value: Final = kwargs.get("cache")
        control: Final = _string_mapping(control_value)
        configured_ttl: Final = cache.ttl if cache.ttl is not None else _duration(kwargs.get("ttl"))
        control_ttl: Final = _duration(control.get("ttl"))
        current_max_age: Final = _duration(control.get("s-max-age"))
        legacy_max_age: Final = _duration(control.get("s-maxage"))
        ttl: Final = configured_ttl if control_ttl is None else control_ttl
        max_age: Final = legacy_max_age if current_max_age is None else current_max_age
        return NativeCacheRequest(
            key=NativeCacheKey(preset=key),
            ttl_seconds=ttl,
            max_age_seconds=max_age,
            messages=kwargs.get("messages"),
            input=kwargs.get("input"),
            metadata=kwargs.get("metadata"),
            litellm_metadata=kwargs.get("litellm_metadata"),
            litellm_params=kwargs.get("litellm_params"),
            scope=cache.semantic_cache_scope,
        )

    def lookup(self, request: NativeCacheRequest) -> object:
        return self.native.lookup(request)

    def lookup_semantic(self, request: NativeCacheRequest) -> tuple[object, float | None]:
        """The cached response and the similarity a semantic backend reports, if any."""
        response, similarity = self.native.lookup_semantic(request)
        return response, similarity

    def store(self, request: NativeCacheRequest, response: object) -> None:
        self.native.store(request, response)

    def lookup_batch(self, requests: Sequence[NativeCacheRequest]) -> object:
        return self.native.lookup_batch(requests)

    async def async_lookup(self, request: NativeCacheRequest) -> object:
        return await self.native.async_lookup(request)

    async def async_lookup_semantic(self, request: NativeCacheRequest) -> tuple[object, float | None]:
        response, similarity = await self.native.async_lookup_semantic(request)
        return response, similarity

    async def async_store(self, request: NativeCacheRequest, response: object) -> None:
        await self.native.async_store(request, response)

    async def async_lookup_batch(self, requests: Sequence[NativeCacheRequest]) -> object:
        return await self.native.async_lookup_batch(requests)

    async def async_store_batch(
        self,
        requests: Sequence[NativeCacheRequest],
        responses: Sequence[object],
    ) -> object:
        return await self.native.async_store_batch(requests, responses)

    async def ping(self) -> object:
        return await self.native.ping()

    async def async_flush(self) -> None:
        await self.native.async_flush()


def resolve_response_cache(
    cache: CacheFacade,
    rules: Rules | None = None,
) -> ResponseCacheRuntime | None:
    native: Final = resolve_native_runtime(cache, rules)
    return None if native is None else ResponseCacheRuntime(native)


def resolve_native_runtime(
    cache: CacheBackendOwner,
    rules: Rules | None = None,
) -> NativeResponseCacheRuntime | None:
    """The native runtime for `cache`'s storage object, or `None` when the catalog keeps it on Python."""
    backend_value: Final = cache.type
    backend: Final = str.__str__(backend_value) if isinstance(backend_value, str) else str(backend_value)
    selected: Final = decision(CacheContext(backend=backend), rules)
    match selected:
        case Decision.PYTHON:
            return None
        case Decision.RUST_WITH_FALLBACK | Decision.RUST_REQUIRED:
            factory: Final = _RUNTIME.load()
            if factory is None:
                if selected is Decision.RUST_REQUIRED:
                    raise RuntimeError("Rust response cache runtime is unavailable")
                return None
            try:
                return factory.from_cache(cache)
            except Exception as error:
                exceptions: Final = native_exception_types()
                if exceptions is None or not isinstance(error, exceptions[0]):
                    raise
                if selected is Decision.RUST_REQUIRED:
                    raise RuntimeError(f"Rust response cache runtime declined the cache: {error}") from error
                return None
        case _:
            assert_never(selected)


def _duration(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    duration: Final = float(value)
    return duration if math.isfinite(duration) and duration >= 0 else None


def _string_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        return {}
    source: Final = cast(Mapping[object, object], value)
    return {key: item for key, item in source.items() if isinstance(key, str)}
