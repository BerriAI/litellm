from __future__ import annotations

from collections.abc import Awaitable, Sequence
from typing import Final, Protocol, TypeVar, cast

from typing_extensions import ReadOnly, Required, TypedDict, assert_never

from litellm.rust_bridge.bindings import NativeBinding, native_exception_types
from litellm.rust_bridge.catalog import CacheContext, Rules, decision
from litellm.rust_bridge.configuration import Decision


class CacheBackendOwner(Protocol):
    @property
    def type(self) -> object: ...


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
    selected: Final = decision(CacheContext(), rules)
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


def resolve_native_runtime(
    cache: CacheBackendOwner,
    rules: Rules | None = None,
) -> NativeResponseCacheRuntime | None:
    """The native runtime for `cache`'s storage object, or `None` when it declines and the catalog allows Python."""
    factory: Final = _RUNTIME.load()
    if factory is None:
        return None
    try:
        return factory.from_cache(cache)
    except Exception as error:
        exceptions: Final = native_exception_types()
        if exceptions is None or not isinstance(error, exceptions[0]):
            raise
        if decision(CacheContext(), rules) is Decision.RUST_REQUIRED:
            raise RuntimeError(f"Rust response cache runtime declined the cache: {error}") from error
        return None
