from __future__ import annotations

from collections.abc import Awaitable, Sequence
from typing import Final, Protocol, cast

from typing_extensions import assert_never

from litellm.rust_bridge.bindings import NativeBinding, native_exception_types
from litellm.rust_bridge.catalog import CacheContext, Rules, decision
from litellm.rust_bridge.configuration import Decision


class CacheFacade(Protocol):
    @property
    def type(self) -> object: ...


class NativeResponseCacheRuntime(Protocol):
    """The native runtime `Cache` binds: `_ResponseCacheRuntime` activated for one facade."""

    @property
    def kind(self) -> str: ...

    def lookup(self, request: object) -> object: ...
    def lookup_semantic(self, request: object) -> tuple[object, float | None]: ...
    def store(self, request: object, response: object) -> None: ...
    def lookup_batch(self, requests: Sequence[object]) -> object: ...
    def async_lookup(self, request: object) -> Awaitable[object]: ...
    def async_lookup_semantic(self, request: object) -> Awaitable[tuple[object, float | None]]: ...
    def async_store(self, request: object, response: object) -> Awaitable[None]: ...
    def async_lookup_batch(self, requests: Sequence[object]) -> Awaitable[object]: ...
    def async_store_batch(self, requests: Sequence[object], responses: Sequence[object]) -> Awaitable[object]: ...
    def async_flush(self) -> Awaitable[None]: ...
    def ping(self) -> Awaitable[object]: ...


class NativeResponseCacheRuntimeFactory(Protocol):
    @staticmethod
    def from_cache(cache: CacheFacade) -> NativeResponseCacheRuntime: ...


def _runtime_factory(value: object) -> NativeResponseCacheRuntimeFactory | None:
    return cast(NativeResponseCacheRuntimeFactory, value) if callable(getattr(value, "from_cache", None)) else None


_RUNTIME: Final = NativeBinding("_ResponseCacheRuntime", validate=_runtime_factory)


def resolve_response_cache(
    cache: CacheFacade,
    rules: Rules | None = None,
) -> NativeResponseCacheRuntime | None:
    """The native runtime for `cache`'s storage object, or `None` when the catalog keeps it on Python.

    `Cache.__init__` and `Cache.cache` assignment call this for every facade, so the rollout
    policy stays in Python while the facade itself is native.
    """
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
