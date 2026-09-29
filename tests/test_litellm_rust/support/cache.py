from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Protocol, TypeAlias
from uuid import uuid4

from litellm.caching.caching import Cache
from litellm.rust_bridge import _native
from litellm.rust_bridge.response_cache import ResponseCacheRuntime


CacheRuntime: TypeAlias = _native._ResponseCacheRuntime  # pyright: ignore[reportPrivateUsage]  # private runtime under test


class CacheNamespace(Protocol):
    @property
    def cache(self) -> object: ...


@dataclass(frozen=True, slots=True)
class CacheTestResolver:
    namespace: CacheNamespace

    def resolve(self) -> CacheRuntime:
        return CacheRuntime.from_selected(self.namespace.cache)


class CacheLookup(Protocol):
    def get_cache(self, **kwargs: object) -> object: ...
    def flush_cache(self) -> object: ...


def request(key: str = "key") -> dict[str, object]:
    return {"key": {"preset": key}}


def native_runtime(facade: Cache) -> CacheRuntime:
    return CacheRuntime.from_cache(facade)


def activate_native(facade: Cache) -> Cache:
    facade._native_cache = ResponseCacheRuntime(_native._ResponseCacheRuntime.from_cache(facade))  # pyright: ignore[reportPrivateUsage]  # explicitly select the runtime under test
    return facade


def assert_native_runtime(facade: Cache) -> ResponseCacheRuntime:
    runtime: Final = facade._native_cache  # pyright: ignore[reportPrivateUsage]  # the activation under test has no public accessor
    assert isinstance(runtime, ResponseCacheRuntime)
    assert runtime.kind == "native"
    return runtime


def completion_kwargs(label: str) -> dict[str, object]:
    return {"model": "gpt-4o", "messages": [{"role": "user", "content": f"{label} {uuid4().hex}"}]}
