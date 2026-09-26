from typing import Final, Protocol
from uuid import uuid4

import pytest

from litellm.caching.caching import Cache
from litellm.rust_bridge import _native, catalog
from litellm.rust_bridge.catalog import CacheRule
from litellm.rust_bridge.configuration import Rollout
from litellm.rust_bridge.response_cache import ResponseCacheRuntime
from litellm.types.caching import LiteLLMCacheType

CacheTestHandle: Final = _native._CacheTestHandle  # pyright: ignore[reportPrivateUsage]  # test-only handle has no public module name


CacheTestResolver: Final = _native._CacheTestResolver  # pyright: ignore[reportPrivateUsage]  # test-only resolver has no public module name


class CacheLookup(Protocol):
    def get_cache(self, **kwargs: object) -> object: ...
    def flush_cache(self) -> object: ...


def request(key: str = "key") -> dict[str, object]:
    return {"key": {"preset": key}}


def require_rust(monkeypatch: pytest.MonkeyPatch, backend: LiteLLMCacheType) -> None:
    monkeypatch.setattr(catalog, "RULES", (CacheRule(Rollout.RUST_REQUIRED, backends=frozenset({backend})),))


def assert_native_runtime(facade: Cache) -> ResponseCacheRuntime:
    runtime: Final = facade._native_cache  # pyright: ignore[reportPrivateUsage]  # the activation under test has no public accessor
    assert isinstance(runtime, ResponseCacheRuntime)
    assert runtime.kind == "native"
    return runtime


def completion_kwargs(label: str) -> dict[str, object]:
    return {"model": "gpt-4o", "messages": [{"role": "user", "content": f"{label} {uuid4().hex}"}]}
