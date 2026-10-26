from typing import Final

import pytest

import litellm
from litellm.caching.caching import Cache
from litellm.rust_bridge.host.cache import configured_cache
from litellm.types.caching import LiteLLMCacheType


def test_configured_cache_preserves_identity_and_observes_runtime_replacement(monkeypatch: pytest.MonkeyPatch) -> None:
    initial: Final = Cache(type=LiteLLMCacheType.LOCAL)
    replacement: Final = Cache(type=LiteLLMCacheType.LOCAL)
    monkeypatch.setattr(litellm, "cache", initial)
    assert configured_cache() is initial

    monkeypatch.setattr(litellm, "cache", replacement)
    assert configured_cache() is replacement

    monkeypatch.setattr(litellm, "cache", None)
    assert configured_cache() is None
