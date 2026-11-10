import asyncio
from typing import Final

import pytest

import litellm
from litellm.caching.caching import Cache
from litellm.caching.caching_handler import _PENDING_CACHE_WRITES
from litellm.rust_bridge.host.cache import configured_cache, store_response
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


_REQUEST: Final = {"model": "claude-sonnet-4-5", "messages": [{"role": "user", "content": "hi"}]}


async def _stored(response: dict[str, object]) -> object:
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)
    store_response(cache, response, **_REQUEST)
    await asyncio.gather(*_PENDING_CACHE_WRITES)
    return await cache.async_get_cache(**_REQUEST)


@pytest.mark.asyncio
async def test_store_response_drops_only_the_rust_marker_header_from_the_cached_entry() -> None:
    stored: Final = await _stored(
        {
            "id": "msg_1",
            "_hidden_params": {"additional_headers": {"x-litellm-rust": "true", "llm_provider-x": "1"}, "model": "m"},
        }
    )
    assert stored == {
        "id": "msg_1",
        "_hidden_params": {"additional_headers": {"llm_provider-x": "1"}, "model": "m"},
    }


@pytest.mark.asyncio
async def test_store_response_keeps_an_entry_without_the_rust_marker_unchanged() -> None:
    response: Final = {"id": "msg_2", "_hidden_params": {"additional_headers": {"llm_provider-x": "1"}}}
    assert await _stored(response) == response
