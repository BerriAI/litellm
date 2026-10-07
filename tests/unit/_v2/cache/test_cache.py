from typing import Final

import pytest

import litellm
from litellm import _v2
from litellm._v2.cache import NativeBackend

pytestmark = pytest.mark.requires_rust_extension


@pytest.mark.asyncio
async def test_v2_global_cache_leaves_legacy_only_calls_usable() -> None:
    litellm.cache = _v2.Cache.memory()  # test-quality-ok: isolate_ocr_test_state restores the cache global
    response: Final = await litellm.aembedding(
        model="openai/cache-test-embedding",
        input=["hello"],
        api_key="test-key",
        mock_response=[0.25, 0.75],
    )
    assert response.model_dump(include={"data"}) == {
        "data": [{"embedding": [0.25, 0.75], "index": 0, "object": "embedding"}]
    }


@pytest.mark.asyncio
async def test_v2_facade_and_backend_share_storage_and_management() -> None:
    cache: Final = _v2.Cache.memory()
    await cache.async_add_cache({"answer": 7}, cache_key="shared")
    assert cache.get_cache(cache_key="shared") == {"answer": 7}
    assert await cache.ping() is True
    await cache.delete_cache_keys(["shared"])
    assert await cache.async_get_cache(cache_key="shared") is None
    cache.add_cache({"answer": 8}, cache_key="flush")
    backend: Final = cache.cache
    assert isinstance(backend, NativeBackend)
    backend.flush_cache()
    assert cache.get_cache(cache_key="flush") is None
    await cache.disconnect()
