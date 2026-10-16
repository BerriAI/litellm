import time
from typing import Final

import pytest

import litellm
from litellm.caching.caching import Cache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.batch_redis_get import PROXY_BatchRedisRequests
from litellm.types.caching import LiteLLMCacheType


class _BackendSpy:
    def __init__(self) -> None:
        self.keys: list[str] = []

    async def async_get_cache(self, key: str, *args, **kwargs):
        self.keys.append(key)
        return {"timestamp": time.time(), "response": {"key": key}}


def _caller(token: str | None) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(api_key=f"raw-{token}").model_copy(update={"token": token})


@pytest.mark.asyncio
async def test_batch_redis_hook_scopes_explicit_keys_and_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)
    backend = _BackendSpy()
    cache.cache = backend
    monkeypatch.setattr(litellm, "cache", cache)
    hook = PROXY_BatchRedisRequests()
    hook.in_memory_cache = InMemoryCache()

    request_a = {"metadata": {"user_api_key_auth": _caller("hashed-a")}, "cache_key": "shared"}
    request_b = {"metadata": {"user_api_key_auth": _caller("hashed-b")}, "cache_key": "shared"}

    first_a = await hook.async_get_cache(**request_a)
    second_a = await hook.async_get_cache(**request_a)
    first_b = await hook.async_get_cache(**request_b)
    missing = await hook.async_get_cache(
        metadata={"user_api_key_auth": _caller(None)},
        cache_key="shared",
    )

    assert first_a == second_a
    assert first_a != first_b
    assert missing is None
    assert len(backend.keys) == 2
    assert backend.keys[0] != backend.keys[1]
    assert all(key.startswith("caller:") for key in backend.keys)
