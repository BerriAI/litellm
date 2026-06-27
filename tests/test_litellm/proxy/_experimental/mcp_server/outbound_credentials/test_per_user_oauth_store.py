import pytest

from litellm.proxy._experimental.mcp_server.outbound_credentials.oauth_token_store import (
    OAuthToken,
    OAuthTokenStore,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.per_user_oauth_store import (
    LazyPerUserOAuthTokenStore,
    ServerLookup,
)


class _RecordingStore:
    def __init__(self, access_token: str) -> None:
        self._access_token = access_token
        self.calls: list[tuple[str, str]] = []

    async def fetch(self, user_id: str, server_id: str) -> OAuthToken | None:
        self.calls.append((user_id, server_id))
        return OAuthToken(access_token=self._access_token)


class _RedisAvailability:
    def __init__(self) -> None:
        self.available = False

    def __call__(self) -> bool:
        return self.available


@pytest.mark.asyncio
async def test_lazy_store_rebuilds_when_redis_becomes_available() -> None:
    local_store = _RecordingStore("local")
    redis_store = _RecordingStore("redis")
    redis_available = _RedisAvailability()
    build_calls = 0

    def build_store(_server_lookup: ServerLookup) -> tuple[OAuthTokenStore, bool]:
        nonlocal build_calls
        build_calls += 1
        if redis_available.available:
            return redis_store, True
        return local_store, False

    def server_lookup(_server_id: str) -> None:
        return None

    store = LazyPerUserOAuthTokenStore(
        server_lookup,
        store_builder=build_store,
        redis_available=redis_available,
    )

    first = await store.fetch("u", "s")
    redis_available.available = True
    second = await store.fetch("u", "s")
    third = await store.fetch("u", "s")

    assert first is not None and first.access_token == "local"
    assert second is not None and second.access_token == "redis"
    assert third is not None and third.access_token == "redis"
    assert build_calls == 2
    assert local_store.calls == [("u", "s")]
    assert redis_store.calls == [("u", "s"), ("u", "s")]
