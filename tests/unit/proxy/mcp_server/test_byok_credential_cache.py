from typing import Final

import pytest

from litellm.proxy import proxy_server
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.mcp_server.byok_credential_cache import (
    CachedByokCredential,
    byok_credential_cache,
    cache_byok_credential,
    get_cached_byok_credential,
)
from litellm.proxy.mcp_server.operations import (
    _invalidate_byok_cred_cache,  # pyright: ignore[reportPrivateUsage]  # the broadcast every BYOK mutation runs
)
from litellm.redis.stores.auth_cache_invalidation import PythonCaches
from tests.unit.redis.stores.auth_cache_invalidation_double import Worker, channel


@pytest.fixture(autouse=True)
def _empty_cache():
    byok_credential_cache.flush_cache()
    yield
    byok_credential_cache.flush_cache()


def test_a_cached_negative_lookup_is_distinguishable_from_a_miss():
    assert get_cached_byok_credential("u-1", "srv-1") is None
    cache_byok_credential("u-1", "srv-1", None)
    assert get_cached_byok_credential("u-1", "srv-1") == CachedByokCredential(credential=None)
    cache_byok_credential("u-1", "srv-1", "sk-stored")
    assert get_cached_byok_credential("u-1", "srv-1") == CachedByokCredential(credential="sk-stored")
    assert get_cached_byok_credential("u-1", "srv-2") is None


@pytest.mark.asyncio
async def test_peer_worker_invalidation_message_evicts_the_cached_credential(monkeypatch: pytest.MonkeyPatch):
    """The key a mutating worker broadcasts must be the key every other worker caches under."""
    invalidation, fake = channel()
    monkeypatch.setattr(proxy_server.proxy_config, "auth_cache_invalidation", invalidation)
    await _invalidate_byok_cred_cache("mallory", "srv-byok")
    cache_byok_credential("mallory", "srv-byok", "sk-revoked")
    cache_byok_credential("alice", "srv-byok", "sk-kept")
    peer: Final = Worker(UserApiKeyCache(), PythonCaches(copies=(byok_credential_cache,)))

    for message in fake.published:
        peer.apply(message)

    assert get_cached_byok_credential("mallory", "srv-byok") is None
    assert get_cached_byok_credential("alice", "srv-byok") == CachedByokCredential(credential="sk-kept")
