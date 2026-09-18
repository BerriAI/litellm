import json

import pytest

from litellm.proxy._experimental.mcp_server.byok_credential_cache import (
    CachedByokCredential,
    byok_credential_cache,
    byok_credential_cache_key,
    cache_byok_credential,
    get_cached_byok_credential,
)
from litellm.proxy.common_utils.auth_cache_invalidation_pubsub import AuthCacheInvalidationSubscriber
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache


class _FakeRedisCache:
    namespace = None

    def init_async_client(self) -> object:
        return object()


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


def test_peer_worker_invalidation_message_evicts_the_cached_credential():
    """The key a mutating worker broadcasts must be the key every other worker caches under."""
    cache_byok_credential("mallory", "srv-byok", "sk-revoked")
    cache_byok_credential("alice", "srv-byok", "sk-kept")
    subscriber = AuthCacheInvalidationSubscriber(
        redis_cache=_FakeRedisCache(),  # pyright: ignore[reportArgumentType]  # subscriber is never started; only its message handler runs
        user_api_key_cache=UserApiKeyCache(),
        additional_in_memory_caches=(byok_credential_cache,),
    )

    subscriber._apply_message(  # pyright: ignore[reportPrivateUsage]  # exercising the real cross-worker message handler
        {
            "type": "message",
            "data": json.dumps({"cache_key": byok_credential_cache_key("mallory", "srv-byok")}).encode(),
        }
    )

    assert get_cached_byok_credential("mallory", "srv-byok") is None
    assert get_cached_byok_credential("alice", "srv-byok") == CachedByokCredential(credential="sk-kept")
