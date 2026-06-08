import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import pytest

from litellm.caching.dual_cache import DualCache
from litellm.identity.cache import (
    IdentityCache,
    identity_cache_key,
    team_generation_key,
    user_generation_key,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache


def _user_api_key_cache() -> UserApiKeyCache:
    return UserApiKeyCache()


@pytest.mark.asyncio
async def test_set_then_get_returns_same_uak():
    cache = IdentityCache(dual_cache=_user_api_key_cache())
    uak = UserAPIKeyAuth(api_key="sk-x", user_id="u1", team_id="t1")
    token_hash = uak.token
    assert token_hash is not None

    await cache.set(token_hash, uak)
    got = await cache.get(token_hash)
    assert got is not None
    assert got.user_id == "u1"
    assert got.team_id == "t1"


@pytest.mark.asyncio
async def test_miss_returns_none():
    cache = IdentityCache(dual_cache=_user_api_key_cache())
    assert await cache.get("missing-hash") is None


@pytest.mark.asyncio
async def test_delete_clears_entry():
    dual = _user_api_key_cache()
    cache = IdentityCache(dual_cache=dual)
    uak = UserAPIKeyAuth(api_key="sk-x", user_id="u1")
    await cache.set(uak.token, uak)
    assert await cache.get(uak.token) is not None
    await cache.delete(uak.token)
    assert await cache.get(uak.token) is None


@pytest.mark.asyncio
async def test_generation_bump_invalidates_team_scoped_entry():
    dual = _user_api_key_cache()
    cache = IdentityCache(dual_cache=dual)
    uak = UserAPIKeyAuth(api_key="sk-x", user_id="u1", team_id="t-rotate")
    await cache.set(uak.token, uak)
    assert await cache.get(uak.token) is not None

    await cache.bump_generation(team_generation_key("t-rotate"))
    assert await cache.get(uak.token) is None


@pytest.mark.asyncio
async def test_generation_bump_for_unrelated_team_keeps_entry():
    dual = _user_api_key_cache()
    cache = IdentityCache(dual_cache=dual)
    uak = UserAPIKeyAuth(api_key="sk-x", user_id="u1", team_id="t-keep")
    await cache.set(uak.token, uak)

    await cache.bump_generation(team_generation_key("t-other"))
    assert await cache.get(uak.token) is not None


def test_key_format_is_versioned():
    assert identity_cache_key("abc").startswith("identity:v1:")


class _CountingCache:
    def __init__(self):
        self.batch_get_calls = 0
        self.single_get_calls = 0
        self.values: dict = {}

    async def async_batch_get_cache(self, keys, **kwargs):
        self.batch_get_calls += 1
        return [self.values.get(key) for key in keys]

    async def async_get_cache(self, key, **kwargs):
        self.single_get_calls += 1
        return self.values.get(key)


@pytest.mark.asyncio
async def test_snapshot_generations_uses_single_batch_read():
    fake = _CountingCache()
    cache = IdentityCache(dual_cache=fake)  # type: ignore[arg-type]
    uak = UserAPIKeyAuth(token="hash-x", user_id="u1", team_id="t1", org_id="o1")

    snapshot = await cache._snapshot_generations_for(uak)

    assert fake.batch_get_calls == 1
    assert fake.single_get_calls == 0
    assert snapshot == {"team": 0, "user": 0, "org": 0}


def test_get_identity_cache_is_a_process_singleton():
    import litellm.identity.cache as cache_module

    saved = cache_module._identity_cache
    cache_module._identity_cache = None
    try:
        first_backend = _user_api_key_cache()
        first = cache_module.get_identity_cache(first_backend)
        second = cache_module.get_identity_cache(_user_api_key_cache())
        assert first is second
        assert first._cache is first_backend
    finally:
        cache_module._identity_cache = saved


@pytest.mark.asyncio
async def test_snapshot_generations_maps_returned_values_by_scope():
    fake = _CountingCache()
    fake.values[team_generation_key("t1")] = 7
    fake.values[user_generation_key("u1")] = 3
    cache = IdentityCache(dual_cache=fake)  # type: ignore[arg-type]
    uak = UserAPIKeyAuth(token="hash-x", user_id="u1", team_id="t1")

    snapshot = await cache._snapshot_generations_for(uak)

    assert snapshot == {"team": 7, "user": 3}
