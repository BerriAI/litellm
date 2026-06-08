import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import pytest

from litellm.caching.dual_cache import DualCache
from litellm.identity.cache import (
    IdentityCache,
    identity_cache_key,
    team_generation_key,
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
