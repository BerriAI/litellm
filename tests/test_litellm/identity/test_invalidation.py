import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import pytest

from litellm.identity.cache import IdentityCache, team_generation_key
from litellm.identity.invalidation import (
    invalidate_identity_for_org,
    invalidate_identity_for_team,
    invalidate_identity_for_token,
    invalidate_identity_for_user,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache


@pytest.mark.asyncio
async def test_token_invalidation_drops_entry():
    backend = UserApiKeyCache()
    cache = IdentityCache(dual_cache=backend)
    uak = UserAPIKeyAuth(api_key="sk-x", user_id="u1")
    await cache.set(uak.token, uak)
    assert await cache.get(uak.token) is not None

    await invalidate_identity_for_token(token_hash=uak.token, dual_cache=backend)
    assert await cache.get(uak.token) is None


@pytest.mark.asyncio
async def test_team_invalidation_bumps_generation():
    backend = UserApiKeyCache()
    cache = IdentityCache(dual_cache=backend)
    uak = UserAPIKeyAuth(api_key="sk-x", user_id="u1", team_id="t-rotate")
    await cache.set(uak.token, uak)

    await invalidate_identity_for_team(team_id="t-rotate", dual_cache=backend)
    assert await cache.get(uak.token) is None


@pytest.mark.asyncio
async def test_user_invalidation_is_scoped_to_user():
    backend = UserApiKeyCache()
    cache = IdentityCache(dual_cache=backend)
    uak_a = UserAPIKeyAuth(api_key="sk-a", user_id="u-rotate", team_id="t1")
    uak_b = UserAPIKeyAuth(api_key="sk-b", user_id="u-keep", team_id="t1")
    await cache.set(uak_a.token, uak_a)
    await cache.set(uak_b.token, uak_b)

    await invalidate_identity_for_user(user_id="u-rotate", dual_cache=backend)

    assert await cache.get(uak_a.token) is None
    assert await cache.get(uak_b.token) is not None


@pytest.mark.asyncio
async def test_org_invalidation_drops_org_scoped_entry():
    backend = UserApiKeyCache()
    cache = IdentityCache(dual_cache=backend)
    uak = UserAPIKeyAuth(api_key="sk-x", user_id="u1", org_id="org-rotate")
    await cache.set(uak.token, uak)

    await invalidate_identity_for_org(org_id="org-rotate", dual_cache=backend)
    assert await cache.get(uak.token) is None
