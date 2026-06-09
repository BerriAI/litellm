import os
import sys
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.abspath("../.."))

import pytest
from fastapi import status

from litellm.identity.cache import IdentityCache
from litellm.identity.store import load_identity
from litellm.proxy._types import (
    LiteLLM_VerificationTokenView,
    ProxyErrorTypes,
    ProxyException,
)
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache


def _stub_prisma_client():
    """Minimal prisma_client stub with the get_data hook the store uses."""

    class _Stub:
        def __init__(self):
            self.get_data = AsyncMock()
            self.calls = 0

    return _Stub()


def _verification_token_view(**fields) -> LiteLLM_VerificationTokenView:
    return LiteLLM_VerificationTokenView(
        token=fields.pop("token", "hash-x"),
        user_id=fields.pop("user_id", "u1"),
        team_id=fields.pop("team_id", "t1"),
        org_id=fields.pop("org_id", None),
        **fields,
    )


@pytest.mark.asyncio
async def test_cache_hit_skips_db():
    cache_backend = UserApiKeyCache()
    identity_cache = IdentityCache(dual_cache=cache_backend)
    prisma = _stub_prisma_client()
    from litellm.proxy._types import UserAPIKeyAuth

    seed = UserAPIKeyAuth(token="hash-cached", user_id="u-cache", team_id="t-cache")
    await identity_cache.set("hash-cached", seed)

    result = await load_identity(
        hashed_token="hash-cached",
        prisma_client=prisma,
        cache=identity_cache,
        user_api_key_cache=cache_backend,
    )

    assert prisma.get_data.await_count == 0
    assert result.user_id == "u-cache"
    assert result.team_id == "t-cache"


@pytest.mark.asyncio
async def test_cache_miss_hits_db_once_and_caches():
    cache_backend = UserApiKeyCache()
    identity_cache = IdentityCache(dual_cache=cache_backend)
    prisma = _stub_prisma_client()
    prisma.get_data.return_value = _verification_token_view(
        token="hash-db", user_id="u-db", team_id="t-db"
    )

    with patch("litellm.identity.store._populate_legacy_cache", new=AsyncMock()):
        result = await load_identity(
            hashed_token="hash-db",
            prisma_client=prisma,
            cache=identity_cache,
            user_api_key_cache=cache_backend,
        )

    assert prisma.get_data.await_count == 1
    assert result.user_id == "u-db"
    cached = await identity_cache.get("hash-db")
    assert cached is not None
    assert cached.user_id == "u-db"


@pytest.mark.asyncio
async def test_warm_load_after_cold_does_not_hit_db():
    cache_backend = UserApiKeyCache()
    identity_cache = IdentityCache(dual_cache=cache_backend)
    prisma = _stub_prisma_client()
    prisma.get_data.return_value = _verification_token_view(
        token="hash-warm", user_id="u-warm", team_id="t-warm"
    )

    with patch("litellm.identity.store._populate_legacy_cache", new=AsyncMock()):
        await load_identity(
            hashed_token="hash-warm",
            prisma_client=prisma,
            cache=identity_cache,
            user_api_key_cache=cache_backend,
        )
        await load_identity(
            hashed_token="hash-warm",
            prisma_client=prisma,
            cache=identity_cache,
            user_api_key_cache=cache_backend,
        )

    assert prisma.get_data.await_count == 1


@pytest.mark.asyncio
async def test_missing_token_raises_proxy_exception_with_401():
    cache_backend = UserApiKeyCache()
    identity_cache = IdentityCache(dual_cache=cache_backend)
    prisma = _stub_prisma_client()
    prisma.get_data.return_value = None

    with pytest.raises(ProxyException) as excinfo:
        await load_identity(
            hashed_token="missing-hash",
            prisma_client=prisma,
            cache=identity_cache,
            user_api_key_cache=cache_backend,
        )

    assert int(excinfo.value.code) == status.HTTP_401_UNAUTHORIZED
    assert excinfo.value.type == ProxyErrorTypes.token_not_found_in_db


@pytest.mark.asyncio
async def test_bundled_user_survives_cache_roundtrip_as_typed_model():
    from litellm.identity.cache import IdentityCache
    from litellm.identity.store import _rehydrate_bundled_user
    from litellm.proxy._types import LiteLLM_UserTable, UserAPIKeyAuth

    cache_backend = UserApiKeyCache()
    identity_cache = IdentityCache(dual_cache=cache_backend)
    uak = UserAPIKeyAuth(
        api_key="sk-bundle",
        user_id="u-bundle",
        user=LiteLLM_UserTable(user_id="u-bundle", user_email="x@y.com", tpm_limit=10),
    )
    await identity_cache.set(uak.token, uak)
    got = await identity_cache.get(uak.token)
    _rehydrate_bundled_user(got)

    assert isinstance(got.user, LiteLLM_UserTable)
    assert got.user.user_email == "x@y.com"
    assert got.user.tpm_limit == 10


@pytest.mark.asyncio
async def test_cold_path_bundles_user_into_cache():
    """``load_identity`` must populate ``user`` on first DB hit so the
    next request reads the user object from the identity cache."""
    cache_backend = UserApiKeyCache()
    identity_cache = IdentityCache(dual_cache=cache_backend)
    prisma = _stub_prisma_client()
    prisma.get_data.return_value = _verification_token_view(
        token="hash-user", user_id="u-bundle", team_id="t-bundle"
    )

    from litellm.proxy._types import LiteLLM_UserTable

    async def _fake_get_user_object(*, user_id, **kwargs):
        return LiteLLM_UserTable(
            user_id=user_id, user_email="bundle@litellm.io", tpm_limit=42
        )

    with (
        patch("litellm.identity.store._populate_legacy_cache", new=AsyncMock()),
        patch(
            "litellm.proxy.auth.auth_checks.get_user_object",
            side_effect=_fake_get_user_object,
        ),
    ):
        result = await load_identity(
            hashed_token="hash-user",
            prisma_client=prisma,
            cache=identity_cache,
            user_api_key_cache=cache_backend,
        )

    assert result.user is not None
    assert result.user.user_email == "bundle@litellm.io"
    cached = await identity_cache.get("hash-user")
    assert cached is not None
    assert cached.user is not None


@pytest.mark.asyncio
async def test_hydrated_copy_is_request_scoped():
    cache_backend = UserApiKeyCache()
    identity_cache = IdentityCache(dual_cache=cache_backend)
    prisma = _stub_prisma_client()
    prisma.get_data.return_value = _verification_token_view(
        token="hash-copy", user_id="u-copy", team_id="t-copy"
    )

    with patch("litellm.identity.store._populate_legacy_cache", new=AsyncMock()):
        a = await load_identity(
            hashed_token="hash-copy",
            prisma_client=prisma,
            cache=identity_cache,
            user_api_key_cache=cache_backend,
        )
        a.parent_otel_span = "request-A-span"
        a.request_route = "/v1/chat/completions"

        b = await load_identity(
            hashed_token="hash-copy",
            prisma_client=prisma,
            cache=identity_cache,
            user_api_key_cache=cache_backend,
        )

    assert b.parent_otel_span is None
    assert b.request_route is None


@pytest.mark.asyncio
async def test_cache_miss_without_prisma_raises_no_db_connected():
    cache_backend = UserApiKeyCache()
    identity_cache = IdentityCache(dual_cache=cache_backend)

    with pytest.raises(Exception) as excinfo:
        await load_identity(
            hashed_token="hash-no-db",
            prisma_client=None,
            cache=identity_cache,
            user_api_key_cache=cache_backend,
        )

    assert "No DB Connected" in str(excinfo.value)


@pytest.mark.asyncio
async def test_cold_path_fetches_object_permission_when_only_id_present():
    cache_backend = UserApiKeyCache()
    identity_cache = IdentityCache(dual_cache=cache_backend)
    prisma = _stub_prisma_client()
    prisma.get_data.return_value = _verification_token_view(
        token="hash-op",
        user_id=None,
        team_id=None,
        object_permission_id="op-99",
    )

    from litellm.proxy._types import LiteLLM_ObjectPermissionTable

    fetched = LiteLLM_ObjectPermissionTable(object_permission_id="op-99")
    with (
        patch("litellm.identity.store._populate_legacy_cache", new=AsyncMock()),
        patch(
            "litellm.proxy.auth.auth_checks.get_object_permission",
            new=AsyncMock(return_value=fetched),
        ) as mock_get_perm,
    ):
        result = await load_identity(
            hashed_token="hash-op",
            prisma_client=prisma,
            cache=identity_cache,
            user_api_key_cache=cache_backend,
        )

    assert mock_get_perm.await_count == 1
    assert result.object_permission is not None
    assert result.object_permission.object_permission_id == "op-99"


@pytest.mark.asyncio
async def test_cold_path_swallows_object_permission_lookup_failure():
    cache_backend = UserApiKeyCache()
    identity_cache = IdentityCache(dual_cache=cache_backend)
    prisma = _stub_prisma_client()
    prisma.get_data.return_value = _verification_token_view(
        token="hash-op-err",
        user_id=None,
        team_id=None,
        object_permission_id="op-err",
    )

    with (
        patch("litellm.identity.store._populate_legacy_cache", new=AsyncMock()),
        patch(
            "litellm.proxy.auth.auth_checks.get_object_permission",
            new=AsyncMock(side_effect=RuntimeError("db blip")),
        ),
    ):
        result = await load_identity(
            hashed_token="hash-op-err",
            prisma_client=prisma,
            cache=identity_cache,
            user_api_key_cache=cache_backend,
        )

    assert result.object_permission is None
    assert result.object_permission_id == "op-err"


@pytest.mark.asyncio
async def test_populate_legacy_cache_delegates_to_cache_key_object():
    from litellm.identity.store import _populate_legacy_cache
    from litellm.proxy._types import UserAPIKeyAuth

    uak = UserAPIKeyAuth(token="hash-legacy", user_id="u1")
    with patch(
        "litellm.proxy.auth.auth_checks._cache_key_object", new=AsyncMock()
    ) as mock_cache_key:
        await _populate_legacy_cache(
            hashed_token="hash-legacy",
            uak=uak,
            user_api_key_cache=UserApiKeyCache(),
            proxy_logging_obj=None,
        )

    assert mock_cache_key.await_count == 1
    assert mock_cache_key.await_args.kwargs["hashed_token"] == "hash-legacy"
    assert mock_cache_key.await_args.kwargs["user_api_key_obj"] is uak


@pytest.mark.asyncio
async def test_populate_legacy_cache_swallows_write_failures():
    from litellm.identity.store import _populate_legacy_cache
    from litellm.proxy._types import UserAPIKeyAuth

    uak = UserAPIKeyAuth(token="hash-legacy", user_id="u1")
    with patch(
        "litellm.proxy.auth.auth_checks._cache_key_object",
        new=AsyncMock(side_effect=RuntimeError("redis down")),
    ):
        await _populate_legacy_cache(
            hashed_token="hash-legacy",
            uak=uak,
            user_api_key_cache=UserApiKeyCache(),
            proxy_logging_obj=None,
        )


def test_rehydrate_bundled_user_nulls_out_uncoercible_dict():
    from litellm.identity.store import _rehydrate_bundled_user
    from litellm.proxy._types import UserAPIKeyAuth

    uak = UserAPIKeyAuth(token="hash-bad-user")
    uak.user = {"user_id": "u1", "tpm_limit": "not-an-int"}

    _rehydrate_bundled_user(uak)

    assert uak.user is None
