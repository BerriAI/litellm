import os
import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from fastapi import HTTPException

from litellm.caching.caching import DualCache
from litellm.proxy._types import (
    JWTKeyMappingResponse,
    LiteLLM_JWTAuth,
    LitellmUserRoles,
    UserAPIKeyAuth,
)
from litellm.proxy.management_endpoints.jwt_key_mapping_endpoints import (
    _to_response,
    _token_hash_for_create,
    _token_hash_for_update,
    create_jwt_key_mapping,
    delete_jwt_key_mapping,
    info_jwt_key_mapping,
    update_jwt_key_mapping,
)

# ──────────────────────────────────────────────
# Tests: _resolve_jwt_to_virtual_key
# ──────────────────────────────────────────────


# ──────────────────────────────────────────────
# Tests: OIDC / JWT routing in user_api_key_auth
# ──────────────────────────────────────────────


# ──────────────────────────────────────────────
# Tests: _to_response redacts hashed token
# ──────────────────────────────────────────────


def test_to_response_excludes_token():
    """_to_response should not expose the hashed token field."""
    now = datetime.now(timezone.utc)
    mock_mapping = MagicMock()
    mock_mapping.id = "mapping-1"
    mock_mapping.jwt_issuer = None
    mock_mapping.jwt_claim_name = "email"
    mock_mapping.jwt_claim_value = "user@example.com"
    mock_mapping.token = "hashed_secret_value"
    mock_mapping.description = "test"
    mock_mapping.is_active = True
    mock_mapping.created_at = now
    mock_mapping.updated_at = now
    mock_mapping.created_by = "admin"
    mock_mapping.updated_by = "admin"

    resp = _to_response(mock_mapping)

    assert isinstance(resp, JWTKeyMappingResponse)
    assert resp.id == "mapping-1"
    assert resp.jwt_claim_name == "email"
    assert "token" not in resp.model_fields


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _make_admin_auth() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        token="sk-admin",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )


def _make_non_admin_auth() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        token="sk-user",
        user_role=LitellmUserRoles.INTERNAL_USER,
    )


def _mock_prisma():
    prisma = MagicMock()
    prisma.db.litellm_jwtkeymapping.create = AsyncMock()
    prisma.db.litellm_jwtkeymapping.find_unique = AsyncMock()
    prisma.db.litellm_jwtkeymapping.find_many = AsyncMock()
    prisma.db.litellm_jwtkeymapping.update = AsyncMock()
    prisma.db.litellm_jwtkeymapping.delete = AsyncMock()
    prisma.db.litellm_jwtkeymapping.count = AsyncMock(return_value=0)
    return prisma


def _mock_mapping(
    id="mapping-1",
    claim_name="email",
    claim_value="user@example.com",
    issuer=None,
):
    now = datetime.now(timezone.utc)
    m = MagicMock()
    m.id = id
    m.jwt_issuer = issuer
    m.jwt_claim_name = claim_name
    m.jwt_claim_value = claim_value
    m.token = "hashed_token"
    m.description = None
    m.is_active = True
    m.created_at = now
    m.updated_at = now
    m.created_by = "admin"
    m.updated_by = "admin"
    return m


# ──────────────────────────────────────────────
# Tests: CRUD endpoint error handling
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_returns_409_on_unique_violation():
    """Duplicate mapping should return 409, not 500."""
    from litellm.proxy._types import CreateJWTKeyMappingRequest

    mock_prisma = _mock_prisma()
    mock_prisma.db.litellm_jwtkeymapping.create.side_effect = Exception("Unique constraint failed (P2002)")
    mock_cache = AsyncMock()

    data = CreateJWTKeyMappingRequest(
        jwt_claim_name="email",
        jwt_claim_value="user@example.com",
        key="sk-test-key",
    )

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.user_api_key_cache", mock_cache),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await create_jwt_key_mapping(data=data, user_api_key_dict=_make_admin_auth())
        assert exc_info.value.status_code == 409
        assert "already exists" in exc_info.value.detail


@pytest.mark.asyncio
async def test_create_returns_400_on_foreign_key_violation():
    """Non-existent key should return 400, not 500."""
    from litellm.proxy._types import CreateJWTKeyMappingRequest

    mock_prisma = _mock_prisma()
    mock_prisma.db.litellm_jwtkeymapping.create.side_effect = Exception(
        "Foreign key constraint failed on field: `token` (P2003)"
    )
    mock_cache = AsyncMock()

    data = CreateJWTKeyMappingRequest(
        jwt_claim_name="sub",
        jwt_claim_value="user-999",
        key="sk-nonexistent",
    )

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.user_api_key_cache", mock_cache),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await create_jwt_key_mapping(data=data, user_api_key_dict=_make_admin_auth())
        assert exc_info.value.status_code == 400
        assert "does not match" in exc_info.value.detail


@pytest.mark.asyncio
async def test_create_non_admin_returns_403():
    """Non-admin users should get 403."""
    from litellm.proxy._types import CreateJWTKeyMappingRequest

    data = CreateJWTKeyMappingRequest(
        jwt_claim_name="email",
        jwt_claim_value="user@example.com",
        key="sk-test",
    )

    with pytest.raises(HTTPException) as exc_info:
        await create_jwt_key_mapping(data=data, user_api_key_dict=_make_non_admin_auth())
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_delete_returns_404_when_not_found():
    """Deleting non-existent mapping should return 404."""
    from litellm.proxy._types import DeleteJWTKeyMappingRequest

    mock_prisma = _mock_prisma()
    mock_prisma.db.litellm_jwtkeymapping.find_unique.return_value = None
    mock_cache = AsyncMock()

    data = DeleteJWTKeyMappingRequest(id="nonexistent-id")

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.user_api_key_cache", mock_cache),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await delete_jwt_key_mapping(data=data, user_api_key_dict=_make_admin_auth())
        assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_update_returns_404_when_not_found():
    """Updating non-existent mapping should return 404."""
    from litellm.proxy._types import UpdateJWTKeyMappingRequest

    mock_prisma = _mock_prisma()
    mock_prisma.db.litellm_jwtkeymapping.find_unique.return_value = None
    mock_cache = AsyncMock()

    data = UpdateJWTKeyMappingRequest(id="nonexistent-id", description="test")

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.user_api_key_cache", mock_cache),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await update_jwt_key_mapping(data=data, user_api_key_dict=_make_admin_auth())
        assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_update_returns_404_when_row_deleted_before_write():
    """A mapping deleted between the read and the write must 404, not 500.

    Prisma's update returns None when the row is gone, and the endpoint used to
    dereference it for the cache key.
    """
    from litellm.proxy._types import UpdateJWTKeyMappingRequest

    mock_prisma = _mock_prisma()
    mock_prisma.db.litellm_jwtkeymapping.find_unique.return_value = _mock_mapping()
    mock_prisma.db.litellm_jwtkeymapping.update.return_value = None
    mock_cache = AsyncMock()

    data = UpdateJWTKeyMappingRequest(id="mapping-1", description="test")

    with (
        patch(
            "litellm.proxy.proxy_server.prisma_client", mock_prisma
        ),  # test-quality-ok: proxy_server module global is the endpoint's only injection point
        patch(
            "litellm.proxy.proxy_server.user_api_key_cache", mock_cache
        ),  # test-quality-ok: proxy_server module global is the endpoint's only injection point
    ):
        with pytest.raises(HTTPException) as exc_info:
            await update_jwt_key_mapping(data=data, user_api_key_dict=_make_admin_auth())
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Mapping not found"


@pytest.mark.asyncio
async def test_info_returns_404_when_not_found():
    """Getting info for non-existent mapping should return 404."""
    mock_prisma = _mock_prisma()
    mock_prisma.db.litellm_jwtkeymapping.find_unique.return_value = None

    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma):
        with pytest.raises(HTTPException) as exc_info:
            await info_jwt_key_mapping(id="nonexistent-id", user_api_key_dict=_make_admin_auth())
        assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_create_success_returns_response_without_token():
    """Successful create should return JWTKeyMappingResponse without hashed token."""
    from litellm.proxy._types import CreateJWTKeyMappingRequest

    mock_prisma = _mock_prisma()
    mock_prisma.db.litellm_jwtkeymapping.create.return_value = _mock_mapping()
    mock_cache = AsyncMock()

    data = CreateJWTKeyMappingRequest(
        jwt_claim_name="email",
        jwt_claim_value="user@example.com",
        key="sk-test-key",
    )

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.user_api_key_cache", mock_cache),
    ):
        result = await create_jwt_key_mapping(data=data, user_api_key_dict=_make_admin_auth())
        assert isinstance(result, JWTKeyMappingResponse)
        assert "token" not in result.model_fields
        assert result.jwt_claim_name == "email"


@pytest.mark.asyncio
async def test_create_without_issuer_stores_empty_string_not_null():
    """LIT-7417: the DB column is NOT NULL (see schema.prisma). Storing a real NULL
    for an unscoped mapping would let Postgres accept unlimited duplicate unscoped
    rows for the same claim (NULL is never equal to NULL in a unique constraint),
    so two mappings for the same claim value could point at two different keys with
    no conflict, and resolution would pick whichever one Postgres returns first."""
    from litellm.proxy._types import CreateJWTKeyMappingRequest

    mock_prisma = _mock_prisma()
    mock_prisma.db.litellm_jwtkeymapping.create.return_value = _mock_mapping()
    mock_cache = AsyncMock()

    data = CreateJWTKeyMappingRequest(jwt_claim_name="sub", jwt_claim_value="dev-alice", key="sk-test-key")

    with (
        patch(  # test-quality-ok: proxy_server module global is the endpoint's only injection point
            "litellm.proxy.proxy_server.prisma_client", mock_prisma
        ),
        patch(  # test-quality-ok: proxy_server module global is the endpoint's only injection point
            "litellm.proxy.proxy_server.user_api_key_cache", mock_cache
        ),
    ):
        await create_jwt_key_mapping(data=data, user_api_key_dict=_make_admin_auth())

    sent_data = mock_prisma.db.litellm_jwtkeymapping.create.call_args.kwargs["data"]
    assert sent_data["jwt_issuer"] == ""


# ──────────────────────────────────────────────
# Tests: unregistered_jwt_client_behavior
# ──────────────────────────────────────────────


# ──────────────────────────────────────────────
# Tests: prisma_client=None does not bypass no-match policy
# ──────────────────────────────────────────────


# ──────────────────────────────────────────────
# Tests: AUTO_REGISTER stamps validated identity from auth_builder
# ──────────────────────────────────────────────


# ──────────────────────────────────────────────
# Tests: backward-compat alias jwt_client_id_field
# ──────────────────────────────────────────────


def test_jwt_client_id_field_alias_maps_to_virtual_key_claim_field():
    """
    jwt_client_id_field (old doc name) must silently alias to virtual_key_claim_field.
    """
    auth = LiteLLM_JWTAuth(jwt_client_id_field="azp")
    assert auth.virtual_key_claim_field == "azp"


def test_jwt_client_id_field_does_not_raise_on_duplicate():
    """
    If both jwt_client_id_field and virtual_key_claim_field are supplied,
    virtual_key_claim_field takes precedence and no error is raised.
    """
    auth = LiteLLM_JWTAuth(
        jwt_client_id_field="old_field",
        virtual_key_claim_field="new_field",
    )
    assert auth.virtual_key_claim_field == "new_field"


# ──────────────────────────────────────────────
# Tests: identifying the mapped key by hash instead of plaintext
# ──────────────────────────────────────────────

_TOKEN_HASH = "1923314ae0efc8b2523c7d421bac5a7cf88df291273b139948b526d396974a41"


def test_create_stores_a_supplied_token_hash_verbatim():
    """A caller that holds only the hash gets it stored as given, not hashed again."""
    from litellm.proxy._types import CreateJWTKeyMappingRequest

    data = CreateJWTKeyMappingRequest(jwt_claim_name="email", jwt_claim_value="user@example.com", token=_TOKEN_HASH)

    assert _token_hash_for_create(data) == _TOKEN_HASH


def test_create_hashes_a_supplied_plaintext_key():
    """Supplying `key` keeps the original behaviour, so existing configs are unaffected."""
    from litellm.proxy._types import CreateJWTKeyMappingRequest, hash_token

    data = CreateJWTKeyMappingRequest(jwt_claim_name="email", jwt_claim_value="user@example.com", key="sk-test-key")

    assert _token_hash_for_create(data) == hash_token("sk-test-key")


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({}, id="neither"),
        pytest.param({"key": "sk-test-key", "token": _TOKEN_HASH}, id="both"),
    ],
)
def test_create_requires_exactly_one_identifier(kwargs):
    """Neither or both is a 400, so a mapping can never be created ambiguously."""
    from litellm.proxy._types import CreateJWTKeyMappingRequest

    data = CreateJWTKeyMappingRequest(jwt_claim_name="email", jwt_claim_value="user@example.com", **kwargs)

    with pytest.raises(HTTPException) as exc_info:
        _token_hash_for_create(data)

    assert exc_info.value.status_code == 400
    assert "exactly one" in exc_info.value.detail.lower()


@pytest.mark.parametrize(
    "token",
    [
        pytest.param("sk-not-a-hash", id="plaintext-key"),
        pytest.param("abc123", id="too-short"),
        pytest.param(_TOKEN_HASH.upper(), id="uppercase"),
        pytest.param(_TOKEN_HASH + "0", id="too-long"),
        pytest.param(_TOKEN_HASH[:-1] + "g", id="non-hex-character"),
    ],
)
def test_create_rejects_a_token_that_is_not_a_sha256_hash(token):
    """hash_token hashes unconditionally, so a bad `token` would be stored as a hash
    of a hash and then silently match nothing at auth time."""
    from litellm.proxy._types import CreateJWTKeyMappingRequest

    data = CreateJWTKeyMappingRequest(jwt_claim_name="email", jwt_claim_value="user@example.com", token=token)

    with pytest.raises(HTTPException) as exc_info:
        _token_hash_for_create(data)

    assert exc_info.value.status_code == 400
    assert "SHA-256" in exc_info.value.detail


def test_update_leaves_the_mapped_key_alone_when_neither_is_given():
    """Updating only the description must not blank out the mapped key."""
    from litellm.proxy._types import UpdateJWTKeyMappingRequest

    data = UpdateJWTKeyMappingRequest(id="mapping-1", description="new text")

    assert _token_hash_for_update(data) is None


def test_update_stores_a_supplied_token_hash_verbatim():
    from litellm.proxy._types import UpdateJWTKeyMappingRequest

    data = UpdateJWTKeyMappingRequest(id="mapping-1", token=_TOKEN_HASH)

    assert _token_hash_for_update(data) == _TOKEN_HASH


def test_update_hashes_a_supplied_plaintext_key():
    from litellm.proxy._types import UpdateJWTKeyMappingRequest, hash_token

    data = UpdateJWTKeyMappingRequest(id="mapping-1", key="sk-rotated")

    assert _token_hash_for_update(data) == hash_token("sk-rotated")


def test_update_rejects_both_identifiers():
    from litellm.proxy._types import UpdateJWTKeyMappingRequest

    data = UpdateJWTKeyMappingRequest(id="mapping-1", key="sk-abc", token=_TOKEN_HASH)

    with pytest.raises(HTTPException) as exc_info:
        _token_hash_for_update(data)

    assert exc_info.value.status_code == 400
    assert "at most one" in exc_info.value.detail.lower()


def test_update_rejects_a_token_that_is_not_a_sha256_hash():
    from litellm.proxy._types import UpdateJWTKeyMappingRequest

    data = UpdateJWTKeyMappingRequest(id="mapping-1", token="sk-not-a-hash")

    with pytest.raises(HTTPException) as exc_info:
        _token_hash_for_update(data)

    assert exc_info.value.status_code == 400
    assert "SHA-256" in exc_info.value.detail


# ──────────────────────────────────────────────
# Tests: cache eviction must happen AFTER the DB write commits
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_delete_evicts_cache_after_row_is_gone():
    """A JWT request racing the delete must not keep the removed mapping authorized.

    The DB delete simulates a concurrent request re-caching the mapping mid-write.
    If the endpoint evicts before the delete commits, that repopulated entry
    survives until TTL and the deleted mapping stays usable.
    """
    from litellm.proxy._types import DeleteJWTKeyMappingRequest
    from litellm.proxy.auth.auth_checks import jwt_key_mapping_cache_key

    cache_key = jwt_key_mapping_cache_key("email", "user@example.com")
    user_api_key_cache = DualCache()
    await user_api_key_cache.async_set_cache(key=cache_key, value="hashed_token")

    mock_prisma = _mock_prisma()
    mock_prisma.db.litellm_jwtkeymapping.find_unique.return_value = _mock_mapping()

    async def concurrent_reader_repopulates(**kwargs):
        await user_api_key_cache.async_set_cache(key=cache_key, value="hashed_token")
        return _mock_mapping()

    mock_prisma.db.litellm_jwtkeymapping.delete.side_effect = concurrent_reader_repopulates

    with (
        patch(
            "litellm.proxy.proxy_server.prisma_client", mock_prisma
        ),  # test-quality-ok: proxy_server module global is the endpoint's only injection point
        patch(
            "litellm.proxy.proxy_server.user_api_key_cache", user_api_key_cache
        ),  # test-quality-ok: proxy_server module global is the endpoint's only injection point
    ):
        result = await delete_jwt_key_mapping(
            data=DeleteJWTKeyMappingRequest(id="mapping-1"),
            user_api_key_dict=_make_admin_auth(),
        )

    assert result == {"status": "success"}
    assert await user_api_key_cache.async_get_cache(cache_key) is None


@pytest.mark.asyncio
async def test_update_evicts_old_and_new_cache_keys_after_write():
    """Renaming a mapping's claim must leave neither claim serving stale cache.

    The DB update simulates a concurrent request re-caching the OLD mapping
    mid-write. Both the old claim's entry (would restore the pre-rename token)
    and the new claim's __NO_MAPPING__ sentinel (would 403 the renamed claim)
    must be gone once the endpoint returns.
    """
    from litellm.proxy._types import UpdateJWTKeyMappingRequest
    from litellm.proxy.auth.auth_checks import jwt_key_mapping_cache_key

    old_cache_key = jwt_key_mapping_cache_key("email", "user@example.com")
    new_cache_key = jwt_key_mapping_cache_key("email", "renamed@example.com")
    user_api_key_cache = DualCache()
    await user_api_key_cache.async_set_cache(key=old_cache_key, value="hashed_token")
    await user_api_key_cache.async_set_cache(key=new_cache_key, value="__NO_MAPPING__")

    mock_prisma = _mock_prisma()
    mock_prisma.db.litellm_jwtkeymapping.find_unique.return_value = _mock_mapping()

    async def concurrent_reader_repopulates(**kwargs):
        await user_api_key_cache.async_set_cache(key=old_cache_key, value="hashed_token")
        return _mock_mapping(claim_value="renamed@example.com")

    mock_prisma.db.litellm_jwtkeymapping.update.side_effect = concurrent_reader_repopulates

    with (
        patch(
            "litellm.proxy.proxy_server.prisma_client", mock_prisma
        ),  # test-quality-ok: proxy_server module global is the endpoint's only injection point
        patch(
            "litellm.proxy.proxy_server.user_api_key_cache", user_api_key_cache
        ),  # test-quality-ok: proxy_server module global is the endpoint's only injection point
    ):
        result = await update_jwt_key_mapping(
            data=UpdateJWTKeyMappingRequest(id="mapping-1", jwt_claim_value="renamed@example.com"),
            user_api_key_dict=_make_admin_auth(),
        )

    assert result.jwt_claim_value == "renamed@example.com"
    assert await user_api_key_cache.async_get_cache(old_cache_key) is None
    assert await user_api_key_cache.async_get_cache(new_cache_key) is None
