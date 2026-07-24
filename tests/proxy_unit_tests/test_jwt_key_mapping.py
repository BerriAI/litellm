import pytest
import sys
import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from litellm.proxy.auth.user_api_key_auth import (
    _resolve_jwt_to_virtual_key,
)
from litellm.proxy.auth.handle_jwt import JWTHandler
from litellm.proxy._types import (
    JWTKeyMappingResponse,
    LiteLLM_JWTAuth,
    LitellmUserRoles,
    UserAPIKeyAuth,
)
from litellm.proxy.management_endpoints.jwt_key_mapping_endpoints import (
    _to_response,
    create_jwt_key_mapping,
    delete_jwt_key_mapping,
    info_jwt_key_mapping,
    update_jwt_key_mapping,
)
from litellm.caching.caching import DualCache
from fastapi import HTTPException

# ──────────────────────────────────────────────
# Tests: _resolve_jwt_to_virtual_key
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_jwt_to_virtual_key_mapping_resolution():
    """
    Test that a JWT claim is correctly resolved to a virtual key token.
    """
    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        virtual_key_claim_field="email", virtual_key_mapping_cache_ttl=3600
    )

    jwt_claims = {"email": "user@example.com", "sub": "123"}

    prisma_client = MagicMock()
    prisma_client.db.litellm_jwtkeymapping.find_first = AsyncMock()

    # Mock finding a mapping
    mock_mapping = MagicMock()
    mock_mapping.token = "sk-1234"
    mock_mapping.is_active = True
    prisma_client.db.litellm_jwtkeymapping.find_first.return_value = mock_mapping

    # Mock getting the key object
    mock_key_obj = UserAPIKeyAuth(token="sk-1234", team_id="team1")

    user_api_key_cache = DualCache()

    # Use patch to mock get_key_object in the module where it's used
    with patch(
        "litellm.proxy.auth.resolvers.store.IdentityStore._resolve_key",
        new_callable=AsyncMock,
    ) as mock_get_key:
        mock_get_key.return_value = mock_key_obj

        result = await _resolve_jwt_to_virtual_key(
            jwt_claims=jwt_claims,
            jwt_handler=jwt_handler,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=None,
            proxy_logging_obj=None,
        )

        assert result == mock_key_obj
        prisma_client.db.litellm_jwtkeymapping.find_first.assert_called_once()

        # Test Cache hit
        prisma_client.db.litellm_jwtkeymapping.find_first.reset_mock()
        result_cached = await _resolve_jwt_to_virtual_key(
            jwt_claims=jwt_claims,
            jwt_handler=jwt_handler,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=None,
            proxy_logging_obj=None,
        )
        assert result_cached == mock_key_obj
        prisma_client.db.litellm_jwtkeymapping.find_first.assert_not_called()


@pytest.mark.asyncio
async def test_jwt_to_virtual_key_mapping_no_mapping():
    """
    Test that when no mapping exists, resolve returns None.
    """
    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(virtual_key_claim_field="email")
    jwt_claims = {"email": "unknown@example.com"}

    prisma_client = MagicMock()
    prisma_client.db.litellm_jwtkeymapping.find_first = AsyncMock()
    prisma_client.db.litellm_jwtkeymapping.find_first.return_value = None

    # Mock get_key_object just in case
    with patch(
        "litellm.proxy.auth.resolvers.store.IdentityStore._resolve_key",
        new_callable=AsyncMock,
    ):
        user_api_key_cache = DualCache()

        result = await _resolve_jwt_to_virtual_key(
            jwt_claims=jwt_claims,
            jwt_handler=jwt_handler,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=None,
            proxy_logging_obj=None,
        )

        assert result is None

        # Test Negative Cache hit
        prisma_client.db.litellm_jwtkeymapping.find_first.reset_mock()
        result_cached = await _resolve_jwt_to_virtual_key(
            jwt_claims=jwt_claims,
            jwt_handler=jwt_handler,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=None,
            proxy_logging_obj=None,
        )
        assert result_cached is None
        prisma_client.db.litellm_jwtkeymapping.find_first.assert_not_called()


# ──────────────────────────────────────────────
# Tests: OIDC / JWT routing in user_api_key_auth
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_virtual_key_mapping_oidc_enabled_jwt_token_uses_auth_jwt():
    """
    Regression test for the is_jwt routing fix in user_api_key_auth.py.

    When oidc_userinfo_enabled=True and virtual_key_claim_field is set, but
    the token is a well-formed JWT (3-part header.payload.sig), the virtual-key
    claim lookup must call auth_jwt — not get_oidc_userinfo.
    """
    # Three-part token: is_jwt() returns True
    api_key = "eyJhbGciOiJSUzI1NiJ9.eyJlbWFpbCI6InVzZXJAZXhhbXBsZS5jb20ifQ.sig"

    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        oidc_userinfo_enabled=True,
        virtual_key_claim_field="email",
    )

    # Confirm our fixture token is treated as a JWT
    assert jwt_handler.is_jwt(token=api_key) is True

    auth_jwt_mock = AsyncMock(return_value={"email": "user@example.com", "sub": "123"})
    oidc_userinfo_mock = AsyncMock(return_value={"email": "user@example.com"})

    # Simulate the routing condition from user_api_key_auth.py
    if jwt_handler.litellm_jwtauth.oidc_userinfo_enabled and not jwt_handler.is_jwt(
        token=api_key
    ):
        jwt_claims = await oidc_userinfo_mock(token=api_key)
    else:
        jwt_claims = await auth_jwt_mock(token=api_key)

    auth_jwt_mock.assert_called_once_with(token=api_key)
    oidc_userinfo_mock.assert_not_called()
    assert jwt_claims["email"] == "user@example.com"


@pytest.mark.asyncio
async def test_virtual_key_mapping_oidc_enabled_opaque_token_uses_oidc_userinfo():
    """
    Complement of the test above: when oidc_userinfo_enabled=True and the token
    is an opaque access token (not a JWT), the virtual-key claim lookup must
    call get_oidc_userinfo — not auth_jwt.
    """
    # Opaque token: no dots → is_jwt() returns False
    api_key = "some_opaque_access_token_with_no_dots"

    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        oidc_userinfo_enabled=True,
        virtual_key_claim_field="email",
    )

    assert jwt_handler.is_jwt(token=api_key) is False

    auth_jwt_mock = AsyncMock(return_value={"email": "user@example.com"})
    oidc_userinfo_mock = AsyncMock(
        return_value={"email": "user@example.com", "sub": "123"}
    )

    if jwt_handler.litellm_jwtauth.oidc_userinfo_enabled and not jwt_handler.is_jwt(
        token=api_key
    ):
        jwt_claims = await oidc_userinfo_mock(token=api_key)
    else:
        jwt_claims = await auth_jwt_mock(token=api_key)

    oidc_userinfo_mock.assert_called_once_with(token=api_key)
    auth_jwt_mock.assert_not_called()
    assert jwt_claims["sub"] == "123"


# ──────────────────────────────────────────────
# Tests: _to_response redacts hashed token
# ──────────────────────────────────────────────


def test_to_response_excludes_token():
    """_to_response should not expose the hashed token field."""
    now = datetime.now(timezone.utc)
    mock_mapping = MagicMock()
    mock_mapping.id = "mapping-1"
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
):
    now = datetime.now(timezone.utc)
    m = MagicMock()
    m.id = id
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
    mock_prisma.db.litellm_jwtkeymapping.create.side_effect = Exception(
        "Unique constraint failed (P2002)"
    )
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
            await create_jwt_key_mapping(
                data=data, user_api_key_dict=_make_admin_auth()
            )
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
            await create_jwt_key_mapping(
                data=data, user_api_key_dict=_make_admin_auth()
            )
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
        await create_jwt_key_mapping(
            data=data, user_api_key_dict=_make_non_admin_auth()
        )
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
            await delete_jwt_key_mapping(
                data=data, user_api_key_dict=_make_admin_auth()
            )
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
            await update_jwt_key_mapping(
                data=data, user_api_key_dict=_make_admin_auth()
            )
        assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_info_returns_404_when_not_found():
    """Getting info for non-existent mapping should return 404."""
    mock_prisma = _mock_prisma()
    mock_prisma.db.litellm_jwtkeymapping.find_unique.return_value = None

    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma):
        with pytest.raises(HTTPException) as exc_info:
            await info_jwt_key_mapping(
                id="nonexistent-id", user_api_key_dict=_make_admin_auth()
            )
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
        result = await create_jwt_key_mapping(
            data=data, user_api_key_dict=_make_admin_auth()
        )
        assert isinstance(result, JWTKeyMappingResponse)
        assert "token" not in result.model_fields
        assert result.jwt_claim_name == "email"


# ──────────────────────────────────────────────
# Tests: unregistered_jwt_client_behavior
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reject_behavior_raises_403_on_no_mapping():
    """
    When unregistered_jwt_client_behavior='reject' and no mapping exists,
    _resolve_jwt_to_virtual_key must raise HTTP 403.
    """
    from litellm.proxy._types import UnregisteredJWTClientBehavior

    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        virtual_key_claim_field="email",
        unregistered_jwt_client_behavior=UnregisteredJWTClientBehavior.REJECT,
    )
    jwt_claims = {"email": "unknown@example.com"}

    prisma_client = MagicMock()
    prisma_client.db.litellm_jwtkeymapping.find_first = AsyncMock(return_value=None)

    user_api_key_cache = DualCache()

    with patch(
        "litellm.proxy.auth.resolvers.store.IdentityStore._resolve_key",
        new_callable=AsyncMock,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await _resolve_jwt_to_virtual_key(
                jwt_claims=jwt_claims,
                jwt_handler=jwt_handler,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=None,
                proxy_logging_obj=None,
            )
        assert exc_info.value.status_code == 403
        assert "unknown@example.com" in exc_info.value.detail


@pytest.mark.asyncio
async def test_reject_behavior_caches_sentinel_after_db_miss():
    """
    On a fresh DB miss with REJECT, the __NO_MAPPING__ sentinel must be written
    to cache so that subsequent rejected requests are served from cache and do
    not re-query the DB.
    """
    from litellm.proxy._types import UnregisteredJWTClientBehavior

    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        virtual_key_claim_field="email",
        unregistered_jwt_client_behavior=UnregisteredJWTClientBehavior.REJECT,
        virtual_key_mapping_cache_ttl=300,
    )
    jwt_claims = {"email": "unknown@example.com"}

    prisma_client = MagicMock()
    prisma_client.db.litellm_jwtkeymapping.find_first = AsyncMock(return_value=None)

    user_api_key_cache = DualCache()

    with patch(
        "litellm.proxy.auth.resolvers.store.IdentityStore._resolve_key",
        new_callable=AsyncMock,
    ):
        # First call — DB miss, should raise 403 and write sentinel
        with pytest.raises(HTTPException) as exc_info:
            await _resolve_jwt_to_virtual_key(
                jwt_claims=jwt_claims,
                jwt_handler=jwt_handler,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=None,
                proxy_logging_obj=None,
            )
        assert exc_info.value.status_code == 403

        # Sentinel must now be in cache
        cached = await user_api_key_cache.async_get_cache(
            "jwt_key_mapping:email:unknown@example.com"
        )
        assert cached == "__NO_MAPPING__"

        # Second call — must raise 403 from cache, no additional DB hit
        prisma_client.db.litellm_jwtkeymapping.find_first.reset_mock()
        with pytest.raises(HTTPException) as exc_info2:
            await _resolve_jwt_to_virtual_key(
                jwt_claims=jwt_claims,
                jwt_handler=jwt_handler,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=None,
                proxy_logging_obj=None,
            )
        assert exc_info2.value.status_code == 403
        prisma_client.db.litellm_jwtkeymapping.find_first.assert_not_called()


@pytest.mark.asyncio
async def test_reject_behavior_raises_403_on_cached_no_mapping():
    """
    When the negative-cache sentinel __NO_MAPPING__ is present and behavior is
    'reject', the function must also raise HTTP 403 (not return None silently).
    """
    from litellm.proxy._types import UnregisteredJWTClientBehavior

    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        virtual_key_claim_field="email",
        unregistered_jwt_client_behavior=UnregisteredJWTClientBehavior.REJECT,
    )
    jwt_claims = {"email": "unknown@example.com"}

    prisma_client = MagicMock()
    prisma_client.db.litellm_jwtkeymapping.find_first = AsyncMock(return_value=None)

    # Pre-populate the negative cache so the DB is not hit
    user_api_key_cache = DualCache()
    cache_key = "jwt_key_mapping:email:unknown@example.com"
    await user_api_key_cache.async_set_cache(cache_key, "__NO_MAPPING__")

    with patch(
        "litellm.proxy.auth.resolvers.store.IdentityStore._resolve_key",
        new_callable=AsyncMock,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await _resolve_jwt_to_virtual_key(
                jwt_claims=jwt_claims,
                jwt_handler=jwt_handler,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=None,
                proxy_logging_obj=None,
            )
        assert exc_info.value.status_code == 403
        # DB must NOT have been hit (sentinel served from cache)
        prisma_client.db.litellm_jwtkeymapping.find_first.assert_not_called()


@pytest.mark.asyncio
async def test_auto_register_returns_pending_signal_without_creating_key():
    """
    Security: when unregistered_jwt_client_behavior='auto_register' and no
    mapping exists, _resolve_jwt_to_virtual_key must NOT create the key yet.
    It returns a _PendingAutoRegister signal so the caller can run
    JWTAuthManager.auth_builder (enforcing RBAC, scope mappings,
    custom_validate, user_allowed_email_domain) FIRST. Creating the key here
    would bypass every JWT policy beyond signature verification.
    """
    from litellm.proxy._types import UnregisteredJWTClientBehavior
    from litellm.proxy.auth.user_api_key_auth import _PendingAutoRegister

    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        virtual_key_claim_field="sub",
        unregistered_jwt_client_behavior=UnregisteredJWTClientBehavior.AUTO_REGISTER,
        virtual_key_mapping_cache_ttl=300,
    )
    jwt_claims = {"sub": "new-user-42"}

    prisma_client = MagicMock()
    prisma_client.db.litellm_jwtkeymapping.find_first = AsyncMock(return_value=None)
    prisma_client.db.litellm_jwtkeymapping.create = AsyncMock()

    user_api_key_cache = DualCache()

    with patch(
        "litellm.proxy.management_endpoints.key_management_endpoints.generate_key_helper_fn",
        new_callable=AsyncMock,
    ) as mock_gen_key:
        result = await _resolve_jwt_to_virtual_key(
            jwt_claims=jwt_claims,
            jwt_handler=jwt_handler,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=None,
            proxy_logging_obj=None,
        )

    assert isinstance(result, _PendingAutoRegister)
    assert result.claim_field == "sub"
    assert result.claim_value == "new-user-42"
    assert result.cache_key == "jwt_key_mapping:sub:new-user-42"
    # CRITICAL: no key was created — that must wait until after auth_builder
    mock_gen_key.assert_not_called()
    prisma_client.db.litellm_jwtkeymapping.create.assert_not_called()


@pytest.mark.asyncio
async def test_auto_register_creates_key_and_mapping_when_helper_invoked():
    """
    When the caller invokes _auto_register_jwt_mapping directly (after
    auth_builder validation), the helper creates the key + mapping row and
    returns a UserAPIKeyAuth. The mapping row stores the hashed token (FK to
    LiteLLM_VerificationToken), not the plaintext key.
    """
    from litellm.proxy._types import hash_token
    from litellm.proxy.auth.user_api_key_auth import _auto_register_jwt_mapping

    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        virtual_key_claim_field="sub",
        virtual_key_mapping_cache_ttl=300,
    )

    prisma_client = MagicMock()
    prisma_client.db.litellm_jwtkeymapping.find_first = AsyncMock(return_value=None)
    prisma_client.db.litellm_jwtkeymapping.create = AsyncMock()

    user_api_key_cache = DualCache()
    plaintext_key = "sk-auto-key"
    expected_hash = hash_token(plaintext_key)
    mock_key_obj = UserAPIKeyAuth(token=expected_hash, team_id="validated-team")

    with (
        patch(
            "litellm.proxy.auth.resolvers.store.IdentityStore._resolve_key",
            new_callable=AsyncMock,
        ) as mock_get_key,
        patch(
            "litellm.proxy.management_endpoints.key_management_endpoints.generate_key_helper_fn",
            new_callable=AsyncMock,
        ) as mock_gen_key,
    ):
        mock_gen_key.return_value = {"token": plaintext_key, "key": plaintext_key}
        mock_get_key.return_value = mock_key_obj

        result = await _auto_register_jwt_mapping(
            virtual_key_claim_field="sub",
            claim_value="new-user-42",
            jwt_handler=jwt_handler,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=None,
            proxy_logging_obj=None,
            cache_key="jwt_key_mapping:sub:new-user-42",
            team_id="validated-team",
            user_id="validated-user",
        )

    assert result == mock_key_obj
    # generate_key_helper_fn was passed table_name="key" (not user-upsert path)
    # and the validated team_id + user_id from auth_builder
    assert mock_gen_key.call_args.kwargs["table_name"] == "key"
    assert mock_gen_key.call_args.kwargs["team_id"] == "validated-team"
    assert mock_gen_key.call_args.kwargs["user_id"] == "validated-user"
    # Mapping row was created with the hashed token (FK target)
    call_data = prisma_client.db.litellm_jwtkeymapping.create.call_args[1]["data"]
    assert call_data["jwt_claim_name"] == "sub"
    assert call_data["jwt_claim_value"] == "new-user-42"
    assert call_data["token"] == expected_hash
    cached = await user_api_key_cache.async_get_cache("jwt_key_mapping:sub:new-user-42")
    assert cached == expected_hash


@pytest.mark.asyncio
async def test_auto_register_returns_pending_signal_on_stale_no_mapping_sentinel():
    """
    If the cache holds a stale __NO_MAPPING__ sentinel (written under a prior
    fallback_team_mapping config) and behavior is now AUTO_REGISTER, the
    resolver must evict the sentinel and return _PendingAutoRegister (so the
    caller can run auth_builder before creating the key) — not silently return
    None and not create the key on the spot.
    """
    from litellm.proxy._types import UnregisteredJWTClientBehavior
    from litellm.proxy.auth.user_api_key_auth import _PendingAutoRegister

    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        virtual_key_claim_field="email",
        unregistered_jwt_client_behavior=UnregisteredJWTClientBehavior.AUTO_REGISTER,
        virtual_key_mapping_cache_ttl=300,
    )
    jwt_claims = {"email": "alice@corp.com"}

    prisma_client = MagicMock()
    prisma_client.db.litellm_jwtkeymapping.find_first = AsyncMock(return_value=None)
    prisma_client.db.litellm_jwtkeymapping.create = AsyncMock()

    user_api_key_cache = DualCache()
    await user_api_key_cache.async_set_cache(
        "jwt_key_mapping:email:alice@corp.com", "__NO_MAPPING__"
    )

    with patch(
        "litellm.proxy.management_endpoints.key_management_endpoints.generate_key_helper_fn",
        new_callable=AsyncMock,
    ) as mock_gen_key:
        result = await _resolve_jwt_to_virtual_key(
            jwt_claims=jwt_claims,
            jwt_handler=jwt_handler,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=None,
            proxy_logging_obj=None,
        )

    assert isinstance(result, _PendingAutoRegister)
    # Stale sentinel must be evicted so the deferred auto-register actually
    # runs after auth_builder validates the JWT
    cached_after = await user_api_key_cache.async_get_cache(
        "jwt_key_mapping:email:alice@corp.com"
    )
    assert cached_after is None
    mock_gen_key.assert_not_called()
    prisma_client.db.litellm_jwtkeymapping.create.assert_not_called()


@pytest.mark.asyncio
async def test_auto_register_race_condition_unique_conflict():
    """
    If two concurrent requests both call _auto_register_jwt_mapping and the
    second hits a unique-constraint violation on create, it must:
      1) delete the orphaned virtual key it just created (so orphans don't
         accumulate in LiteLLM_VerificationToken under sustained concurrency),
      2) fall back to the winner's mapping,
      3) not surface an error.
    """
    from litellm.proxy.auth.user_api_key_auth import _auto_register_jwt_mapping
    from litellm.proxy._types import UnregisteredJWTClientBehavior, hash_token

    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        virtual_key_claim_field="sub",
        unregistered_jwt_client_behavior=UnregisteredJWTClientBehavior.AUTO_REGISTER,
        virtual_key_mapping_cache_ttl=300,
    )

    prisma_client = MagicMock()
    prisma_client.db.litellm_jwtkeymapping.create = AsyncMock(
        side_effect=Exception("Unique constraint failed (P2002)")
    )
    prisma_client.db.litellm_verificationtoken.delete = AsyncMock()
    # Simulate the winner's mapping already in DB after the conflict
    winner_mapping = MagicMock()
    winner_mapping.token = "winner_token_hash"
    winner_mapping.is_active = True
    prisma_client.db.litellm_jwtkeymapping.find_first = AsyncMock(
        return_value=winner_mapping
    )

    user_api_key_cache = DualCache()
    loser_plaintext = "sk-loser"
    loser_hash = hash_token(loser_plaintext)
    mock_key_obj = UserAPIKeyAuth(token="winner_token_hash", team_id=None)

    with (
        patch(
            "litellm.proxy.auth.resolvers.store.IdentityStore._resolve_key",
            new_callable=AsyncMock,
        ) as mock_get_key,
        patch(
            "litellm.proxy.management_endpoints.key_management_endpoints.generate_key_helper_fn",
            new_callable=AsyncMock,
            return_value={"token": loser_plaintext, "key": loser_plaintext},
        ),
    ):
        mock_get_key.return_value = mock_key_obj

        result = await _auto_register_jwt_mapping(
            virtual_key_claim_field="sub",
            claim_value="user-42",
            jwt_handler=jwt_handler,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=None,
            proxy_logging_obj=None,
            cache_key="jwt_key_mapping:sub:user-42",
        )

    assert result == mock_key_obj
    # The orphaned loser key must be deleted from LiteLLM_VerificationToken
    prisma_client.db.litellm_verificationtoken.delete.assert_called_once_with(
        where={"token": loser_hash}
    )
    # Cache should hold the winner's token, not the loser's
    cached = await user_api_key_cache.async_get_cache("jwt_key_mapping:sub:user-42")
    assert cached == "winner_token_hash"
    mock_get_key.assert_called_once_with("winner_token_hash")


# ──────────────────────────────────────────────
# Tests: prisma_client=None does not bypass no-match policy
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reject_behavior_enforced_when_prisma_client_is_none():
    """
    When prisma_client is None and behavior is REJECT, a 403 must be raised —
    not silently fallen through to team auth.
    """
    from litellm.proxy._types import UnregisteredJWTClientBehavior

    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        virtual_key_claim_field="email",
        unregistered_jwt_client_behavior=UnregisteredJWTClientBehavior.REJECT,
    )
    jwt_claims = {"email": "unknown@example.com"}

    user_api_key_cache = DualCache()

    with pytest.raises(HTTPException) as exc_info:
        await _resolve_jwt_to_virtual_key(
            jwt_claims=jwt_claims,
            jwt_handler=jwt_handler,
            prisma_client=None,  # no DB
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=None,
            proxy_logging_obj=None,
        )
    assert exc_info.value.status_code == 403
    assert "unknown@example.com" in exc_info.value.detail


@pytest.mark.asyncio
async def test_reject_raises_403_when_claim_field_missing_from_jwt():
    """
    Security: a JWT that omits the configured virtual_key_claim_field must NOT
    bypass the REJECT policy. Previously the early `if claim_value is None:
    return None` branch ran before the policy check, letting a caller who knows
    the configured claim-field name silently fall through to team-based auth.
    """
    from litellm.proxy._types import UnregisteredJWTClientBehavior

    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        virtual_key_claim_field="sub",
        unregistered_jwt_client_behavior=UnregisteredJWTClientBehavior.REJECT,
    )
    # JWT does NOT contain "sub"
    jwt_claims = {"email": "user@example.com"}

    with pytest.raises(HTTPException) as exc_info:
        await _resolve_jwt_to_virtual_key(
            jwt_claims=jwt_claims,
            jwt_handler=jwt_handler,
            prisma_client=MagicMock(),
            user_api_key_cache=DualCache(),
            parent_otel_span=None,
            proxy_logging_obj=None,
        )
    assert exc_info.value.status_code == 403
    assert "'sub'" in exc_info.value.detail
    assert "missing from the JWT" in exc_info.value.detail


@pytest.mark.asyncio
async def test_auto_register_raises_403_when_claim_field_missing_from_jwt():
    """
    AUTO_REGISTER cannot create a mapping without a stable identity. When the
    configured claim field is missing from the JWT, return 403 rather than
    silently falling through (which would bypass the unregistered-client policy)
    or creating a sentinel-keyed record.
    """
    from litellm.proxy._types import UnregisteredJWTClientBehavior

    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        virtual_key_claim_field="sub",
        unregistered_jwt_client_behavior=UnregisteredJWTClientBehavior.AUTO_REGISTER,
    )
    jwt_claims = {"email": "user@example.com"}

    with pytest.raises(HTTPException) as exc_info:
        await _resolve_jwt_to_virtual_key(
            jwt_claims=jwt_claims,
            jwt_handler=jwt_handler,
            prisma_client=MagicMock(),
            user_api_key_cache=DualCache(),
            parent_otel_span=None,
            proxy_logging_obj=None,
        )
    assert exc_info.value.status_code == 403
    assert "missing from the JWT" in exc_info.value.detail


@pytest.mark.asyncio
async def test_fallback_team_mapping_returns_none_when_claim_field_missing_from_jwt():
    """
    Under FALLBACK_TEAM_MAPPING (the default, backward-compatible mode), a JWT
    without the configured claim field must still fall through to team-based
    JWT auth — not raise. This preserves the pre-existing contract.
    """
    from litellm.proxy._types import UnregisteredJWTClientBehavior

    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        virtual_key_claim_field="sub",
        unregistered_jwt_client_behavior=UnregisteredJWTClientBehavior.FALLBACK_TEAM_MAPPING,
    )
    jwt_claims = {"email": "user@example.com"}

    result = await _resolve_jwt_to_virtual_key(
        jwt_claims=jwt_claims,
        jwt_handler=jwt_handler,
        prisma_client=MagicMock(),
        user_api_key_cache=DualCache(),
        parent_otel_span=None,
        proxy_logging_obj=None,
    )
    assert result is None


@pytest.mark.asyncio
async def test_fallback_team_mapping_returns_none_when_prisma_client_is_none():
    """
    When prisma_client is None and behavior is FALLBACK_TEAM_MAPPING, the
    function must return None (fall through to team auth) — not raise.
    """
    from litellm.proxy._types import UnregisteredJWTClientBehavior

    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        virtual_key_claim_field="email",
        unregistered_jwt_client_behavior=UnregisteredJWTClientBehavior.FALLBACK_TEAM_MAPPING,
    )
    jwt_claims = {"email": "anyone@example.com"}

    result = await _resolve_jwt_to_virtual_key(
        jwt_claims=jwt_claims,
        jwt_handler=jwt_handler,
        prisma_client=None,
        user_api_key_cache=DualCache(),
        parent_otel_span=None,
        proxy_logging_obj=None,
    )
    assert result is None


@pytest.mark.asyncio
async def test_auto_register_raises_500_when_prisma_client_is_none():
    """
    AUTO_REGISTER without a DB connection must raise HTTP 500 with a clear
    message — it cannot create keys without a database.
    """
    from litellm.proxy._types import UnregisteredJWTClientBehavior

    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        virtual_key_claim_field="sub",
        unregistered_jwt_client_behavior=UnregisteredJWTClientBehavior.AUTO_REGISTER,
    )
    jwt_claims = {"sub": "new-user-42"}

    with pytest.raises(HTTPException) as exc_info:
        await _resolve_jwt_to_virtual_key(
            jwt_claims=jwt_claims,
            jwt_handler=jwt_handler,
            prisma_client=None,
            user_api_key_cache=DualCache(),
            parent_otel_span=None,
            proxy_logging_obj=None,
        )
    assert exc_info.value.status_code == 500
    assert "AUTO_REGISTER requires a database" in exc_info.value.detail


@pytest.mark.asyncio
async def test_auto_register_raises_500_when_sentinel_cached_and_no_db():
    """
    AUTO_REGISTER + cached __NO_MAPPING__ sentinel + prisma_client is None must
    raise HTTP 500, matching the fresh-path behavior. Previously this path
    silently returned None and let the request fall through to team auth,
    creating different access-control outcomes under identical configuration.
    """
    from litellm.proxy._types import UnregisteredJWTClientBehavior

    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        virtual_key_claim_field="sub",
        unregistered_jwt_client_behavior=UnregisteredJWTClientBehavior.AUTO_REGISTER,
        virtual_key_mapping_cache_ttl=300,
    )
    jwt_claims = {"sub": "user-42"}

    user_api_key_cache = DualCache()
    # Stale sentinel written under a prior fallback_team_mapping config
    await user_api_key_cache.async_set_cache(
        "jwt_key_mapping:sub:user-42", "__NO_MAPPING__"
    )

    with pytest.raises(HTTPException) as exc_info:
        await _resolve_jwt_to_virtual_key(
            jwt_claims=jwt_claims,
            jwt_handler=jwt_handler,
            prisma_client=None,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=None,
            proxy_logging_obj=None,
        )
    assert exc_info.value.status_code == 500
    assert "AUTO_REGISTER requires a database" in exc_info.value.detail


@pytest.mark.asyncio
async def test_auto_register_race_conflict_tolerates_delete_failure():
    """
    If deleting the orphaned virtual key after a race-condition conflict fails
    (e.g. transient DB error), the request must still succeed by returning the
    winner's mapping — the orphan is unmapped and inert.
    """
    from litellm.proxy.auth.user_api_key_auth import _auto_register_jwt_mapping
    from litellm.proxy._types import UnregisteredJWTClientBehavior

    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        virtual_key_claim_field="sub",
        unregistered_jwt_client_behavior=UnregisteredJWTClientBehavior.AUTO_REGISTER,
        virtual_key_mapping_cache_ttl=300,
    )

    prisma_client = MagicMock()
    prisma_client.db.litellm_jwtkeymapping.create = AsyncMock(
        side_effect=Exception("Unique constraint failed (P2002)")
    )
    prisma_client.db.litellm_verificationtoken.delete = AsyncMock(
        side_effect=Exception("transient DB error")
    )
    winner_mapping = MagicMock()
    winner_mapping.token = "winner_token_hash"
    winner_mapping.is_active = True
    prisma_client.db.litellm_jwtkeymapping.find_first = AsyncMock(
        return_value=winner_mapping
    )

    user_api_key_cache = DualCache()
    mock_key_obj = UserAPIKeyAuth(token="winner_token_hash", team_id=None)

    with (
        patch(
            "litellm.proxy.auth.resolvers.store.IdentityStore._resolve_key",
            new_callable=AsyncMock,
        ) as mock_get_key,
        patch(
            "litellm.proxy.management_endpoints.key_management_endpoints.generate_key_helper_fn",
            new_callable=AsyncMock,
            return_value={"token": "sk-loser", "key": "sk-loser"},
        ),
    ):
        mock_get_key.return_value = mock_key_obj

        result = await _auto_register_jwt_mapping(
            virtual_key_claim_field="sub",
            claim_value="user-42",
            jwt_handler=jwt_handler,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=None,
            proxy_logging_obj=None,
            cache_key="jwt_key_mapping:sub:user-42",
        )

    # Caller still receives the winner's mapping even when cleanup fails
    assert result == mock_key_obj
    prisma_client.db.litellm_verificationtoken.delete.assert_called_once()


@pytest.mark.asyncio
async def test_auto_register_raises_503_when_winner_mapping_vanishes():
    """
    Race edge case: this request loses the unique-constraint race, deletes its
    orphan, then refetches the winner's mapping — but the winner's row was
    concurrently deleted. Previously this returned None, silently falling
    through to less-restrictive team-based JWT auth (bypassing the configured
    AUTO_REGISTER policy). Must now raise HTTP 503 so the caller retries
    rather than getting unintended fallback access.
    """
    from litellm.proxy.auth.user_api_key_auth import _auto_register_jwt_mapping
    from litellm.proxy._types import UnregisteredJWTClientBehavior

    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        virtual_key_claim_field="sub",
        unregistered_jwt_client_behavior=UnregisteredJWTClientBehavior.AUTO_REGISTER,
        virtual_key_mapping_cache_ttl=300,
    )

    prisma_client = MagicMock()
    prisma_client.db.litellm_jwtkeymapping.create = AsyncMock(
        side_effect=Exception("Unique constraint failed (P2002)")
    )
    prisma_client.db.litellm_verificationtoken.delete = AsyncMock()
    # Winner row no longer exists by the time we refetch
    prisma_client.db.litellm_jwtkeymapping.find_first = AsyncMock(return_value=None)

    user_api_key_cache = DualCache()

    with (
        patch(
            "litellm.proxy.management_endpoints.key_management_endpoints.generate_key_helper_fn",
            new_callable=AsyncMock,
            return_value={"token": "sk-loser", "key": "sk-loser"},
        ),
        pytest.raises(HTTPException) as exc_info,
    ):
        await _auto_register_jwt_mapping(
            virtual_key_claim_field="sub",
            claim_value="user-42",
            jwt_handler=jwt_handler,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=None,
            proxy_logging_obj=None,
            cache_key="jwt_key_mapping:sub:user-42",
        )

    assert exc_info.value.status_code == 503
    assert "concurrently removed" in exc_info.value.detail


@pytest.mark.asyncio
async def test_proxy_admin_sentinel_skips_db_lookup_on_cache_hit():
    """
    When the cache holds the proxy-admin sentinel (written after a prior
    request's is_proxy_admin early-return), _resolve_jwt_to_virtual_key must
    return None *without* hitting the DB. Caller proceeds to auth_builder.

    Without this, every subsequent proxy-admin request under AUTO_REGISTER
    would re-query get_jwt_key_mapping_object — a cache-miss regression
    introduced by the deferred-auto-register refactor.
    """
    from litellm.proxy._types import UnregisteredJWTClientBehavior

    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        virtual_key_claim_field="sub",
        unregistered_jwt_client_behavior=UnregisteredJWTClientBehavior.AUTO_REGISTER,
        virtual_key_mapping_cache_ttl=300,
    )
    jwt_claims = {"sub": "admin-user"}

    prisma_client = MagicMock()
    # Will fail the test if accessed — proves the sentinel short-circuits DB
    prisma_client.db.litellm_jwtkeymapping.find_first = AsyncMock(
        side_effect=AssertionError("DB must not be hit when sentinel is cached")
    )

    user_api_key_cache = DualCache()
    await user_api_key_cache.async_set_cache(
        "jwt_key_mapping:sub:admin-user", "__JWT_PROXY_ADMIN__"
    )

    result = await _resolve_jwt_to_virtual_key(
        jwt_claims=jwt_claims,
        jwt_handler=jwt_handler,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        parent_otel_span=None,
        proxy_logging_obj=None,
    )

    assert result is None
    prisma_client.db.litellm_jwtkeymapping.find_first.assert_not_called()


# ──────────────────────────────────────────────
# Tests: AUTO_REGISTER stamps validated identity from auth_builder
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_auto_register_helper_stamps_validated_identity_context():
    """
    The deferred-auto-register contract: _auto_register_jwt_mapping is called
    with identity fields from JWTAuthManager.auth_builder's *validated*
    result (after RBAC, scope mappings, custom_validate, email-domain policy).
    These must be passed to generate_key_helper_fn so the created key carries
    them — the cached future-request path then inherits the same team/user/org
    limits the auth_builder path would have applied.
    """
    from litellm.proxy.auth.user_api_key_auth import _auto_register_jwt_mapping

    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        virtual_key_claim_field="sub",
        virtual_key_mapping_cache_ttl=300,
    )

    prisma_client = MagicMock()
    prisma_client.db.litellm_jwtkeymapping.create = AsyncMock()
    mock_key_obj = UserAPIKeyAuth(
        token="hashed", team_id="validated-team", user_id="validated-user"
    )

    with (
        patch(
            "litellm.proxy.auth.resolvers.store.IdentityStore._resolve_key",
            new_callable=AsyncMock,
        ) as mock_get_key,
        patch(
            "litellm.proxy.management_endpoints.key_management_endpoints.generate_key_helper_fn",
            new_callable=AsyncMock,
        ) as mock_gen_key,
    ):
        mock_gen_key.return_value = {"token": "sk-newkey", "key": "sk-newkey"}
        mock_get_key.return_value = mock_key_obj

        result = await _auto_register_jwt_mapping(
            virtual_key_claim_field="sub",
            claim_value="new-user",
            jwt_handler=jwt_handler,
            prisma_client=prisma_client,
            user_api_key_cache=DualCache(),
            parent_otel_span=None,
            proxy_logging_obj=None,
            cache_key="jwt_key_mapping:sub:new-user",
            team_id="validated-team",
            user_id="validated-user",
            org_id="validated-org",
            end_user_id="validated-end-user",
        )

    assert result == mock_key_obj
    assert mock_gen_key.call_args.kwargs["team_id"] == "validated-team"
    assert mock_gen_key.call_args.kwargs["user_id"] == "validated-user"
    assert mock_gen_key.call_args.kwargs["organization_id"] == "validated-org"
    assert result.org_id == "validated-org"
    assert result.end_user_id == "validated-end-user"


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
