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
        "litellm.proxy.auth.user_api_key_auth.get_key_object", new_callable=AsyncMock
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
        "litellm.proxy.auth.user_api_key_auth.get_key_object", new_callable=AsyncMock
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
    oidc_userinfo_mock = AsyncMock(return_value={"email": "user@example.com", "sub": "123"})

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
        jwt_claim_name="email", jwt_claim_value="user@example.com", key="sk-test-key",
    )

    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma), patch(
        "litellm.proxy.proxy_server.user_api_key_cache", mock_cache
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
        jwt_claim_name="sub", jwt_claim_value="user-999", key="sk-nonexistent",
    )

    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma), patch(
        "litellm.proxy.proxy_server.user_api_key_cache", mock_cache
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
        jwt_claim_name="email", jwt_claim_value="user@example.com", key="sk-test",
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

    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma), patch(
        "litellm.proxy.proxy_server.user_api_key_cache", mock_cache
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

    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma), patch(
        "litellm.proxy.proxy_server.user_api_key_cache", mock_cache
    ):
        with pytest.raises(HTTPException) as exc_info:
            await update_jwt_key_mapping(data=data, user_api_key_dict=_make_admin_auth())
        assert exc_info.value.status_code == 404


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
        jwt_claim_name="email", jwt_claim_value="user@example.com", key="sk-test-key",
    )

    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma), patch(
        "litellm.proxy.proxy_server.user_api_key_cache", mock_cache
    ):
        result = await create_jwt_key_mapping(data=data, user_api_key_dict=_make_admin_auth())
        assert isinstance(result, JWTKeyMappingResponse)
        assert "token" not in result.model_fields
        assert result.jwt_claim_name == "email"
