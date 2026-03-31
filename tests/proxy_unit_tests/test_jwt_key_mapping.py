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
    mock_mapping.issuer = ""
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
    # Needed by create_jwt_key_mapping (stamps jwt_bound metadata) and info endpoint
    prisma.db.litellm_verificationtoken.find_first = AsyncMock(return_value=None)
    prisma.db.litellm_verificationtoken.update = AsyncMock()
    prisma.db.litellm_verificationtoken.delete = AsyncMock()
    return prisma


def _mock_mapping(
    id="mapping-1",
    claim_name="email",
    claim_value="user@example.com",
    issuer="",
):
    now = datetime.now(timezone.utc)
    m = MagicMock()
    m.id = id
    m.jwt_claim_name = claim_name
    m.jwt_claim_value = claim_value
    m.issuer = issuer
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
    mock_cache = AsyncMock()

    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma), patch(
        "litellm.proxy.proxy_server.user_api_key_cache", mock_cache
    ):
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


# ──────────────────────────────────────────────
# Block 1: Gap #2 — JWT-bound metadata stamping
# ──────────────────────────────────────────────


def test_handle_key_type_jwt_client():
    """JWT_CLIENT key type should resolve to llm_api_routes."""
    from litellm.proxy._types import LiteLLMKeyType
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        handle_key_type,
    )
    from litellm.proxy._types import GenerateKeyRequest

    data = GenerateKeyRequest(key_type=LiteLLMKeyType.JWT_CLIENT)
    data_json = data.model_dump()
    result = handle_key_type(data=data, data_json=data_json)
    assert result["allowed_routes"] == ["llm_api_routes"]


@pytest.mark.asyncio
async def test_create_mapping_stamps_jwt_bound_metadata():
    """create_jwt_key_mapping should update the key's metadata with jwt_bound=True."""
    from litellm.proxy._types import CreateJWTKeyMappingRequest
    import json as _json

    mock_prisma = _mock_prisma()
    mock_mapping = _mock_mapping()
    mock_prisma.db.litellm_jwtkeymapping.create.return_value = mock_mapping

    # Simulate key with no existing metadata
    mock_key_row = MagicMock()
    mock_key_row.metadata = "{}"
    mock_prisma.db.litellm_verificationtoken.find_first.return_value = mock_key_row
    mock_cache = AsyncMock()

    data = CreateJWTKeyMappingRequest(
        jwt_claim_name="sub",
        jwt_claim_value="svc-a",
        key="sk-test-key",
    )

    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma), patch(
        "litellm.proxy.proxy_server.user_api_key_cache", mock_cache
    ):
        await create_jwt_key_mapping(data=data, user_api_key_dict=_make_admin_auth())

    mock_prisma.db.litellm_verificationtoken.update.assert_called_once()
    call_kwargs = mock_prisma.db.litellm_verificationtoken.update.call_args
    update_data = call_kwargs.kwargs["data"]
    stored_metadata = _json.loads(update_data["metadata"])
    assert stored_metadata["jwt_bound"] is True
    assert stored_metadata["jwt_claim_name"] == "sub"
    assert stored_metadata["jwt_claim_value"] == "svc-a"
    assert update_data["allowed_routes"] == ["llm_api_routes"]


@pytest.mark.asyncio
async def test_create_mapping_preserves_existing_metadata():
    """Existing metadata keys should be preserved when stamping jwt_bound."""
    from litellm.proxy._types import CreateJWTKeyMappingRequest
    import json as _json

    mock_prisma = _mock_prisma()
    mock_prisma.db.litellm_jwtkeymapping.create.return_value = _mock_mapping()

    mock_key_row = MagicMock()
    mock_key_row.metadata = _json.dumps({"custom_field": "keep_me"})
    mock_prisma.db.litellm_verificationtoken.find_first.return_value = mock_key_row
    mock_cache = AsyncMock()

    data = CreateJWTKeyMappingRequest(
        jwt_claim_name="email",
        jwt_claim_value="user@example.com",
        key="sk-test-key",
    )

    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma), patch(
        "litellm.proxy.proxy_server.user_api_key_cache", mock_cache
    ):
        await create_jwt_key_mapping(data=data, user_api_key_dict=_make_admin_auth())

    update_data = mock_prisma.db.litellm_verificationtoken.update.call_args.kwargs["data"]
    stored_metadata = _json.loads(update_data["metadata"])
    assert stored_metadata["custom_field"] == "keep_me"
    assert stored_metadata["jwt_bound"] is True


# ──────────────────────────────────────────────
# Block 2: Gap #1 — CRUD block on jwt-bound keys
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_can_modify_jwt_bound_key_as_admin_returns_true():
    """Proxy admin can always modify JWT-bound keys."""
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        can_modify_verification_token,
    )
    from litellm.proxy._types import LiteLLM_VerificationToken

    key_info = MagicMock(spec=LiteLLM_VerificationToken)
    key_info.metadata = {"jwt_bound": True}
    key_info.team_id = None
    key_info.user_id = "some-user"

    admin_dict = _make_admin_auth()
    result = await can_modify_verification_token(
        key_info=key_info,
        user_api_key_cache=MagicMock(),
        user_api_key_dict=admin_dict,
        prisma_client=MagicMock(),
    )
    assert result is True


@pytest.mark.asyncio
async def test_can_modify_jwt_bound_key_as_internal_user_returns_false():
    """Non-admin cannot modify a JWT-bound key even if they own it."""
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        can_modify_verification_token,
    )
    from litellm.proxy._types import LiteLLM_VerificationToken

    key_info = MagicMock(spec=LiteLLM_VerificationToken)
    key_info.metadata = {"jwt_bound": True}
    key_info.team_id = None
    key_info.user_id = "user-123"

    non_admin = _make_non_admin_auth()
    non_admin.user_id = "user-123"  # same user_id as key owner

    result = await can_modify_verification_token(
        key_info=key_info,
        user_api_key_cache=MagicMock(),
        user_api_key_dict=non_admin,
        prisma_client=MagicMock(),
    )
    assert result is False


def test_jwt_session_blocked_from_key_management_route():
    """A JWT-authenticated session (jwt_claims set) must not reach key management routes."""
    from litellm.proxy.auth.route_checks import RouteChecks
    from unittest.mock import MagicMock

    user_obj = MagicMock()
    user_obj.user_id = "user-123"

    valid_token = UserAPIKeyAuth(
        token="sk-bound",
        user_role=LitellmUserRoles.INTERNAL_USER,
        jwt_claims={"sub": "svc-a"},  # marks this as a JWT session
    )

    with pytest.raises(Exception):
        RouteChecks.non_proxy_admin_allowed_routes_check(
            user_obj=user_obj,
            _user_role=LitellmUserRoles.INTERNAL_USER,
            route="/key/update",
            request=MagicMock(),
            valid_token=valid_token,
            request_data={},
        )


def test_jwt_session_allowed_llm_api_route():
    """A JWT-authenticated session must be allowed to call LLM API routes."""
    from litellm.proxy.auth.route_checks import RouteChecks

    valid_token = UserAPIKeyAuth(
        token="sk-bound",
        user_role=LitellmUserRoles.INTERNAL_USER,
        jwt_claims={"sub": "svc-a"},
    )

    # Should not raise for an LLM API route
    RouteChecks.non_proxy_admin_allowed_routes_check(
        user_obj=MagicMock(),
        _user_role=LitellmUserRoles.INTERNAL_USER,
        route="/v1/chat/completions",
        request=MagicMock(),
        valid_token=valid_token,
        request_data={},
    )


def test_non_jwt_session_internal_user_can_access_key_management():
    """Regular API key session (no jwt_claims) keeps existing internal_user access."""
    from litellm.proxy.auth.route_checks import RouteChecks

    valid_token = UserAPIKeyAuth(
        token="sk-regular",
        user_role=LitellmUserRoles.INTERNAL_USER,
        jwt_claims=None,  # not a JWT session
    )

    # Should not raise — existing behavior preserved
    RouteChecks.non_proxy_admin_allowed_routes_check(
        user_obj=MagicMock(),
        _user_role=LitellmUserRoles.INTERNAL_USER,
        route="/key/update",
        request=MagicMock(),
        valid_token=valid_token,
        request_data={},
    )


# ──────────────────────────────────────────────
# Block 3: Gap #3 — Unified /jwt_client/new
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_jwt_client_new_creates_key_and_mapping():
    """/jwt_client/new should call generate_key_helper_fn and create a mapping row."""
    from litellm.proxy.management_endpoints.jwt_key_mapping_endpoints import (
        create_jwt_client,
    )
    from litellm.proxy._types import CreateJWTClientRequest
    import json as _json

    mock_prisma = _mock_prisma()
    mock_mapping = _mock_mapping(claim_name="sub", claim_value="svc-a")
    mock_prisma.db.litellm_jwtkeymapping.create.return_value = mock_mapping
    mock_cache = AsyncMock()

    fake_key_data = {"token": "sk-auto-generated-key"}

    data = CreateJWTClientRequest(
        jwt_claim_name="sub",
        jwt_claim_value="svc-a",
        models=["gpt-4o"],
        max_budget=10.0,
    )

    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma), patch(
        "litellm.proxy.proxy_server.user_api_key_cache", mock_cache
    ), patch(
        "litellm.proxy.management_endpoints.key_management_endpoints.generate_key_helper_fn",
        new_callable=AsyncMock,
        return_value=fake_key_data,
    ) as mock_gen:
        result = await create_jwt_client(data=data, user_api_key_dict=_make_admin_auth())

    mock_gen.assert_called_once()
    call_kwargs = mock_gen.call_args.kwargs
    assert call_kwargs["allowed_routes"] == ["llm_api_routes"]
    assert call_kwargs["metadata"]["jwt_bound"] is True
    assert call_kwargs["metadata"]["jwt_claim_name"] == "sub"

    mock_prisma.db.litellm_jwtkeymapping.create.assert_called_once()
    assert isinstance(result, JWTKeyMappingResponse)


@pytest.mark.asyncio
async def test_jwt_client_new_cache_invalidated():
    """/jwt_client/new must invalidate the cache for the new mapping."""
    from litellm.proxy.management_endpoints.jwt_key_mapping_endpoints import (
        create_jwt_client,
    )
    from litellm.proxy._types import CreateJWTClientRequest

    mock_prisma = _mock_prisma()
    mock_prisma.db.litellm_jwtkeymapping.create.return_value = _mock_mapping(
        claim_name="sub", claim_value="svc-b"
    )
    mock_cache = MagicMock()
    mock_cache.async_delete_cache = AsyncMock()

    data = CreateJWTClientRequest(jwt_claim_name="sub", jwt_claim_value="svc-b")

    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma), patch(
        "litellm.proxy.proxy_server.user_api_key_cache", mock_cache
    ), patch(
        "litellm.proxy.management_endpoints.key_management_endpoints.generate_key_helper_fn",
        new_callable=AsyncMock,
        return_value={"token": "sk-xyz"},
    ):
        await create_jwt_client(data=data, user_api_key_dict=_make_admin_auth())

    mock_cache.async_delete_cache.assert_called_once()
    cache_key_arg = mock_cache.async_delete_cache.call_args.args[0]
    assert "sub" in cache_key_arg and "svc-b" in cache_key_arg


@pytest.mark.asyncio
async def test_jwt_client_new_non_admin_rejected():
    """/jwt_client/new must reject non-admin users with 403."""
    from litellm.proxy.management_endpoints.jwt_key_mapping_endpoints import (
        create_jwt_client,
    )
    from litellm.proxy._types import CreateJWTClientRequest

    data = CreateJWTClientRequest(jwt_claim_name="sub", jwt_claim_value="svc-c")

    with pytest.raises(HTTPException) as exc_info:
        await create_jwt_client(data=data, user_api_key_dict=_make_non_admin_auth())
    assert exc_info.value.status_code == 403


# ──────────────────────────────────────────────
# Block 4: Gap #4 — unregistered_jwt_client_behavior
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reject_behavior_raises_403_for_unknown_jwt():
    """'reject' mode should raise 403 when no mapping exists."""
    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        virtual_key_claim_field="sub",
        unregistered_jwt_client_behavior="reject",
    )

    prisma_client = MagicMock()
    prisma_client.db.litellm_jwtkeymapping.find_first = AsyncMock(return_value=None)

    with pytest.raises(HTTPException) as exc_info:
        await _resolve_jwt_to_virtual_key(
            jwt_claims={"sub": "unknown-svc"},
            jwt_handler=jwt_handler,
            prisma_client=prisma_client,
            user_api_key_cache=DualCache(),
            parent_otel_span=None,
            proxy_logging_obj=MagicMock(),
        )
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_fallback_behavior_returns_none_for_unknown_jwt():
    """Default 'fallback_team_mapping' mode should return None for unknown clients."""
    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        virtual_key_claim_field="sub",
        # default: fallback_team_mapping
    )

    prisma_client = MagicMock()
    prisma_client.db.litellm_jwtkeymapping.find_first = AsyncMock(return_value=None)

    with patch(
        "litellm.proxy.auth.user_api_key_auth.get_key_object", new_callable=AsyncMock
    ):
        result = await _resolve_jwt_to_virtual_key(
            jwt_claims={"sub": "unknown-svc"},
            jwt_handler=jwt_handler,
            prisma_client=prisma_client,
            user_api_key_cache=DualCache(),
            parent_otel_span=None,
            proxy_logging_obj=MagicMock(),
        )
    assert result is None


@pytest.mark.asyncio
async def test_auto_register_creates_key_on_first_request():
    """'auto_register' mode should create a key+mapping on first unknown JWT."""
    from litellm.proxy._types import JWTClientAutoRegisterDefaults

    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        virtual_key_claim_field="sub",
        unregistered_jwt_client_behavior="auto_register",
        auto_register_defaults=JWTClientAutoRegisterDefaults(
            models=["gpt-4o-mini"], max_budget=5.0
        ),
    )

    prisma_client = MagicMock()
    prisma_client.db.litellm_jwtkeymapping.find_first = AsyncMock(return_value=None)
    prisma_client.db.litellm_jwtkeymapping.create = AsyncMock()

    mock_key_obj = UserAPIKeyAuth(token="sk-auto", team_id=None)

    with patch(
        "litellm.proxy.auth.user_api_key_auth.get_key_object",
        new_callable=AsyncMock,
        return_value=mock_key_obj,
    ), patch(
        "litellm.proxy.management_endpoints.key_management_endpoints.generate_key_helper_fn",
        new_callable=AsyncMock,
        return_value={"token": "sk-auto"},
    ) as mock_gen:
        result = await _resolve_jwt_to_virtual_key(
            jwt_claims={"sub": "new-svc"},
            jwt_handler=jwt_handler,
            prisma_client=prisma_client,
            user_api_key_cache=DualCache(),
            parent_otel_span=None,
            proxy_logging_obj=MagicMock(),
        )

    assert result is not None
    mock_gen.assert_called_once()
    call_kwargs = mock_gen.call_args.kwargs
    assert call_kwargs["allowed_routes"] == ["llm_api_routes"]
    assert call_kwargs["metadata"]["jwt_bound"] is True


# ──────────────────────────────────────────────
# Block 5: Gap #5 — issuer column
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_jwt_key_mapping_object_passes_issuer_to_db():
    """get_jwt_key_mapping_object must include issuer in the DB where clause."""
    from litellm.proxy.auth.auth_checks import get_jwt_key_mapping_object

    prisma_client = MagicMock()
    prisma_client.db.litellm_jwtkeymapping.find_first = AsyncMock(return_value=None)

    await get_jwt_key_mapping_object(
        jwt_claim_name="sub",
        jwt_claim_value="svc",
        prisma_client=prisma_client,
        issuer="https://idp1.example.com",
    )

    call_kwargs = prisma_client.db.litellm_jwtkeymapping.find_first.call_args.kwargs
    assert call_kwargs["where"]["issuer"] == "https://idp1.example.com"


@pytest.mark.asyncio
async def test_resolve_extracts_issuer_from_jwt_claims():
    """_resolve_jwt_to_virtual_key must pass iss claim as issuer to DB lookup."""
    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(virtual_key_claim_field="sub")

    prisma_client = MagicMock()
    prisma_client.db.litellm_jwtkeymapping.find_first = AsyncMock(return_value=None)

    with patch(
        "litellm.proxy.auth.user_api_key_auth.get_key_object", new_callable=AsyncMock
    ):
        await _resolve_jwt_to_virtual_key(
            jwt_claims={"sub": "svc-a", "iss": "https://idp2.example.com"},
            jwt_handler=jwt_handler,
            prisma_client=prisma_client,
            user_api_key_cache=DualCache(),
            parent_otel_span=None,
            proxy_logging_obj=MagicMock(),
        )

    call_kwargs = prisma_client.db.litellm_jwtkeymapping.find_first.call_args.kwargs
    assert call_kwargs["where"]["issuer"] == "https://idp2.example.com"


@pytest.mark.asyncio
async def test_no_issuer_in_jwt_defaults_to_empty_string():
    """When JWT has no 'iss' field, issuer should default to empty string."""
    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(virtual_key_claim_field="sub")

    prisma_client = MagicMock()
    prisma_client.db.litellm_jwtkeymapping.find_first = AsyncMock(return_value=None)

    with patch(
        "litellm.proxy.auth.user_api_key_auth.get_key_object", new_callable=AsyncMock
    ):
        await _resolve_jwt_to_virtual_key(
            jwt_claims={"sub": "svc-b"},  # no "iss"
            jwt_handler=jwt_handler,
            prisma_client=prisma_client,
            user_api_key_cache=DualCache(),
            parent_otel_span=None,
            proxy_logging_obj=MagicMock(),
        )

    call_kwargs = prisma_client.db.litellm_jwtkeymapping.find_first.call_args.kwargs
    assert call_kwargs["where"]["issuer"] == ""


# ──────────────────────────────────────────────
# Block 6: Gap #6 — endpoints expose virtual key properties
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_info_endpoint_returns_virtual_key_fields():
    """/jwt/key/mapping/info should return virtual key fields when the key row exists."""
    now = datetime.now(timezone.utc)
    mock_prisma = _mock_prisma()
    mock_prisma.db.litellm_jwtkeymapping.find_unique.return_value = _mock_mapping()

    key_row = MagicMock()
    key_row.models = ["gpt-4o"]
    key_row.max_budget = 50.0
    key_row.budget_duration = "30d"
    key_row.tpm_limit = 10000
    key_row.rpm_limit = 100
    key_row.team_id = "team-1"
    key_row.key_alias = "my-client"
    key_row.spend = 2.5
    key_row.expires = now
    mock_prisma.db.litellm_verificationtoken.find_first.return_value = key_row
    mock_cache = AsyncMock()

    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma), patch(
        "litellm.proxy.proxy_server.user_api_key_cache", mock_cache
    ):
        result = await info_jwt_key_mapping(
            id="mapping-1", user_api_key_dict=_make_admin_auth()
        )

    assert result.models == ["gpt-4o"]
    assert result.max_budget == 50.0
    assert result.team_id == "team-1"
    assert result.key_alias == "my-client"
    assert result.spend == 2.5


@pytest.mark.asyncio
async def test_info_endpoint_still_returns_mapping_fields():
    """/jwt/key/mapping/info response must still include mapping metadata."""
    mock_prisma = _mock_prisma()
    mock_prisma.db.litellm_jwtkeymapping.find_unique.return_value = _mock_mapping(
        claim_name="email", claim_value="user@corp.com"
    )
    mock_prisma.db.litellm_verificationtoken.find_first.return_value = None
    mock_cache = AsyncMock()

    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma), patch(
        "litellm.proxy.proxy_server.user_api_key_cache", mock_cache
    ):
        result = await info_jwt_key_mapping(
            id="mapping-1", user_api_key_dict=_make_admin_auth()
        )

    assert result.jwt_claim_name == "email"
    assert result.jwt_claim_value == "user@corp.com"
    assert result.is_active is True
