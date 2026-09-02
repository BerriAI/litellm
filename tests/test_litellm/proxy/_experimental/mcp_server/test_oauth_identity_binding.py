import time
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException
from pydantic import ValidationError

from litellm.proxy._experimental.mcp_server.oauth_identity_binding import (
    RefreshOwnershipProven,
    RefreshTokenPresented,
    _discover_jwks_url,
    _fetch_issuer_jwks,
    _load_caller_principal,
    _load_stored_refresh_token,
    _select_signing_key,
    enforce_oauth_identity_binding,
)
from litellm.types.mcp import MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPOAuthIdentityBinding, MCPServer

ISSUER: Final = "https://idp.example.com"
AUDIENCE: Final = "litellm-client"
KID: Final = "test-key"

_PRIVATE_KEY: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_PRIVATE_PEM: Final = _PRIVATE_KEY.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption(),
)
_PUBLIC_JWK: Final = {
    **jwt.algorithms.RSAAlgorithm.to_jwk(_PRIVATE_KEY.public_key(), as_dict=True),
    "kid": KID,
    "alg": "RS256",
    "use": "sig",
}


def _sign_id_token(claims: Mapping[str, object]) -> str:
    payload: Final = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "exp": int(time.time()) + 300,
        "iat": int(time.time()),
        **claims,
    }
    return jwt.encode(payload, _PRIVATE_PEM, algorithm="RS256", headers={"kid": KID})


async def _jwks_fetcher(_binding: MCPOAuthIdentityBinding) -> list[Mapping[str, object]]:
    return [_PUBLIC_JWK]


def _caller_loader(email: str | None):
    async def load(_user_id: str, _binding: MCPOAuthIdentityBinding) -> str | None:
        return email

    return load


def _stored_refresh_token_loader(refresh_token: str | None):
    async def load(_user_id: str, _server_id: str) -> str | None:
        return refresh_token

    return load


def _server(mode: str = "enforce", **binding_overrides: object) -> MCPServer:
    return MCPServer(
        server_id="srv-1",
        name="srv-1",
        url="https://mcp.example.com",
        transport=MCPTransport.http,
        oauth_identity_binding=MCPOAuthIdentityBinding(
            mode=mode,
            issuer=ISSUER,
            audiences=[AUDIENCE],
            **binding_overrides,
        ),
    )


@pytest.mark.asyncio
async def test_fetch_issuer_jwks_fetches_and_caches_keys():
    binding: Final = _server(jwks_url="https://idp.example.com/jwks-cache").oauth_identity_binding
    assert binding is not None
    response: Final = MagicMock()
    response.json.return_value = {"keys": [_PUBLIC_JWK]}
    client: Final = MagicMock()
    client.get = AsyncMock(return_value=response)

    with patch(
        "litellm.proxy._experimental.mcp_server.oauth_identity_binding.get_async_httpx_client",
        return_value=client,
    ):
        first: Final = await _fetch_issuer_jwks(binding)
        second: Final = await _fetch_issuer_jwks(binding)

    assert first == second == [_PUBLIC_JWK]
    client.get.assert_awaited_once_with("https://idp.example.com/jwks-cache")


@pytest.mark.asyncio
async def test_fetch_issuer_jwks_rejects_malformed_document():
    binding: Final = _server(jwks_url="https://idp.example.com/jwks-invalid").oauth_identity_binding
    assert binding is not None
    response: Final = MagicMock()
    response.json.return_value = {}
    client: Final = MagicMock()
    client.get = AsyncMock(return_value=response)

    with patch(
        "litellm.proxy._experimental.mcp_server.oauth_identity_binding.get_async_httpx_client",
        return_value=client,
    ):
        with pytest.raises(TypeError, match="has no 'keys' array"):
            await _fetch_issuer_jwks(binding)


@pytest.mark.asyncio
async def test_discover_jwks_url_returns_provider_uri():
    response: Final = MagicMock()
    response.json.return_value = {"jwks_uri": "https://idp.example.com/jwks"}
    client: Final = MagicMock()
    client.get = AsyncMock(return_value=response)

    with patch(
        "litellm.proxy._experimental.mcp_server.oauth_identity_binding.get_async_httpx_client",
        return_value=client,
    ):
        result: Final = await _discover_jwks_url("https://idp.example.com/")

    assert result == "https://idp.example.com/jwks"
    client.get.assert_awaited_once_with("https://idp.example.com/.well-known/openid-configuration")


@pytest.mark.asyncio
async def test_discover_jwks_url_rejects_missing_provider_uri():
    response: Final = MagicMock()
    response.json.return_value = {}
    client: Final = MagicMock()
    client.get = AsyncMock(return_value=response)

    with patch(
        "litellm.proxy._experimental.mcp_server.oauth_identity_binding.get_async_httpx_client",
        return_value=client,
    ):
        with pytest.raises(ValueError, match="returned no jwks_uri"):
            await _discover_jwks_url("https://idp.example.com")


def test_select_signing_key_returns_matching_key_or_rejection():
    token: Final = _sign_id_token({"email": "alice@example.com", "email_verified": True})

    selected: Final = _select_signing_key(token, [_PUBLIC_JWK])
    rejected: Final = _select_signing_key(token, [])

    assert selected.__class__.__name__ == "PyJWK"
    assert rejected.code == "oauth_identity_binding_failed"


@pytest.mark.asyncio
async def test_load_caller_principal_supports_user_id_and_database_email():
    user_id_binding: Final = _server(caller_field="user_id").oauth_identity_binding
    assert user_id_binding is not None
    assert await _load_caller_principal("user-a", user_id_binding) == "user-a"

    with patch(
        "litellm.proxy._experimental.mcp_server.bridge_token_flow.load_active_user_by_id",
        new=AsyncMock(side_effect=["no_active_key", SimpleNamespace(user_email="alice@example.com")]),
    ):
        assert await _load_caller_principal("user-a", _server().oauth_identity_binding) is None
        assert await _load_caller_principal("user-a", _server().oauth_identity_binding) == "alice@example.com"


@pytest.mark.asyncio
async def test_load_stored_refresh_token_returns_credential_and_fails_closed():
    get_credential: Final = AsyncMock(return_value={"refresh_token": "rt-1"})
    with (
        patch(
            "litellm.proxy._experimental.mcp_server.db.get_user_oauth_credential",
            new=get_credential,
        ),
        patch(
            "litellm.proxy.utils.get_prisma_client_or_throw",
            return_value="prisma",
        ),
    ):
        assert await _load_stored_refresh_token("user-a", "srv-1") == "rt-1"

    with patch(
        "litellm.proxy.utils.get_prisma_client_or_throw",
        side_effect=RuntimeError("database unavailable"),
    ):
        assert await _load_stored_refresh_token("user-a", "srv-1") is None


@pytest.mark.asyncio
async def test_matching_principal_passes():
    token: Final = _sign_id_token({"email": "Alice@Example.com", "email_verified": True})
    result: Final = await enforce_oauth_identity_binding(
        server=_server(),
        token_response={"access_token": "at", "id_token": token},
        litellm_user_id="user-a",
        grant_type="authorization_code",
        refresh_ownership=None,
        jwks_fetcher=_jwks_fetcher,
        caller_principal_loader=_caller_loader("alice@example.com"),
    )
    assert result is None


@pytest.mark.asyncio
async def test_mismatched_principal_rejected():
    token: Final = _sign_id_token({"email": "mallory@example.com", "email_verified": True})
    with pytest.raises(HTTPException) as exc_info:
        await enforce_oauth_identity_binding(
            server=_server(),
            token_response={"access_token": "at", "id_token": token},
            litellm_user_id="user-a",
            grant_type="authorization_code",
            refresh_ownership=None,
            jwks_fetcher=_jwks_fetcher,
            caller_principal_loader=_caller_loader("alice@example.com"),
        )
    assert exc_info.value.status_code == 403
    assert exc_info.value.detail["error"] == "oauth_principal_mismatch"
    assert exc_info.value.detail["credential_stored"] is False


@pytest.mark.asyncio
async def test_missing_upstream_principal_rejected():
    token: Final = _sign_id_token({"email_verified": True})
    with pytest.raises(HTTPException) as exc_info:
        await enforce_oauth_identity_binding(
            server=_server(),
            token_response={"access_token": "at", "id_token": token},
            litellm_user_id="user-a",
            grant_type="authorization_code",
            refresh_ownership=None,
            jwks_fetcher=_jwks_fetcher,
            caller_principal_loader=_caller_loader("alice@example.com"),
        )
    assert exc_info.value.detail["error"] == "oauth_identity_binding_failed"
    assert "no usable" in exc_info.value.detail["error_description"]


@pytest.mark.asyncio
async def test_jwks_fetch_failure_is_rejected():
    async def fail(_binding: MCPOAuthIdentityBinding) -> list[Mapping[str, object]]:
        raise RuntimeError("jwks unavailable")

    token: Final = _sign_id_token({"email": "alice@example.com", "email_verified": True})
    with pytest.raises(HTTPException) as exc_info:
        await enforce_oauth_identity_binding(
            server=_server(),
            token_response={"access_token": "at", "id_token": token},
            litellm_user_id="user-a",
            grant_type="authorization_code",
            refresh_ownership=None,
            jwks_fetcher=fail,
            caller_principal_loader=_caller_loader("alice@example.com"),
        )
    assert exc_info.value.detail["error"] == "oauth_identity_binding_failed"
    assert "jwks unavailable" in exc_info.value.detail["error_description"]


@pytest.mark.asyncio
async def test_missing_signing_key_is_rejected():
    token: Final = _sign_id_token({"email": "alice@example.com", "email_verified": True})

    async def no_keys(_binding: MCPOAuthIdentityBinding) -> tuple[Mapping[str, object], ...]:
        return ()

    with pytest.raises(HTTPException) as exc_info:
        await enforce_oauth_identity_binding(
            server=_server(),
            token_response={"access_token": "at", "id_token": token},
            litellm_user_id="user-a",
            grant_type="authorization_code",
            refresh_ownership=None,
            jwks_fetcher=no_keys,
            caller_principal_loader=_caller_loader("alice@example.com"),
        )
    assert exc_info.value.detail["error"] == "oauth_identity_binding_failed"
    assert "signing key" in exc_info.value.detail["error_description"]


@pytest.mark.asyncio
async def test_missing_caller_principal_is_rejected():
    token: Final = _sign_id_token({"email": "alice@example.com", "email_verified": True})
    with pytest.raises(HTTPException) as exc_info:
        await enforce_oauth_identity_binding(
            server=_server(),
            token_response={"access_token": "at", "id_token": token},
            litellm_user_id="user-a",
            grant_type="authorization_code",
            refresh_ownership=None,
            jwks_fetcher=_jwks_fetcher,
            caller_principal_loader=_caller_loader(None),
        )
    assert exc_info.value.detail["error"] == "oauth_identity_binding_failed"
    assert "has no" in exc_info.value.detail["error_description"]


@pytest.mark.asyncio
async def test_refresh_without_id_token_requires_litellm_identity_for_presented_token():
    with pytest.raises(HTTPException) as exc_info:
        await enforce_oauth_identity_binding(
            server=_server(),
            token_response={"access_token": "at"},
            litellm_user_id=None,
            grant_type="refresh_token",
            refresh_ownership=RefreshTokenPresented("rt-1"),
            jwks_fetcher=_jwks_fetcher,
            caller_principal_loader=_caller_loader("alice@example.com"),
            stored_refresh_token_loader=_stored_refresh_token_loader("rt-1"),
        )
    assert exc_info.value.detail["error"] == "oauth_identity_binding_failed"
    assert "no resolvable LiteLLM user identity" in exc_info.value.detail["error_description"]


@pytest.mark.asyncio
async def test_user_id_principal_matching_uses_exact_comparison():
    token: Final = _sign_id_token({"sub": "user-a"})
    result: Final = await enforce_oauth_identity_binding(
        server=_server(principal_claim="sub", caller_field="user_id"),
        token_response={"access_token": "at", "id_token": token},
        litellm_user_id="user-a",
        grant_type="authorization_code",
        refresh_ownership=None,
        jwks_fetcher=_jwks_fetcher,
        caller_principal_loader=_caller_loader("user-a"),
    )
    assert result is None


@pytest.mark.asyncio
async def test_missing_id_token_rejected_on_authorization_code():
    with pytest.raises(HTTPException) as exc_info:
        await enforce_oauth_identity_binding(
            server=_server(),
            token_response={"access_token": "at"},
            litellm_user_id="user-a",
            grant_type="authorization_code",
            refresh_ownership=None,
            jwks_fetcher=_jwks_fetcher,
            caller_principal_loader=_caller_loader("alice@example.com"),
        )
    assert exc_info.value.status_code == 403
    assert exc_info.value.detail["error"] == "oauth_identity_binding_failed"


@pytest.mark.asyncio
async def test_refresh_without_id_token_allowed_when_presented_token_matches_stored_credential():
    result: Final = await enforce_oauth_identity_binding(
        server=_server(),
        token_response={"access_token": "at"},
        litellm_user_id="user-a",
        grant_type="refresh_token",
        refresh_ownership=RefreshTokenPresented("rt-1"),
        jwks_fetcher=_jwks_fetcher,
        caller_principal_loader=_caller_loader("alice@example.com"),
        stored_refresh_token_loader=_stored_refresh_token_loader("rt-1"),
    )
    assert result is None


@pytest.mark.asyncio
async def test_refresh_without_id_token_rejects_different_presented_token():
    with pytest.raises(HTTPException) as exc_info:
        await enforce_oauth_identity_binding(
            server=_server(),
            token_response={"access_token": "at"},
            litellm_user_id="user-a",
            grant_type="refresh_token",
            refresh_ownership=RefreshTokenPresented("rt-stolen"),
            jwks_fetcher=_jwks_fetcher,
            caller_principal_loader=_caller_loader("alice@example.com"),
            stored_refresh_token_loader=_stored_refresh_token_loader("rt-1"),
        )
    assert exc_info.value.status_code == 403
    assert exc_info.value.detail["error"] == "oauth_identity_binding_failed"


@pytest.mark.asyncio
async def test_refresh_without_id_token_rejects_missing_stored_token():
    with pytest.raises(HTTPException) as exc_info:
        await enforce_oauth_identity_binding(
            server=_server(),
            token_response={"access_token": "at"},
            litellm_user_id="user-a",
            grant_type="refresh_token",
            refresh_ownership=RefreshTokenPresented("rt-1"),
            jwks_fetcher=_jwks_fetcher,
            caller_principal_loader=_caller_loader("alice@example.com"),
            stored_refresh_token_loader=_stored_refresh_token_loader(None),
        )
    assert exc_info.value.status_code == 403
    assert exc_info.value.detail["error"] == "oauth_identity_binding_failed"


@pytest.mark.asyncio
async def test_refresh_without_id_token_passes_when_bridge_proves_ownership():
    async def fail_if_called(_user_id: str, _server_id: str) -> str | None:
        raise AssertionError("stored refresh token loader should not be called")

    result: Final = await enforce_oauth_identity_binding(
        server=_server(),
        token_response={"access_token": "at"},
        litellm_user_id="user-a",
        grant_type="refresh_token",
        refresh_ownership=RefreshOwnershipProven(),
        jwks_fetcher=_jwks_fetcher,
        caller_principal_loader=_caller_loader("alice@example.com"),
        stored_refresh_token_loader=fail_if_called,
    )
    assert result is None


@pytest.mark.asyncio
async def test_refresh_without_id_token_rejects_without_ownership_proof():
    with pytest.raises(HTTPException) as exc_info:
        await enforce_oauth_identity_binding(
            server=_server(),
            token_response={"access_token": "at"},
            litellm_user_id="user-a",
            grant_type="refresh_token",
            refresh_ownership=None,
            jwks_fetcher=_jwks_fetcher,
            caller_principal_loader=_caller_loader("alice@example.com"),
            stored_refresh_token_loader=_stored_refresh_token_loader("rt-1"),
        )
    assert exc_info.value.status_code == 403
    assert exc_info.value.detail["error"] == "oauth_identity_binding_failed"


@pytest.mark.asyncio
async def test_refresh_with_mismatched_id_token_rejected():
    token: Final = _sign_id_token({"email": "mallory@example.com", "email_verified": True})
    with pytest.raises(HTTPException):
        await enforce_oauth_identity_binding(
            server=_server(),
            token_response={"access_token": "at", "id_token": token},
            litellm_user_id="user-a",
            grant_type="refresh_token",
            refresh_ownership=None,
            jwks_fetcher=_jwks_fetcher,
            caller_principal_loader=_caller_loader("alice@example.com"),
        )


@pytest.mark.asyncio
async def test_audit_mode_logs_but_does_not_reject():
    token: Final = _sign_id_token({"email": "mallory@example.com", "email_verified": True})
    result: Final = await enforce_oauth_identity_binding(
        server=_server(mode="audit"),
        token_response={"access_token": "at", "id_token": token},
        litellm_user_id="user-a",
        grant_type="authorization_code",
        refresh_ownership=None,
        jwks_fetcher=_jwks_fetcher,
        caller_principal_loader=_caller_loader("alice@example.com"),
    )
    assert result is None


@pytest.mark.asyncio
async def test_unverified_email_rejected():
    token: Final = _sign_id_token({"email": "alice@example.com", "email_verified": False})
    with pytest.raises(HTTPException) as exc_info:
        await enforce_oauth_identity_binding(
            server=_server(),
            token_response={"access_token": "at", "id_token": token},
            litellm_user_id="user-a",
            grant_type="authorization_code",
            refresh_ownership=None,
            jwks_fetcher=_jwks_fetcher,
            caller_principal_loader=_caller_loader("alice@example.com"),
        )
    assert exc_info.value.detail["error"] == "oauth_identity_binding_failed"


@pytest.mark.asyncio
async def test_wrong_issuer_rejected():
    payload: Final = {
        "iss": "https://evil.example.com",
        "aud": AUDIENCE,
        "exp": int(time.time()) + 300,
        "email": "alice@example.com",
        "email_verified": True,
    }
    token: Final = jwt.encode(payload, _PRIVATE_PEM, algorithm="RS256", headers={"kid": KID})
    with pytest.raises(HTTPException) as exc_info:
        await enforce_oauth_identity_binding(
            server=_server(),
            token_response={"access_token": "at", "id_token": token},
            litellm_user_id="user-a",
            grant_type="authorization_code",
            refresh_ownership=None,
            jwks_fetcher=_jwks_fetcher,
            caller_principal_loader=_caller_loader("alice@example.com"),
        )
    assert exc_info.value.detail["error"] == "oauth_identity_binding_failed"


@pytest.mark.asyncio
async def test_no_litellm_identity_rejected():
    token: Final = _sign_id_token({"email": "alice@example.com", "email_verified": True})
    with pytest.raises(HTTPException) as exc_info:
        await enforce_oauth_identity_binding(
            server=_server(),
            token_response={"access_token": "at", "id_token": token},
            litellm_user_id=None,
            grant_type="authorization_code",
            refresh_ownership=None,
            jwks_fetcher=_jwks_fetcher,
            caller_principal_loader=_caller_loader("alice@example.com"),
        )
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_disabled_binding_is_noop():
    result: Final = await enforce_oauth_identity_binding(
        server=_server(mode="disabled"),
        token_response={"access_token": "at"},
        litellm_user_id=None,
        grant_type="authorization_code",
        refresh_ownership=None,
        jwks_fetcher=_jwks_fetcher,
        caller_principal_loader=_caller_loader(None),
    )
    assert result is None


@pytest.mark.asyncio
async def test_no_binding_is_noop():
    server: Final = MCPServer(
        server_id="srv-2",
        name="srv-2",
        url="https://mcp.example.com",
        transport=MCPTransport.http,
    )
    result: Final = await enforce_oauth_identity_binding(
        server=server,
        token_response={"access_token": "at"},
        litellm_user_id=None,
        grant_type="authorization_code",
        refresh_ownership=None,
        jwks_fetcher=_jwks_fetcher,
        caller_principal_loader=_caller_loader(None),
    )
    assert result is None


def test_identity_binding_requires_non_empty_audiences():
    with pytest.raises(ValidationError):
        MCPOAuthIdentityBinding(mode="enforce", issuer=ISSUER, audiences=[])
    with pytest.raises(ValidationError):
        MCPOAuthIdentityBinding(mode="enforce", issuer=ISSUER)


@pytest.mark.asyncio
async def test_wrong_audience_rejected():
    token: Final = _sign_id_token({"aud": "other-client", "email": "alice@example.com", "email_verified": True})
    with pytest.raises(HTTPException) as exc_info:
        await enforce_oauth_identity_binding(
            server=_server(),
            token_response={"access_token": "at", "id_token": token},
            litellm_user_id="user-a",
            grant_type="authorization_code",
            refresh_ownership=None,
            jwks_fetcher=_jwks_fetcher,
            caller_principal_loader=_caller_loader("alice@example.com"),
        )
    assert exc_info.value.status_code == 403
    assert exc_info.value.detail["error"] == "oauth_identity_binding_failed"
