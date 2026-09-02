import time
from collections.abc import Mapping
from typing import Final

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException
from pydantic import ValidationError

from litellm.proxy._experimental.mcp_server.oauth_identity_binding import (
    RefreshOwnershipProven,
    RefreshTokenPresented,
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
