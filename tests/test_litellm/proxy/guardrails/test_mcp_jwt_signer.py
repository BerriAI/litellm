"""
Tests for the MCPJWTSigner built-in guardrail.

Tests cover:
  - RSA key generation and loading
  - JWT signing and JWKS format
  - Claim building (sub, act, scope)
  - Hook fires for call_mcp_tool, skips other call types
  - get_mcp_jwt_signer() singleton pattern
"""

import base64
import time
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_user_api_key_dict(
    user_id: str = "user-123",
    team_id: str = "team-abc",
    user_email: str = "user@example.com",
    end_user_id: Optional[str] = None,
) -> MagicMock:
    mock = MagicMock()
    mock.user_id = user_id
    mock.team_id = team_id
    mock.user_email = user_email
    mock.end_user_id = end_user_id
    mock.org_id = None
    return mock


def _decode_unverified(token: str) -> Dict[str, Any]:
    return jwt.decode(token, options={"verify_signature": False})


# ---------------------------------------------------------------------------
# Import target (inline so we can reset the singleton between tests)
# ---------------------------------------------------------------------------


def _make_signer(**kwargs: Any):
    # Reset singleton before each signer creation to avoid cross-test pollution
    import litellm.proxy.guardrails.guardrail_hooks.mcp_jwt_signer.mcp_jwt_signer as mod

    mod._mcp_jwt_signer_instance = None

    from litellm.proxy.guardrails.guardrail_hooks.mcp_jwt_signer.mcp_jwt_signer import (
        MCPJWTSigner,
    )

    return MCPJWTSigner(
        guardrail_name="test-jwt-signer",
        event_hook="pre_mcp_call",
        default_on=True,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Key generation tests
# ---------------------------------------------------------------------------


def test_auto_generates_rsa_keypair():
    """MCPJWTSigner auto-generates an RSA-2048 keypair when env var is unset."""
    signer = _make_signer()
    assert signer._private_key is not None
    assert signer._public_key is not None
    assert signer._kid is not None and len(signer._kid) == 16


def test_kid_is_deterministic():
    """Two signers built from the same key have the same kid."""
    signer1 = _make_signer()
    private_pem = signer1._private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")

    with patch.dict("os.environ", {"MCP_JWT_SIGNING_KEY": private_pem}):
        signer2 = _make_signer()

    assert signer1._kid == signer2._kid


def test_load_key_from_env_var():
    """MCPJWTSigner loads a user-provided RSA key from the env var."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")

    with patch.dict("os.environ", {"MCP_JWT_SIGNING_KEY": pem}):
        signer = _make_signer()

    assert signer._kid is not None


# ---------------------------------------------------------------------------
# JWKS tests
# ---------------------------------------------------------------------------


def test_get_jwks_format():
    """get_jwks() returns a valid JWKS dict with RSA fields."""
    signer = _make_signer()
    jwks = signer.get_jwks()

    assert "keys" in jwks
    assert len(jwks["keys"]) == 1
    key = jwks["keys"][0]

    assert key["kty"] == "RSA"
    assert key["alg"] == "RS256"
    assert key["use"] == "sig"
    assert key["kid"] == signer._kid
    assert "n" in key and len(key["n"]) > 0
    assert "e" in key and key["e"] == "AQAB"  # 65537 in base64url


def test_jwks_public_key_can_verify_signed_jwt():
    """A JWT signed by MCPJWTSigner can be verified using the JWKS public key."""
    signer = _make_signer(issuer="https://litellm.example.com", audience="mcp")
    now = int(time.time())
    claims = {"iss": "https://litellm.example.com", "aud": "mcp", "iat": now, "exp": now + 300}

    token = jwt.encode(claims, signer._private_key, algorithm="RS256")

    # Reconstruct public key from JWKS
    jwks = signer.get_jwks()
    key_data = jwks["keys"][0]
    n = int.from_bytes(base64.urlsafe_b64decode(key_data["n"] + "=="), byteorder="big")
    e = int.from_bytes(base64.urlsafe_b64decode(key_data["e"] + "=="), byteorder="big")
    from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicNumbers
    pub_key = RSAPublicNumbers(e=e, n=n).public_key()

    decoded = jwt.decode(
        token,
        pub_key,
        algorithms=["RS256"],
        audience="mcp",
        issuer="https://litellm.example.com",
    )
    assert decoded["iss"] == "https://litellm.example.com"


# ---------------------------------------------------------------------------
# Claim building tests
# ---------------------------------------------------------------------------


def test_build_claims_standard_fields():
    """_build_claims() populates iss, aud, iat, exp, nbf."""
    signer = _make_signer(issuer="https://litellm.example.com", audience="mcp", ttl_seconds=300)
    user_dict = _make_user_api_key_dict()
    data = {"mcp_tool_name": "get_weather"}

    claims = signer._build_claims(user_dict, data)

    assert claims["iss"] == "https://litellm.example.com"
    assert claims["aud"] == "mcp"
    assert "iat" in claims
    assert "exp" in claims
    assert claims["exp"] - claims["iat"] == 300
    assert "nbf" in claims


def test_build_claims_identity():
    """_build_claims() sets sub from user_id and act from team_id (RFC 8693)."""
    signer = _make_signer()
    user_dict = _make_user_api_key_dict(user_id="user-xyz", team_id="team-eng")
    data: Dict[str, Any] = {}

    claims = signer._build_claims(user_dict, data)

    assert claims["sub"] == "user-xyz"
    assert claims["act"]["sub"] == "team-eng"
    assert claims["email"] == "user@example.com"


def test_build_claims_scope_with_tool():
    """_build_claims() encodes tool-specific scope when mcp_tool_name is set."""
    signer = _make_signer()
    user_dict = _make_user_api_key_dict()
    data = {"mcp_tool_name": "search_web"}

    claims = signer._build_claims(user_dict, data)

    scopes = set(claims["scope"].split())
    assert "mcp:tools/call" in scopes
    assert "mcp:tools/list" in scopes
    assert "mcp:tools/search_web:call" in scopes


def test_build_claims_scope_without_tool():
    """_build_claims() omits per-tool scope when mcp_tool_name is not set."""
    signer = _make_signer()
    user_dict = _make_user_api_key_dict()
    data: Dict[str, Any] = {}

    claims = signer._build_claims(user_dict, data)

    scopes = set(claims["scope"].split())
    assert "mcp:tools/call" in scopes
    assert "mcp:tools/list" in scopes
    # No per-tool scope
    for scope in scopes:
        assert ":" not in scope.replace("mcp:", "") or scope.endswith(":call") is False or scope == "mcp:tools/call"


def test_build_claims_act_fallback_to_litellm_proxy():
    """_build_claims() falls back to 'litellm-proxy' when team_id and org_id are absent."""
    signer = _make_signer()
    user_dict = _make_user_api_key_dict()
    user_dict.team_id = None
    user_dict.org_id = None

    claims = signer._build_claims(user_dict, {})

    assert claims["act"]["sub"] == "litellm-proxy"


# ---------------------------------------------------------------------------
# Hook dispatch tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hook_fires_for_call_mcp_tool():
    """async_pre_call_hook() injects Authorization header for call_mcp_tool."""
    signer = _make_signer(issuer="https://litellm.example.com", audience="mcp")
    user_dict = _make_user_api_key_dict()
    data = {"mcp_tool_name": "do_thing"}

    result = await signer.async_pre_call_hook(
        user_api_key_dict=user_dict,
        cache=MagicMock(),
        data=data,
        call_type="call_mcp_tool",
    )

    assert isinstance(result, dict)
    assert "extra_headers" in result
    assert result["extra_headers"]["Authorization"].startswith("Bearer ")


@pytest.mark.asyncio
async def test_hook_skips_non_mcp_call_types():
    """async_pre_call_hook() leaves data unchanged for non-MCP call types."""
    signer = _make_signer()
    user_dict = _make_user_api_key_dict()
    data = {"messages": [{"role": "user", "content": "hello"}]}

    for call_type in ("completion", "acompletion", "embedding", "list_mcp_tools"):
        original_data = {**data}
        result = await signer.async_pre_call_hook(
            user_api_key_dict=user_dict,
            cache=MagicMock(),
            data=original_data,
            call_type=call_type,  # type: ignore[arg-type]
        )
        assert "extra_headers" not in (result or {}), f"extra_headers should not be set for {call_type}"


@pytest.mark.asyncio
async def test_signed_token_is_verifiable():
    """The JWT injected by the hook can be verified against the JWKS public key."""
    signer = _make_signer(issuer="https://litellm.example.com", audience="mcp", ttl_seconds=300)
    user_dict = _make_user_api_key_dict(user_id="alice", team_id="backend")
    data = {"mcp_tool_name": "search"}

    result = await signer.async_pre_call_hook(
        user_api_key_dict=user_dict,
        cache=MagicMock(),
        data=data,
        call_type="call_mcp_tool",
    )

    assert isinstance(result, dict)
    token = result["extra_headers"]["Authorization"].removeprefix("Bearer ")

    decoded = _decode_unverified(token)
    assert decoded["sub"] == "alice"
    assert decoded["act"]["sub"] == "backend"
    assert "mcp:tools/search:call" in decoded["scope"]
    assert decoded["iss"] == "https://litellm.example.com"
    assert decoded["aud"] == "mcp"


# ---------------------------------------------------------------------------
# Singleton tests
# ---------------------------------------------------------------------------


def test_get_mcp_jwt_signer_returns_none_before_init():
    """get_mcp_jwt_signer() returns None before any MCPJWTSigner is created."""
    import litellm.proxy.guardrails.guardrail_hooks.mcp_jwt_signer.mcp_jwt_signer as mod

    mod._mcp_jwt_signer_instance = None

    from litellm.proxy.guardrails.guardrail_hooks.mcp_jwt_signer.mcp_jwt_signer import (
        get_mcp_jwt_signer,
    )

    assert get_mcp_jwt_signer() is None


def test_get_mcp_jwt_signer_returns_instance_after_init():
    """get_mcp_jwt_signer() returns the initialized signer instance."""
    from litellm.proxy.guardrails.guardrail_hooks.mcp_jwt_signer.mcp_jwt_signer import (
        get_mcp_jwt_signer,
    )

    signer = _make_signer()
    assert get_mcp_jwt_signer() is signer
