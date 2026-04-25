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
from unittest.mock import AsyncMock, MagicMock, patch

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
    mock.token = None
    mock.api_key = None
    # Explicit None so MagicMock doesn't auto-create a truthy proxy attribute
    mock.jwt_claims = None
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
    claims = {
        "iss": "https://litellm.example.com",
        "aud": "mcp",
        "iat": now,
        "exp": now + 300,
    }

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
    signer = _make_signer(
        issuer="https://litellm.example.com", audience="mcp", ttl_seconds=300
    )
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
    assert "mcp:tools/search_web:call" in scopes
    # Tool-call JWTs must NOT carry mcp:tools/list — least-privilege
    assert "mcp:tools/list" not in scopes


def test_build_claims_scope_without_tool():
    """_build_claims() includes mcp:tools/list when no specific tool is called."""
    signer = _make_signer()
    user_dict = _make_user_api_key_dict()
    data: Dict[str, Any] = {}

    claims = signer._build_claims(user_dict, data)

    scopes = set(claims["scope"].split())
    assert "mcp:tools/call" in scopes
    assert "mcp:tools/list" in scopes
    # No per-tool call scope when no tool name was given
    assert not any(s.endswith(":call") and s != "mcp:tools/call" for s in scopes)


def test_build_claims_act_fallback_to_litellm_proxy():
    """_build_claims() falls back to 'litellm-proxy' when team_id and org_id are absent."""
    signer = _make_signer()
    user_dict = _make_user_api_key_dict()
    user_dict.team_id = None
    user_dict.org_id = None

    claims = signer._build_claims(user_dict, {})

    assert claims["act"]["sub"] == "litellm-proxy"


def test_build_claims_sub_fallback_to_token_hash():
    """_build_claims() sets sub to an apikey: hash when user_id is absent."""
    signer = _make_signer()
    user_dict = _make_user_api_key_dict(user_id="")
    user_dict.user_id = None
    user_dict.token = "sk-test-api-key-abc123"

    claims = signer._build_claims(user_dict, {})

    assert claims["sub"].startswith("apikey:")
    assert len(claims["sub"]) == len("apikey:") + 16  # sha256 hex[:16]


def test_build_claims_sub_fallback_to_litellm_proxy_when_no_token():
    """_build_claims() falls back to 'litellm-proxy' when user_id and token are both absent."""
    signer = _make_signer()
    user_dict = _make_user_api_key_dict(user_id="")
    user_dict.user_id = None
    user_dict.token = None
    user_dict.api_key = None

    claims = signer._build_claims(user_dict, {})

    assert claims["sub"] == "litellm-proxy"


def test_init_raises_on_zero_ttl():
    """MCPJWTSigner raises ValueError when ttl_seconds is 0."""
    with pytest.raises(ValueError, match="ttl_seconds must be > 0"):
        _make_signer(ttl_seconds=0)


def test_init_raises_on_negative_ttl():
    """MCPJWTSigner raises ValueError when ttl_seconds is negative."""
    with pytest.raises(ValueError, match="ttl_seconds must be > 0"):
        _make_signer(ttl_seconds=-60)


def test_jwks_max_age_persistent_key():
    """jwks_max_age is 3600 when key loaded from env var."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa as crsa

    private_key = crsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")

    with patch.dict("os.environ", {"MCP_JWT_SIGNING_KEY": pem}):
        signer = _make_signer()

    assert signer.jwks_max_age == 3600


def test_jwks_max_age_auto_generated_key():
    """jwks_max_age is 300 for auto-generated (ephemeral) keys."""
    signer = _make_signer()
    assert signer.jwks_max_age == 300


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
        assert "extra_headers" not in (
            result or {}
        ), f"extra_headers should not be set for {call_type}"


@pytest.mark.asyncio
async def test_signed_token_is_verifiable():
    """The JWT injected by the hook can be verified against the JWKS public key."""
    signer = _make_signer(
        issuer="https://litellm.example.com", audience="mcp", ttl_seconds=300
    )
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


# ---------------------------------------------------------------------------
# FR-10: Configurable scopes
# ---------------------------------------------------------------------------


def test_allowed_scopes_replaces_auto_generation():
    """When allowed_scopes is set it is used verbatim instead of auto-generating."""
    signer = _make_signer(allowed_scopes=["mcp:admin", "mcp:tools/call"])
    user_dict = _make_user_api_key_dict()
    data = {"mcp_tool_name": "some_tool"}

    claims = signer._build_claims(user_dict, data)

    assert claims["scope"] == "mcp:admin mcp:tools/call"


def test_tool_call_scope_no_list_permission():
    """Tool-call JWTs must NOT carry mcp:tools/list (least-privilege)."""
    signer = _make_signer()
    user_dict = _make_user_api_key_dict()
    data = {"mcp_tool_name": "my_tool"}

    claims = signer._build_claims(user_dict, data)

    scopes = set(claims["scope"].split())
    assert "mcp:tools/list" not in scopes
    assert "mcp:tools/call" in scopes
    assert "mcp:tools/my_tool:call" in scopes


# ---------------------------------------------------------------------------
# FR-12: End-user identity mapping
# ---------------------------------------------------------------------------


def test_end_user_claim_sources_token_sub():
    """end_user_claim_sources resolves sub from incoming JWT claims."""
    signer = _make_signer(end_user_claim_sources=["token:sub", "litellm:user_id"])
    user_dict = _make_user_api_key_dict(user_id="litellm-user")
    jwt_claims = {"sub": "idp-user-123", "email": "idp@example.com"}

    claims = signer._build_claims(user_dict, {}, jwt_claims=jwt_claims)

    assert claims["sub"] == "idp-user-123"


def test_end_user_claim_sources_falls_back_to_litellm_user_id():
    """Falls back to litellm:user_id when token:sub is absent."""
    signer = _make_signer(end_user_claim_sources=["token:sub", "litellm:user_id"])
    user_dict = _make_user_api_key_dict(user_id="litellm-user")
    jwt_claims: Dict[str, Any] = {}  # no sub

    claims = signer._build_claims(user_dict, {}, jwt_claims=jwt_claims)

    assert claims["sub"] == "litellm-user"


def test_end_user_claim_sources_email_source():
    """token:email resolves correctly."""
    signer = _make_signer(end_user_claim_sources=["token:email"])
    user_dict = _make_user_api_key_dict(user_id="")
    user_dict.user_id = None
    jwt_claims = {"email": "alice@corp.com"}

    claims = signer._build_claims(user_dict, {}, jwt_claims=jwt_claims)

    assert claims["sub"] == "alice@corp.com"


def test_end_user_claim_sources_litellm_email():
    """litellm:email resolves from UserAPIKeyAuth.user_email."""
    signer = _make_signer(end_user_claim_sources=["litellm:email"])
    user_dict = _make_user_api_key_dict(user_email="proxy-user@example.com")
    user_dict.user_id = None

    claims = signer._build_claims(user_dict, {})

    assert claims["sub"] == "proxy-user@example.com"


# ---------------------------------------------------------------------------
# FR-13: Claim operations
# ---------------------------------------------------------------------------


def test_add_claims_inserts_when_absent():
    """add_claims inserts key when it is not already in the JWT."""
    signer = _make_signer(add_claims={"deployment_id": "prod-001"})
    user_dict = _make_user_api_key_dict()

    claims = signer._build_claims(user_dict, {})

    assert claims["deployment_id"] == "prod-001"


def test_add_claims_does_not_overwrite_existing():
    """add_claims does NOT overwrite an existing claim (use set_claims for that)."""
    signer = _make_signer(add_claims={"iss": "should-not-win"})
    user_dict = _make_user_api_key_dict()

    claims = signer._build_claims(user_dict, {})

    # iss should be the configured issuer, not overwritten
    assert claims["iss"] != "should-not-win"


def test_set_claims_always_overrides():
    """set_claims always overrides computed claims."""
    signer = _make_signer(set_claims={"iss": "override-issuer", "custom": "x"})
    user_dict = _make_user_api_key_dict()

    claims = signer._build_claims(user_dict, {})

    assert claims["iss"] == "override-issuer"
    assert claims["custom"] == "x"


def test_remove_claims_deletes_keys():
    """remove_claims deletes specified keys from the final JWT."""
    signer = _make_signer(remove_claims=["nbf", "email"])
    user_dict = _make_user_api_key_dict()

    claims = signer._build_claims(user_dict, {})

    assert "nbf" not in claims
    assert "email" not in claims


def test_claim_operations_order_add_then_set_then_remove():
    """add → set → remove is applied in order: set wins over add, remove beats both."""
    signer = _make_signer(
        add_claims={"x": "from-add"},
        set_claims={"x": "from-set"},
        remove_claims=["x"],
    )
    user_dict = _make_user_api_key_dict()

    claims = signer._build_claims(user_dict, {})

    assert "x" not in claims  # remove wins


# ---------------------------------------------------------------------------
# FR-14: Two-token model
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_channel_token_injected_when_configured():
    """When channel_token_audience is set, x-mcp-channel-token header is injected."""
    signer = _make_signer(
        channel_token_audience="bedrock-gateway",
        channel_token_ttl=60,
    )
    user_dict = _make_user_api_key_dict()
    data = {"mcp_tool_name": "list_tables"}

    result = await signer.async_pre_call_hook(
        user_api_key_dict=user_dict,
        cache=MagicMock(),
        data=data,
        call_type="call_mcp_tool",
    )

    assert isinstance(result, dict)
    assert "x-mcp-channel-token" in result["extra_headers"]
    channel_token = result["extra_headers"]["x-mcp-channel-token"].removeprefix(
        "Bearer "
    )
    channel_payload = _decode_unverified(channel_token)
    assert channel_payload["aud"] == "bedrock-gateway"


@pytest.mark.asyncio
async def test_channel_token_absent_when_not_configured():
    """x-mcp-channel-token is not injected when channel_token_audience is unset."""
    signer = _make_signer()
    user_dict = _make_user_api_key_dict()
    data = {"mcp_tool_name": "tool"}

    result = await signer.async_pre_call_hook(
        user_api_key_dict=user_dict,
        cache=MagicMock(),
        data=data,
        call_type="call_mcp_tool",
    )

    assert isinstance(result, dict)
    assert "x-mcp-channel-token" not in result["extra_headers"]


# ---------------------------------------------------------------------------
# FR-15: Incoming claim validation
# ---------------------------------------------------------------------------


def test_required_claims_pass_when_present():
    """_validate_required_claims() passes when all required claims are present."""
    signer = _make_signer(required_claims=["sub", "email"])
    # Should not raise
    signer._validate_required_claims({"sub": "user", "email": "u@example.com"})


def test_required_claims_raise_403_when_missing():
    """_validate_required_claims() raises HTTP 403 when a required claim is missing."""
    from fastapi import HTTPException

    signer = _make_signer(required_claims=["sub", "email"])
    with pytest.raises(HTTPException) as exc_info:
        signer._validate_required_claims({"sub": "user"})  # email missing

    assert exc_info.value.status_code == 403
    assert "email" in str(exc_info.value.detail)


def test_required_claims_raise_when_no_jwt_claims():
    """_validate_required_claims() raises when jwt_claims is None and claims are required."""
    from fastapi import HTTPException

    signer = _make_signer(required_claims=["sub"])
    with pytest.raises(HTTPException):
        signer._validate_required_claims(None)


def test_optional_claims_passed_through():
    """optional_claims are forwarded from incoming jwt_claims into the outbound JWT."""
    signer = _make_signer(optional_claims=["groups", "roles"])
    user_dict = _make_user_api_key_dict()
    jwt_claims = {"sub": "u", "groups": ["admin"], "roles": ["editor"]}

    claims = signer._build_claims(user_dict, {}, jwt_claims=jwt_claims)

    assert claims["groups"] == ["admin"]
    assert claims["roles"] == ["editor"]


def test_optional_claims_not_injected_if_absent():
    """optional_claims are silently skipped when absent in incoming jwt_claims."""
    signer = _make_signer(optional_claims=["groups"])
    user_dict = _make_user_api_key_dict()
    jwt_claims: Dict[str, Any] = {"sub": "u"}  # no groups

    claims = signer._build_claims(user_dict, {}, jwt_claims=jwt_claims)

    assert "groups" not in claims


# ---------------------------------------------------------------------------
# FR-9: Debug headers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_debug_header_injected_when_enabled():
    """x-litellm-mcp-debug header is injected when debug_headers=True."""
    signer = _make_signer(debug_headers=True)
    user_dict = _make_user_api_key_dict()
    data = {"mcp_tool_name": "my_tool"}

    result = await signer.async_pre_call_hook(
        user_api_key_dict=user_dict,
        cache=MagicMock(),
        data=data,
        call_type="call_mcp_tool",
    )

    assert isinstance(result, dict)
    assert "x-litellm-mcp-debug" in result["extra_headers"]
    debug_val = result["extra_headers"]["x-litellm-mcp-debug"]
    assert "v=1" in debug_val
    assert "kid=" in debug_val
    assert "sub=" in debug_val


@pytest.mark.asyncio
async def test_debug_header_absent_when_disabled():
    """x-litellm-mcp-debug is NOT injected when debug_headers=False (default)."""
    signer = _make_signer()
    user_dict = _make_user_api_key_dict()
    data = {"mcp_tool_name": "tool"}

    result = await signer.async_pre_call_hook(
        user_api_key_dict=user_dict,
        cache=MagicMock(),
        data=data,
        call_type="call_mcp_tool",
    )

    assert isinstance(result, dict)
    assert "x-litellm-mcp-debug" not in result["extra_headers"]


# ---------------------------------------------------------------------------
# P1 fix: extra_headers merging (multi-guardrail chains)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extra_headers_are_merged_not_replaced():
    """
    Existing extra_headers from a prior guardrail are preserved — only
    Authorization is added/overwritten, other keys survive.
    """
    signer = _make_signer()
    user_dict = _make_user_api_key_dict()
    # Simulate a prior guardrail having injected a tracing header
    data = {
        "mcp_tool_name": "list",
        "extra_headers": {"x-trace-id": "abc123", "x-correlation-id": "xyz"},
    }

    result = await signer.async_pre_call_hook(
        user_api_key_dict=user_dict,
        cache=MagicMock(),
        data=data,
        call_type="call_mcp_tool",
    )

    assert isinstance(result, dict)
    headers = result["extra_headers"]
    # Prior headers preserved
    assert headers.get("x-trace-id") == "abc123"
    assert headers.get("x-correlation-id") == "xyz"
    # Authorization injected
    assert "Authorization" in headers


# ---------------------------------------------------------------------------
# FR-5: Verify + re-sign — jwt_claims fallback from UserAPIKeyAuth
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sub_resolved_from_user_api_key_dict_jwt_claims():
    """
    When no raw token is present but UserAPIKeyAuth.jwt_claims has a sub,
    the guardrail resolves sub from jwt_claims (LiteLLM-decoded JWT path).
    """
    signer = _make_signer(end_user_claim_sources=["token:sub", "litellm:user_id"])
    user_dict = _make_user_api_key_dict(user_id="litellm-fallback")
    # jwt_claims populated by LiteLLM's JWT auth machinery
    user_dict.jwt_claims = {"sub": "idp-alice", "email": "alice@idp.com"}
    data = {"mcp_tool_name": "query"}

    result = await signer.async_pre_call_hook(
        user_api_key_dict=user_dict,
        cache=MagicMock(),
        data=data,
        call_type="call_mcp_tool",
    )

    assert isinstance(result, dict)
    token = result["extra_headers"]["Authorization"].removeprefix("Bearer ")
    payload = _decode_unverified(token)
    assert payload["sub"] == "idp-alice"


# ---------------------------------------------------------------------------
# initialize_guardrail factory — regression test for config.yaml wire-up
# ---------------------------------------------------------------------------


def test_initialize_guardrail_passes_all_params():
    """
    initialize_guardrail must wire every documented config.yaml param through
    to MCPJWTSigner.  Previously only issuer/audience/ttl_seconds were passed;
    all FR-5/9/10/12/13/14/15 params were silently dropped.
    """
    import litellm.proxy.guardrails.guardrail_hooks.mcp_jwt_signer.mcp_jwt_signer as mod

    mod._mcp_jwt_signer_instance = None

    from litellm.proxy.guardrails.guardrail_hooks.mcp_jwt_signer import (
        initialize_guardrail,
    )

    litellm_params = MagicMock()
    litellm_params.mode = "pre_mcp_call"
    litellm_params.default_on = True
    litellm_params.optional_params = None
    # Set every non-default param directly on litellm_params
    litellm_params.issuer = "https://litellm.example.com"
    litellm_params.audience = "mcp-test"
    litellm_params.ttl_seconds = 120
    litellm_params.access_token_discovery_uri = (
        "https://idp.example.com/.well-known/openid-configuration"
    )
    litellm_params.token_introspection_endpoint = "https://idp.example.com/introspect"
    litellm_params.verify_issuer = "https://idp.example.com"
    litellm_params.verify_audience = "api://test"
    litellm_params.end_user_claim_sources = ["token:email", "litellm:user_id"]
    litellm_params.add_claims = {"deployment_id": "prod"}
    litellm_params.set_claims = {"env": "production"}
    litellm_params.remove_claims = ["nbf"]
    litellm_params.channel_token_audience = "bedrock-gateway"
    litellm_params.channel_token_ttl = 60
    litellm_params.required_claims = ["sub", "email"]
    litellm_params.optional_claims = ["groups"]
    litellm_params.debug_headers = True
    litellm_params.allowed_scopes = ["mcp:tools/call"]

    guardrail = {"guardrail_name": "mcp-jwt-signer"}

    with patch("litellm.logging_callback_manager.add_litellm_callback"):
        signer = initialize_guardrail(litellm_params, guardrail)

    assert signer.issuer == "https://litellm.example.com"
    assert signer.audience == "mcp-test"
    assert signer.ttl_seconds == 120
    assert (
        signer.access_token_discovery_uri
        == "https://idp.example.com/.well-known/openid-configuration"
    )
    assert signer.token_introspection_endpoint == "https://idp.example.com/introspect"
    assert signer.verify_issuer == "https://idp.example.com"
    assert signer.verify_audience == "api://test"
    assert signer.end_user_claim_sources == ["token:email", "litellm:user_id"]
    assert signer.add_claims == {"deployment_id": "prod"}
    assert signer.set_claims == {"env": "production"}
    assert signer.remove_claims == ["nbf"]
    assert signer.channel_token_audience == "bedrock-gateway"
    assert signer.channel_token_ttl == 60
    assert signer.required_claims == ["sub", "email"]
    assert signer.optional_claims == ["groups"]
    assert signer.debug_headers is True
    assert signer.allowed_scopes == ["mcp:tools/call"]


# ---------------------------------------------------------------------------
# FR-5: _fetch_jwks, _get_oidc_discovery, _verify_incoming_jwt,
#        _introspect_opaque_token
# ---------------------------------------------------------------------------

import litellm.proxy.guardrails.guardrail_hooks.mcp_jwt_signer.mcp_jwt_signer as _signer_mod


def _make_httpx_response(json_body: dict, status_code: int = 200):
    """Build a minimal fake httpx Response object."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_body
    mock_resp.raise_for_status = MagicMock()
    if status_code >= 400:
        from httpx import HTTPStatusError, Request, Response

        mock_resp.raise_for_status.side_effect = HTTPStatusError(
            "error", request=MagicMock(), response=MagicMock()
        )
    return mock_resp


# --- _fetch_jwks ---


@pytest.mark.asyncio
async def test_fetch_jwks_returns_keys_and_caches():
    """_fetch_jwks returns keys from the remote JWKS URI and caches the result."""
    _signer_mod._jwks_cache.clear()

    fake_keys = [{"kty": "RSA", "kid": "k1", "n": "abc", "e": "AQAB"}]
    fake_resp = _make_httpx_response({"keys": fake_keys})

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=fake_resp)

    with patch(
        "litellm.llms.custom_httpx.http_handler.get_async_httpx_client",
        return_value=mock_client,
    ):
        keys = await _signer_mod._fetch_jwks("https://idp.example.com/jwks")

    assert keys == fake_keys
    assert "https://idp.example.com/jwks" in _signer_mod._jwks_cache
    _signer_mod._jwks_cache.clear()


@pytest.mark.asyncio
async def test_fetch_jwks_uses_cache_on_second_call():
    """_fetch_jwks returns the cached value without a second HTTP call."""
    _signer_mod._jwks_cache.clear()
    fake_keys = [{"kty": "RSA", "kid": "k1"}]
    _signer_mod._jwks_cache["https://idp.example.com/jwks"] = (
        fake_keys,
        time.time(),
    )

    mock_client = MagicMock()
    mock_client.get = AsyncMock()

    with patch(
        "litellm.llms.custom_httpx.http_handler.get_async_httpx_client",
        return_value=mock_client,
    ):
        keys = await _signer_mod._fetch_jwks("https://idp.example.com/jwks")

    mock_client.get.assert_not_called()
    assert keys == fake_keys
    _signer_mod._jwks_cache.clear()


# --- _get_oidc_discovery ---


@pytest.mark.asyncio
async def test_get_oidc_discovery_caches_when_jwks_uri_present():
    """_get_oidc_discovery caches the doc when jwks_uri is in the response."""
    signer = _make_signer(
        access_token_discovery_uri="https://idp.example.com/.well-known/openid-configuration"
    )
    signer._oidc_discovery_doc = None  # ensure fresh

    discovery_doc = {
        "issuer": "https://idp.example.com",
        "jwks_uri": "https://idp.example.com/jwks",
    }

    with patch(
        "litellm.proxy.guardrails.guardrail_hooks.mcp_jwt_signer.mcp_jwt_signer._fetch_oidc_discovery",
        new_callable=AsyncMock,
        return_value=discovery_doc,
    ):
        result = await signer._get_oidc_discovery()

    assert result["jwks_uri"] == "https://idp.example.com/jwks"
    assert signer._oidc_discovery_doc == discovery_doc


@pytest.mark.asyncio
async def test_get_oidc_discovery_does_not_cache_when_jwks_uri_absent():
    """_get_oidc_discovery does NOT cache a doc that is missing jwks_uri."""
    signer = _make_signer(
        access_token_discovery_uri="https://idp.example.com/.well-known/openid-configuration"
    )
    signer._oidc_discovery_doc = None

    bad_doc = {"issuer": "https://idp.example.com"}  # no jwks_uri

    with patch(
        "litellm.proxy.guardrails.guardrail_hooks.mcp_jwt_signer.mcp_jwt_signer._fetch_oidc_discovery",
        new_callable=AsyncMock,
        return_value=bad_doc,
    ) as mock_fetch:
        result1 = await signer._get_oidc_discovery()
        result2 = await signer._get_oidc_discovery()

    # Returns the bad doc each time without caching it
    assert "jwks_uri" not in result1
    assert signer._oidc_discovery_doc is None  # never cached
    assert mock_fetch.call_count == 2  # retried on second call


# --- _verify_incoming_jwt ---


@pytest.mark.asyncio
async def test_verify_incoming_jwt_returns_payload_on_valid_token():
    """_verify_incoming_jwt decodes and returns claims from a valid JWT."""
    # Build a signer to get a real RSA key pair; use its key to mint the "incoming" JWT
    signer = _make_signer(
        access_token_discovery_uri="https://idp.example.com/.well-known/openid-configuration",
        verify_audience="api://test",
        verify_issuer="https://idp.example.com",
    )
    # Mint a JWT with signer's own key — we'll pretend it came from the IdP
    now = int(time.time())
    incoming_claims = {
        "sub": "idp-user-42",
        "iss": "https://idp.example.com",
        "aud": "api://test",
        "iat": now,
        "exp": now + 300,
    }
    incoming_token = jwt.encode(
        incoming_claims,
        signer._private_key,
        algorithm="RS256",
        headers={"kid": signer._kid},
    )

    # Build a JWKS from the same public key so verification passes
    jwks = signer.get_jwks()

    with patch.object(
        signer,
        "_get_oidc_discovery",
        new_callable=AsyncMock,
        return_value={"jwks_uri": "https://idp.example.com/jwks"},
    ):
        with patch(
            "litellm.proxy.guardrails.guardrail_hooks.mcp_jwt_signer.mcp_jwt_signer._fetch_jwks",
            new_callable=AsyncMock,
            return_value=jwks["keys"],
        ):
            payload = await signer._verify_incoming_jwt(incoming_token)

    assert payload["sub"] == "idp-user-42"


@pytest.mark.asyncio
async def test_verify_incoming_jwt_raises_on_expired_token():
    """_verify_incoming_jwt raises PyJWTError on an expired token."""
    signer = _make_signer(
        access_token_discovery_uri="https://idp.example.com/.well-known/openid-configuration",
    )
    expired_claims = {
        "sub": "idp-user",
        "iss": "https://idp.example.com",
        "aud": "api://test",
        "iat": int(time.time()) - 600,
        "exp": int(time.time()) - 300,  # expired
    }
    expired_token = jwt.encode(expired_claims, signer._private_key, algorithm="RS256")
    jwks = signer.get_jwks()

    with patch.object(
        signer,
        "_get_oidc_discovery",
        new_callable=AsyncMock,
        return_value={"jwks_uri": "https://idp.example.com/jwks"},
    ):
        with patch(
            "litellm.proxy.guardrails.guardrail_hooks.mcp_jwt_signer.mcp_jwt_signer._fetch_jwks",
            new_callable=AsyncMock,
            return_value=jwks["keys"],
        ):
            with pytest.raises(jwt.PyJWTError):
                await signer._verify_incoming_jwt(expired_token)


# --- _introspect_opaque_token ---


@pytest.mark.asyncio
async def test_introspect_opaque_token_returns_claims_when_active():
    """_introspect_opaque_token returns the introspection payload for active tokens."""
    signer = _make_signer(
        token_introspection_endpoint="https://idp.example.com/introspect"
    )

    introspection_response = {
        "active": True,
        "sub": "service-account",
        "scope": "read write",
    }
    fake_resp = _make_httpx_response(introspection_response)
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=fake_resp)

    with patch(
        "litellm.llms.custom_httpx.http_handler.get_async_httpx_client",
        return_value=mock_client,
    ):
        result = await signer._introspect_opaque_token("opaque-token-abc")

    assert result["sub"] == "service-account"
    assert result["active"] is True


@pytest.mark.asyncio
async def test_introspect_opaque_token_raises_on_inactive_token():
    """_introspect_opaque_token raises ExpiredSignatureError when active=false."""
    signer = _make_signer(
        token_introspection_endpoint="https://idp.example.com/introspect"
    )

    fake_resp = _make_httpx_response({"active": False})
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=fake_resp)

    with patch(
        "litellm.llms.custom_httpx.http_handler.get_async_httpx_client",
        return_value=mock_client,
    ):
        with pytest.raises(jwt.ExpiredSignatureError):
            await signer._introspect_opaque_token("opaque-token-xyz")


@pytest.mark.asyncio
async def test_introspect_opaque_token_raises_without_endpoint_configured():
    """_introspect_opaque_token raises ValueError when no endpoint is set."""
    signer = _make_signer()  # no token_introspection_endpoint

    with pytest.raises(ValueError, match="token_introspection_endpoint"):
        await signer._introspect_opaque_token("some-token")


# --- FR-5 end-to-end hook path ---


@pytest.mark.asyncio
async def test_hook_raises_401_when_jwt_verification_fails():
    """async_pre_call_hook raises HTTP 401 when incoming JWT verification fails."""
    from fastapi import HTTPException

    signer = _make_signer(
        access_token_discovery_uri="https://idp.example.com/.well-known/openid-configuration"
    )

    with patch.object(
        signer,
        "_verify_incoming_jwt",
        new_callable=AsyncMock,
        side_effect=jwt.InvalidSignatureError("bad signature"),
    ):
        with patch.object(
            signer,
            "_get_oidc_discovery",
            new_callable=AsyncMock,
            return_value={"jwks_uri": "https://idp.example.com/jwks"},
        ):
            with pytest.raises(HTTPException) as exc_info:
                await signer.async_pre_call_hook(
                    user_api_key_dict=_make_user_api_key_dict(),
                    cache=MagicMock(),
                    data={
                        "mcp_tool_name": "tool",
                        "incoming_bearer_token": "hdr.pld.sig",
                    },
                    call_type="call_mcp_tool",
                )

    assert exc_info.value.status_code == 401
