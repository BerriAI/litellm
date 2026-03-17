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
import json
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
    # jwt_claims must default to None so the mock doesn't pretend to have
    # upstream JWT claims when none were configured.
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


# ---------------------------------------------------------------------------
# FR-5: Verify + re-sign — uses upstream jwt_claims when available
# ---------------------------------------------------------------------------


def test_build_claims_uses_jwt_claims_sub_when_available():
    """FR-5: When jwt_claims is populated, sub is taken from the upstream token."""
    signer = _make_signer(
        access_token_discovery_uri="https://okta.example.com/.well-known/openid-configuration"
    )
    user_dict = _make_user_api_key_dict(user_id="litellm-user-999")
    # Simulate upstream Okta JWT claims already decoded by litellm JWT auth.
    user_dict.jwt_claims = {"sub": "okta-user-abc123", "email": "alice@corp.com"}

    claims = signer._build_claims(user_dict, {})

    # sub must come from upstream jwt_claims, not litellm user_id
    assert claims["sub"] == "okta-user-abc123"
    assert claims["email"] == "alice@corp.com"


def test_build_claims_falls_back_to_user_id_when_no_jwt_claims():
    """FR-5 M2M mode: falls back to user_id when jwt_claims is absent."""
    signer = _make_signer(
        access_token_discovery_uri="https://okta.example.com/.well-known/openid-configuration"
    )
    user_dict = _make_user_api_key_dict(user_id="svc-account-42")
    user_dict.jwt_claims = None

    claims = signer._build_claims(user_dict, {})

    assert claims["sub"] == "svc-account-42"


# ---------------------------------------------------------------------------
# FR-12: End-user identity mapping via end_user_claim_sources
# ---------------------------------------------------------------------------


def test_end_user_claim_sources_picks_first_non_empty():
    """FR-12: Identity is resolved from the first non-empty source."""
    signer = _make_signer(
        end_user_claim_sources=["sso_id", "preferred_username", "sub"]
    )
    user_dict = _make_user_api_key_dict(user_id="ignored")
    user_dict.jwt_claims = {
        "sub": "fallback-sub",
        "preferred_username": "alice",
        # sso_id absent
    }

    claims = signer._build_claims(user_dict, {})

    # sso_id absent, preferred_username present → use preferred_username
    assert claims["sub"] == "alice"


def test_end_user_claim_sources_header_resolution():
    """FR-12: Identity can be resolved from a raw header when claim is absent."""
    signer = _make_signer(end_user_claim_sources=["x-end-user-id", "sub"])
    user_dict = _make_user_api_key_dict(user_id="litellm-user")
    user_dict.jwt_claims = {"sub": "fallback-sub"}
    data = {
        "mcp_raw_headers": {"x-end-user-id": "header-user-789"},
    }

    claims = signer._build_claims(user_dict, data)

    assert claims["sub"] == "header-user-789"


def test_end_user_claim_sources_falls_through_all():
    """FR-12: Falls back gracefully when no source matches."""
    signer = _make_signer(end_user_claim_sources=["nonexistent_claim"])
    user_dict = _make_user_api_key_dict(user_id="")
    user_dict.user_id = None
    user_dict.jwt_claims = {}
    user_dict.token = "sk-abc123"

    claims = signer._build_claims(user_dict, {})

    # Falls back to apikey hash
    assert claims["sub"].startswith("apikey:")


# ---------------------------------------------------------------------------
# FR-13: Claim operations (add_claims, set_claims, remove_claims)
# ---------------------------------------------------------------------------


def test_add_claims_adds_missing_claims():
    """FR-13: add_claims adds claims that are not already present."""
    signer = _make_signer(add_claims={"tenant_id": "acme", "env": "prod"})
    user_dict = _make_user_api_key_dict()

    claims = signer._build_claims(user_dict, {})

    assert claims["tenant_id"] == "acme"
    assert claims["env"] == "prod"


def test_add_claims_does_not_override_existing():
    """FR-13: add_claims does NOT override claims that already exist (e.g., iss)."""
    signer = _make_signer(
        issuer="https://litellm.example.com",
        add_claims={"iss": "https://imposter.example.com"},
    )
    user_dict = _make_user_api_key_dict()

    claims = signer._build_claims(user_dict, {})

    # add_claims must not override the signer-built issuer
    assert claims["iss"] == "https://litellm.example.com"


def test_set_claims_overrides_existing():
    """FR-13: set_claims overrides existing claims (including signer-built ones)."""
    signer = _make_signer(
        audience="mcp",
        set_claims={"aud": "custom-audience", "custom_key": "custom_val"},
    )
    user_dict = _make_user_api_key_dict()

    claims = signer._build_claims(user_dict, {})

    assert claims["aud"] == "custom-audience"
    assert claims["custom_key"] == "custom_val"


def test_remove_claims_strips_specified_claims():
    """FR-13: remove_claims strips listed claim names from the output JWT."""
    signer = _make_signer(remove_claims=["email", "act"])
    user_dict = _make_user_api_key_dict(user_email="alice@example.com")

    claims = signer._build_claims(user_dict, {})

    assert "email" not in claims
    assert "act" not in claims


def test_claim_operations_order():
    """FR-13: Operations are applied add → set → remove. Set wins over add; remove wins over both."""
    signer = _make_signer(
        add_claims={"x": "from-add"},
        set_claims={"x": "from-set", "y": "from-set"},
        remove_claims=["y"],
    )
    user_dict = _make_user_api_key_dict()

    claims = signer._build_claims(user_dict, {})

    # add adds x, then set overrides x
    assert claims["x"] == "from-set"
    # set adds y, then remove strips it
    assert "y" not in claims


# ---------------------------------------------------------------------------
# FR-14: Two-token model (channel token)
# ---------------------------------------------------------------------------


def test_build_claims_uses_channel_token_as_act():
    """FR-14: When channel_token_claims is provided, act.sub comes from its sub."""
    signer = _make_signer()
    user_dict = _make_user_api_key_dict(team_id="team-ignored")
    channel_claims = {"sub": "agent-service-001", "client_id": "agent-client"}

    claims = signer._build_claims(user_dict, {}, channel_token_claims=channel_claims)

    assert claims["act"]["sub"] == "agent-service-001"
    assert claims["act"]["client_id"] == "agent-client"


def test_build_claims_channel_token_fallback_to_client_id():
    """FR-14: Falls back to client_id when sub is absent in channel token."""
    signer = _make_signer()
    user_dict = _make_user_api_key_dict()
    channel_claims = {"client_id": "m2m-client-xyz"}

    claims = signer._build_claims(user_dict, {}, channel_token_claims=channel_claims)

    assert claims["act"]["sub"] == "m2m-client-xyz"


@pytest.mark.asyncio
async def test_hook_reads_channel_token_from_raw_headers():
    """FR-14: async_pre_call_hook picks up X-Channel-Token from mcp_raw_headers."""
    signer = _make_signer()

    # Build a valid channel token to inject
    channel_payload = {"sub": "channel-agent", "exp": int(time.time()) + 300}
    channel_token = jwt.encode(channel_payload, signer._private_key, algorithm="RS256")

    user_dict = _make_user_api_key_dict()
    data = {
        "mcp_tool_name": "search",
        "mcp_raw_headers": {"x-channel-token": channel_token},
    }

    result = await signer.async_pre_call_hook(
        user_api_key_dict=user_dict,
        cache=MagicMock(),
        data=data,
        call_type="call_mcp_tool",
    )

    assert isinstance(result, dict)
    token = result["extra_headers"]["Authorization"].removeprefix("Bearer ")
    decoded = _decode_unverified(token)
    # act should come from channel token (no sig verification since no discovery_uri)
    assert decoded["act"]["sub"] == "channel-agent"


# ---------------------------------------------------------------------------
# FR-15: Required/optional claim validation
# ---------------------------------------------------------------------------


def test_required_claims_passes_when_all_present():
    """FR-15: No error when all required_claims are present in jwt_claims."""
    from litellm.proxy.guardrails.guardrail_hooks.mcp_jwt_signer.mcp_jwt_signer import (
        _validate_required_claims,
    )

    _validate_required_claims(
        jwt_claims={"sub": "alice", "email": "alice@example.com"},
        required_claims=["sub", "email"],
    )  # must not raise


def test_required_claims_raises_when_missing():
    """FR-15: ValueError when a required claim is missing from jwt_claims."""
    from litellm.proxy.guardrails.guardrail_hooks.mcp_jwt_signer.mcp_jwt_signer import (
        _validate_required_claims,
    )

    with pytest.raises(ValueError, match="missing required_claims"):
        _validate_required_claims(
            jwt_claims={"sub": "alice"},
            required_claims=["sub", "groups"],
        )


def test_required_claims_raises_when_no_jwt_claims():
    """FR-15: ValueError when required_claims set but no JWT claims present."""
    from litellm.proxy.guardrails.guardrail_hooks.mcp_jwt_signer.mcp_jwt_signer import (
        _validate_required_claims,
    )

    with pytest.raises(ValueError, match="no JWT claims"):
        _validate_required_claims(
            jwt_claims=None,
            required_claims=["sub"],
        )


def test_required_claims_empty_list_always_passes():
    """FR-15: Empty required_claims never raises (even with None jwt_claims)."""
    from litellm.proxy.guardrails.guardrail_hooks.mcp_jwt_signer.mcp_jwt_signer import (
        _validate_required_claims,
    )

    _validate_required_claims(jwt_claims=None, required_claims=[])  # must not raise


@pytest.mark.asyncio
async def test_hook_raises_when_required_claims_missing():
    """FR-15: async_pre_call_hook raises ValueError when required_claims are absent."""
    signer = _make_signer(required_claims=["groups"])
    user_dict = _make_user_api_key_dict()
    user_dict.jwt_claims = {"sub": "alice"}  # no 'groups' claim

    with pytest.raises(ValueError, match="missing required_claims"):
        await signer.async_pre_call_hook(
            user_api_key_dict=user_dict,
            cache=MagicMock(),
            data={"mcp_tool_name": "do_thing"},
            call_type="call_mcp_tool",
        )


# ---------------------------------------------------------------------------
# FR-9: Debug headers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hook_emits_debug_header_by_default():
    """FR-9: x-litellm-mcp-debug header is emitted by default."""
    signer = _make_signer()
    user_dict = _make_user_api_key_dict()

    result = await signer.async_pre_call_hook(
        user_api_key_dict=user_dict,
        cache=MagicMock(),
        data={"mcp_tool_name": "test_tool"},
        call_type="call_mcp_tool",
    )

    assert isinstance(result, dict)
    assert "x-litellm-mcp-debug" in result["extra_headers"]
    debug = json.loads(result["extra_headers"]["x-litellm-mcp-debug"])
    assert debug["signer"] == "mcp_jwt_signer"
    assert debug["kid"] == signer._kid
    assert debug["issuer"] == signer.issuer
    assert "sub" in debug
    assert "mode" in debug


@pytest.mark.asyncio
async def test_hook_omits_debug_header_when_disabled():
    """FR-9: x-litellm-mcp-debug is not emitted when debug_header=False."""
    signer = _make_signer(debug_header=False)
    user_dict = _make_user_api_key_dict()

    result = await signer.async_pre_call_hook(
        user_api_key_dict=user_dict,
        cache=MagicMock(),
        data={"mcp_tool_name": "test_tool"},
        call_type="call_mcp_tool",
    )

    assert isinstance(result, dict)
    assert "x-litellm-mcp-debug" not in result["extra_headers"]


@pytest.mark.asyncio
async def test_debug_header_reports_re_sign_mode_when_jwt_claims_present():
    """FR-9: Debug header shows mode=re-sign when upstream jwt_claims are available."""
    signer = _make_signer()
    user_dict = _make_user_api_key_dict()
    user_dict.jwt_claims = {"sub": "upstream-user"}

    result = await signer.async_pre_call_hook(
        user_api_key_dict=user_dict,
        cache=MagicMock(),
        data={"mcp_tool_name": "tool"},
        call_type="call_mcp_tool",
    )

    assert isinstance(result, dict)
    debug = json.loads(result["extra_headers"]["x-litellm-mcp-debug"])
    assert debug["mode"] == "re-sign"


# ---------------------------------------------------------------------------
# FR-10: Configurable scope (allowed_tools)
# ---------------------------------------------------------------------------


def test_allowed_tools_restricts_scope():
    """FR-10: allowed_tools overrides auto-generated scope to admin-defined list."""
    signer = _make_signer(allowed_tools=["search_web", "get_weather"])
    user_dict = _make_user_api_key_dict()

    claims = signer._build_claims(user_dict, {"mcp_tool_name": "search_web"})

    scopes = set(claims["scope"].split())
    assert "mcp:tools/search_web:call" in scopes
    assert "mcp:tools/search_web:list" in scopes
    assert "mcp:tools/get_weather:call" in scopes
    # tools/list should NOT be present during a specific tool call
    assert "mcp:tools/list" not in scopes


def test_allowed_tools_grants_list_when_no_tool_name():
    """FR-10: allowed_tools grants mcp:tools/list when not calling a specific tool."""
    signer = _make_signer(allowed_tools=["search_web"])
    user_dict = _make_user_api_key_dict()

    claims = signer._build_claims(user_dict, {})

    assert "mcp:tools/list" in claims["scope"]


def test_allowed_tools_empty_falls_back_to_auto_scope():
    """FR-10: Empty allowed_tools uses original auto-generated scope."""
    signer = _make_signer(allowed_tools=[])
    user_dict = _make_user_api_key_dict()

    claims = signer._build_claims(user_dict, {"mcp_tool_name": "some_tool"})

    # Auto-scope: call + tool-specific
    assert "mcp:tools/call" in claims["scope"]
    assert "mcp:tools/some_tool:call" in claims["scope"]


# ---------------------------------------------------------------------------
# FR-5: access_token_discovery_uri with jwt_claims integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hook_uses_jwt_claims_for_sub_when_discovery_uri_set():
    """FR-5: In re-sign mode, hook uses upstream jwt_claims.sub rather than user_id."""
    signer = _make_signer(
        access_token_discovery_uri="https://login.example.com/.well-known/openid-configuration"
    )
    user_dict = _make_user_api_key_dict(user_id="litellm-internal-user")
    user_dict.jwt_claims = {"sub": "upstream-alice@corp.com"}

    result = await signer.async_pre_call_hook(
        user_api_key_dict=user_dict,
        cache=MagicMock(),
        data={"mcp_tool_name": "tool"},
        call_type="call_mcp_tool",
    )

    assert isinstance(result, dict)
    token = result["extra_headers"]["Authorization"].removeprefix("Bearer ")
    decoded = _decode_unverified(token)
    # sub must come from upstream jwt_claims
    assert decoded["sub"] == "upstream-alice@corp.com"
