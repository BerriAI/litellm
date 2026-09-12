"""Tests for the session-token KDF and the edge/token-endpoint resolvers."""

from datetime import datetime, timedelta, timezone

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.utils import base64url_decode, base64url_encode
from pydantic import SecretStr

from litellm.proxy._experimental.mcp_server.outbound_credentials.bridge_credentials import (
    envelope_keys_from_master_key,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.session_credentials import (
    NotSessionBearer,
    SessionBearerAdmitted,
    SessionBearerInvalid,
    SessionRefreshInvalid,
    SessionRefreshOpened,
    SessionSigningConfigError,
    is_session_bearer_shaped,
    open_session_refresh_bearer,
    resolve_session_bearer,
    resolve_session_signing_keys,
    session_keys_from_master_key,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.session_token import (
    SESSION_TTL_SECONDS,
    AsymmetricSessionKeys,
    MintedSessionToken,
    SessionKeys,
    SessionPrincipal,
    mint_session_refresh_token,
    mint_session_token,
    session_public_key_pem,
)

NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
MASTER_KEY = "sk-master-key-for-tests"
KEYS = session_keys_from_master_key(MASTER_KEY)
PRINCIPAL = SessionPrincipal(user_id="user-123", client_id="llm_client_abc")


def _access_token() -> str:
    minted = mint_session_token(PRINCIPAL, KEYS, NOW)
    assert isinstance(minted, MintedSessionToken)
    return minted.token.get_secret_value()


def _refresh_token() -> str:
    minted = mint_session_refresh_token(PRINCIPAL, KEYS, NOW)
    assert isinstance(minted, MintedSessionToken)
    return minted.token.get_secret_value()


def _corrupt_signature(token: str) -> str:
    unsigned, signature = token.rsplit(".", 1)
    raw = base64url_decode(signature)
    return f"{unsigned}.{base64url_encode(bytes((raw[0] ^ 0x01,)) + raw[1:]).decode()}"


def test_kdf_is_deterministic_and_key_length_is_256_bit():
    again = session_keys_from_master_key(MASTER_KEY)
    assert again.signing_key.get_secret_value() == KEYS.signing_key.get_secret_value()
    assert len(bytes.fromhex(KEYS.signing_key.get_secret_value())) == 32


def test_kdf_domain_separated_from_envelope_keys():
    envelope_keys = envelope_keys_from_master_key(MASTER_KEY)
    session_signing = KEYS.signing_key.get_secret_value()
    assert session_signing != envelope_keys.signing_key.get_secret_value()
    assert session_signing != envelope_keys.encryption_key.get_secret_value()


def test_kdf_differs_across_master_keys():
    other = session_keys_from_master_key("sk-a-different-master-key")
    assert other.signing_key.get_secret_value() != KEYS.signing_key.get_secret_value()


@pytest.mark.parametrize(
    "value,expected",
    [
        ("Bearer sk-1234", False),
        ("sk-1234", False),
        ("Bearer llm_env_abc", False),
        ("Bearer llm_refresh_abc", False),
        ("llm_session_abc", True),
        ("Bearer llm_session_abc", True),
        ("bearer llm_srefresh_abc", True),
    ],
)
def test_is_session_bearer_shaped(value, expected):
    assert is_session_bearer_shaped(value) is expected


def test_resolve_admits_valid_access_token_with_and_without_scheme():
    token = _access_token()
    for value in (token, f"Bearer {token}", f"bearer {token}"):
        result = resolve_session_bearer(value, KEYS, NOW)
        assert isinstance(result, SessionBearerAdmitted)
        assert result.principal == PRINCIPAL


def test_resolve_passes_non_session_bearers_through():
    for value in ("Bearer sk-1234", "Bearer llm_env_whatever", "Bearer eyJhbGciOi"):
        assert isinstance(resolve_session_bearer(value, KEYS, NOW), NotSessionBearer)


def test_resolve_fails_expired_token_closed_and_flags_expiry():
    token = _access_token()
    later = NOW + timedelta(seconds=SESSION_TTL_SECONDS + 1)
    result = resolve_session_bearer(f"Bearer {token}", KEYS, later)
    assert isinstance(result, SessionBearerInvalid)
    assert result.expired is True


def test_resolve_fails_tampered_token_closed_without_expiry_flag():
    token = _access_token()
    result = resolve_session_bearer(f"Bearer {_corrupt_signature(token)}", KEYS, NOW)
    assert isinstance(result, SessionBearerInvalid)
    assert result.expired is False


def test_resolve_rejects_refresh_token_at_the_edge():
    result = resolve_session_bearer(f"Bearer {_refresh_token()}", KEYS, NOW)
    assert isinstance(result, SessionBearerInvalid)
    assert result.expired is False


def test_resolve_wrong_master_key_fails_closed():
    other_keys = session_keys_from_master_key("sk-rotated-master-key")
    result = resolve_session_bearer(f"Bearer {_access_token()}", other_keys, NOW)
    assert isinstance(result, SessionBearerInvalid)


def test_refresh_grant_opens_for_the_issued_client():
    result = open_session_refresh_bearer(_refresh_token(), KEYS, NOW, expected_client_id="llm_client_abc")
    assert isinstance(result, SessionRefreshOpened)
    assert result.principal == PRINCIPAL


def test_refresh_grant_rejects_a_different_client():
    result = open_session_refresh_bearer(_refresh_token(), KEYS, NOW, expected_client_id="llm_client_other")
    assert isinstance(result, SessionRefreshInvalid)


def test_refresh_grant_rejects_access_token_presented_as_refresh():
    result = open_session_refresh_bearer(_access_token(), KEYS, NOW, expected_client_id="llm_client_abc")
    assert isinstance(result, SessionRefreshInvalid)


def _rsa_private_pem() -> str:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()


def test_absent_signing_setting_keeps_the_master_key_hs256_default():
    resolved = resolve_session_signing_keys(MASTER_KEY, None)
    assert isinstance(resolved, SessionKeys)
    assert resolved.signing_key.get_secret_value() == KEYS.signing_key.get_secret_value()


def test_rs256_signing_setting_resolves_inline_pem_material():
    pem = _rsa_private_pem()
    resolved = resolve_session_signing_keys(
        MASTER_KEY,
        {"algorithm": "RS256", "kid": "2026-01", "private_key": pem},
    )
    assert isinstance(resolved, AsymmetricSessionKeys)
    assert resolved.kid == "2026-01"
    minted = mint_session_token(PRINCIPAL, resolved, NOW)
    assert isinstance(minted, MintedSessionToken)
    admitted = resolve_session_bearer(f"Bearer {minted.token.get_secret_value()}", resolved, NOW)
    assert isinstance(admitted, SessionBearerAdmitted)


def test_rs256_signing_setting_resolves_env_reference(monkeypatch):
    monkeypatch.setenv("MCP_SESSION_PRIVATE_KEY", _rsa_private_pem())
    resolved = resolve_session_signing_keys(
        MASTER_KEY,
        {"algorithm": "RS256", "kid": "2026-01", "private_key": "os.environ/MCP_SESSION_PRIVATE_KEY"},
    )
    assert isinstance(resolved, AsymmetricSessionKeys)


def test_rs256_signing_setting_resolves_previous_public_keys():
    old_pem = _rsa_private_pem()
    old_keys = AsymmetricSessionKeys(private_key_pem=SecretStr(old_pem), kid="2025-06")
    resolved = resolve_session_signing_keys(
        MASTER_KEY,
        {
            "algorithm": "RS256",
            "kid": "2026-01",
            "private_key": _rsa_private_pem(),
            "previous_public_keys": [{"kid": "2025-06", "public_key": session_public_key_pem(old_keys)}],
        },
    )
    assert isinstance(resolved, AsymmetricSessionKeys)
    minted = mint_session_token(PRINCIPAL, old_keys, NOW)
    assert isinstance(minted, MintedSessionToken)
    admitted = resolve_session_bearer(f"Bearer {minted.token.get_secret_value()}", resolved, NOW)
    assert isinstance(admitted, SessionBearerAdmitted)


@pytest.mark.parametrize(
    "raw",
    [
        {"algorithm": "HS512", "kid": "k", "private_key": "irrelevant"},
        {"algorithm": "RS256", "kid": "k"},
        {"algorithm": "RS256", "kid": "k", "private_key": "not a pem"},
        {"algorithm": "RS256", "kid": "k", "private_key": "os.environ/UNSET_MCP_SESSION_KEY_VAR"},
        {"algorithm": "RS256", "kid": "k", "private_key": "x", "unexpected": True},
        "not-a-mapping",
    ],
)
def test_defective_signing_setting_fails_closed_never_falls_back_to_hs256(raw):
    resolved = resolve_session_signing_keys(MASTER_KEY, raw)
    assert isinstance(resolved, SessionSigningConfigError)


def test_signing_config_error_detail_never_leaks_key_material():
    pem = _rsa_private_pem()
    resolved = resolve_session_signing_keys(
        MASTER_KEY,
        {"algorithm": "RS256", "kid": "k", "private_key": pem, "unexpected": True},
    )
    assert isinstance(resolved, SessionSigningConfigError)
    assert pem.splitlines()[1] not in resolved.detail
