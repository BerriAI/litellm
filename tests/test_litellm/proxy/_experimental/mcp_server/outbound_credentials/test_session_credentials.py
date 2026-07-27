"""Tests for the session-token KDF and the edge/token-endpoint resolvers."""

from datetime import datetime, timedelta, timezone

import pytest

from litellm.proxy._experimental.mcp_server.outbound_credentials.bridge_credentials import (
    envelope_keys_from_master_key,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.session_credentials import (
    NotSessionBearer,
    SessionBearerAdmitted,
    SessionBearerInvalid,
    SessionRefreshInvalid,
    SessionRefreshOpened,
    is_session_bearer_shaped,
    open_session_refresh_bearer,
    resolve_session_bearer,
    session_keys_from_master_key,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.session_token import (
    SESSION_TTL_SECONDS,
    MintedSessionToken,
    SessionPrincipal,
    mint_session_refresh_token,
    mint_session_token,
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
    tampered = token[:-2] + ("aa" if not token.endswith("aa") else "bb")
    result = resolve_session_bearer(f"Bearer {tampered}", KEYS, NOW)
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
