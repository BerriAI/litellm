"""Tests for the identity-only gateway session token (mint/open, hostile-input totality)."""

from datetime import datetime, timedelta, timezone

import jwt
import pytest
from pydantic import SecretStr, ValidationError

from litellm.proxy._experimental.mcp_server.outbound_credentials.session_token import (
    MAX_SESSION_TOKEN_BYTES,
    SESSION_ISSUER,
    SESSION_REFRESH_PREFIX,
    SESSION_REFRESH_TTL_SECONDS,
    SESSION_TOKEN_PREFIX,
    SESSION_TTL_SECONDS,
    MintedSessionToken,
    NotASessionToken,
    OpenedSessionToken,
    SessionBadSignature,
    SessionExpired,
    SessionKeys,
    SessionMalformed,
    SessionPrincipal,
    SessionTokenTooLarge,
    is_session_refresh_token,
    is_session_token,
    mint_session_refresh_token,
    mint_session_token,
    open_session_refresh_token,
    open_session_token,
)

NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
KEYS = SessionKeys(signing_key=SecretStr("k" * 32))
OTHER_KEYS = SessionKeys(signing_key=SecretStr("x" * 32))
PRINCIPAL = SessionPrincipal(user_id="user-123", client_id="llm_client_abc")


def _mint_access() -> str:
    minted = mint_session_token(PRINCIPAL, KEYS, NOW)
    assert isinstance(minted, MintedSessionToken)
    return minted.token.get_secret_value()


def _mint_refresh() -> str:
    minted = mint_session_refresh_token(PRINCIPAL, KEYS, NOW)
    assert isinstance(minted, MintedSessionToken)
    return minted.token.get_secret_value()


def _sign_claims(payload: dict, prefix: str = SESSION_TOKEN_PREFIX, keys: SessionKeys = KEYS) -> str:
    return prefix + jwt.encode(payload, keys.signing_key.get_secret_value(), algorithm="HS256")


def _valid_claims(**overrides) -> dict:
    base = {
        "iss": SESSION_ISSUER,
        "iat": int(NOW.timestamp()),
        "exp": int((NOW + timedelta(seconds=600)).timestamp()),
        "kind": "session",
        "user_id": "user-123",
        "client_id": "llm_client_abc",
    }
    return {**base, **overrides}


def test_access_round_trip_recovers_principal_and_caps_ttl():
    minted = mint_session_token(PRINCIPAL, KEYS, NOW)
    assert isinstance(minted, MintedSessionToken)
    assert minted.expires_at == NOW + timedelta(seconds=SESSION_TTL_SECONDS)
    token = minted.token.get_secret_value()
    assert is_session_token(token)
    assert not is_session_refresh_token(token)
    opened = open_session_token(token, KEYS, NOW)
    assert isinstance(opened, OpenedSessionToken)
    assert opened.principal == PRINCIPAL


def test_refresh_round_trip_recovers_principal_and_caps_ttl():
    minted = mint_session_refresh_token(PRINCIPAL, KEYS, NOW)
    assert isinstance(minted, MintedSessionToken)
    assert minted.expires_at == NOW + timedelta(seconds=SESSION_REFRESH_TTL_SECONDS)
    token = minted.token.get_secret_value()
    assert is_session_refresh_token(token)
    opened = open_session_refresh_token(token, KEYS, NOW)
    assert isinstance(opened, OpenedSessionToken)
    assert opened.principal == PRINCIPAL


def test_access_token_reprefixed_as_refresh_is_rejected_by_signed_kind():
    body = _mint_access().removeprefix(SESSION_TOKEN_PREFIX)
    swapped = SESSION_REFRESH_PREFIX + body
    assert isinstance(open_session_refresh_token(swapped, KEYS, NOW), SessionMalformed)


def test_refresh_token_reprefixed_as_access_is_rejected_by_signed_kind():
    body = _mint_refresh().removeprefix(SESSION_REFRESH_PREFIX)
    swapped = SESSION_TOKEN_PREFIX + body
    assert isinstance(open_session_token(swapped, KEYS, NOW), SessionMalformed)


def test_refresh_token_is_not_an_access_token_at_the_edge():
    assert isinstance(open_session_token(_mint_refresh(), KEYS, NOW), NotASessionToken)


def test_expired_access_token_is_expired_not_malformed():
    token = _mint_access()
    at_expiry = NOW + timedelta(seconds=SESSION_TTL_SECONDS)
    assert isinstance(open_session_token(token, KEYS, at_expiry), SessionExpired)
    after = NOW + timedelta(seconds=SESSION_TTL_SECONDS + 1)
    assert isinstance(open_session_token(token, KEYS, after), SessionExpired)


def test_still_valid_one_second_before_expiry():
    token = _mint_access()
    just_before = NOW + timedelta(seconds=SESSION_TTL_SECONDS - 1)
    assert isinstance(open_session_token(token, KEYS, just_before), OpenedSessionToken)


def test_tampered_signature_is_bad_signature():
    token = _mint_access()
    tampered = token[:-2] + ("aa" if not token.endswith("aa") else "bb")
    assert isinstance(open_session_token(tampered, KEYS, NOW), SessionBadSignature)


def test_key_rotation_invalidates_outstanding_tokens():
    token = _mint_access()
    assert isinstance(open_session_token(token, OTHER_KEYS, NOW), SessionBadSignature)


@pytest.mark.parametrize(
    "candidate,expected",
    [
        ("sk-1234", NotASessionToken),
        ("llm_env_something", NotASessionToken),
        ("", NotASessionToken),
        (SESSION_TOKEN_PREFIX, SessionMalformed),
        (SESSION_TOKEN_PREFIX + "not-a-jwt", SessionMalformed),
        (SESSION_TOKEN_PREFIX + "\ud800garbage", SessionMalformed),
        (SESSION_TOKEN_PREFIX + "a" * (MAX_SESSION_TOKEN_BYTES + 1), SessionMalformed),
    ],
)
def test_hostile_candidates_never_raise(candidate, expected):
    assert isinstance(open_session_token(candidate, KEYS, NOW), expected)


def test_multibyte_candidate_over_byte_cap_but_under_char_cap_is_rejected():
    filler = "€" * (MAX_SESSION_TOKEN_BYTES // 3)
    candidate = SESSION_TOKEN_PREFIX + filler
    assert len(candidate) <= MAX_SESSION_TOKEN_BYTES
    assert isinstance(open_session_token(candidate, KEYS, NOW), SessionMalformed)


def test_alg_none_token_is_rejected():
    unsigned = jwt.api_jws.encode(b'{"iss":"litellm-mcp-gateway"}', key=None, algorithm="none")
    assert isinstance(open_session_token(SESSION_TOKEN_PREFIX + unsigned, KEYS, NOW), SessionMalformed)


@pytest.mark.parametrize(
    "claims",
    [
        _valid_claims(iss="wrong-issuer"),
        _valid_claims(exp=str(int((NOW + timedelta(seconds=600)).timestamp()))),
        _valid_claims(iat="evil"),
        _valid_claims(kind="access"),
        _valid_claims(user_id=""),
        _valid_claims(nbf=0),
        {k: v for k, v in _valid_claims().items() if k != "client_id"},
        {k: v for k, v in _valid_claims().items() if k != "exp"},
    ],
)
def test_signed_but_malformed_claims_are_rejected_without_raising(claims):
    token = _sign_claims(claims)
    assert isinstance(open_session_token(token, KEYS, NOW), SessionMalformed)


def test_signed_claims_with_exact_shape_open():
    token = _sign_claims(_valid_claims())
    opened = open_session_token(token, KEYS, NOW)
    assert isinstance(opened, OpenedSessionToken)
    assert opened.principal.user_id == "user-123"


def test_oversized_client_id_fails_mint_with_typed_error_not_truncation():
    principal = SessionPrincipal(user_id="user-123", client_id="c" * (MAX_SESSION_TOKEN_BYTES + 100))
    minted = mint_session_token(principal, KEYS, NOW)
    assert isinstance(minted, SessionTokenTooLarge)
    assert minted.max_bytes == MAX_SESSION_TOKEN_BYTES


def test_empty_principal_fields_rejected_at_construction():
    with pytest.raises(ValidationError):
        SessionPrincipal(user_id="", client_id="c")
    with pytest.raises(ValidationError):
        SessionPrincipal(user_id="u", client_id="")


def test_short_signing_key_rejected_at_construction():
    with pytest.raises(ValidationError):
        SessionKeys(signing_key=SecretStr("short"))


def test_minted_token_repr_never_leaks_value():
    minted = mint_session_token(PRINCIPAL, KEYS, NOW)
    assert isinstance(minted, MintedSessionToken)
    assert minted.token.get_secret_value() not in repr(minted)
